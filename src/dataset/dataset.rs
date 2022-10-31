use std::{cmp::min, sync::Arc};

use derive_builder::Builder;
use pariter::IteratorExt;
use raddar_derive::{DatasetFromIter, DatasetIntoIter};
use rand::seq::SliceRandom;

/// A simple dataset with a vector of inputs and a vector of targets.
#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct SimpleDataset<InputType: Send + Sync + 'static, LabelType: Send + Sync + 'static> {
    pub inputs: Vec<Arc<InputType>>,
    pub labels: Vec<Arc<LabelType>>,
}

/// An unlabelled dataset with a vector of inputs.
#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct UnsupervisedDataset<InputType: Send + Sync + 'static> {
    pub inputs: Vec<Arc<InputType>>,
}

/// A trait for datasets.
pub trait Dataset:
    IntoIterator<Item = Self::SampleType>
    + FromIterator<Self::BatchType>
    + FromIterator<Self::SampleType>
where
    Self: Sized,
{
    type SampleType: Clone + Send + Sync + 'static;
    type BatchType: Send + Sync + 'static;

    /// Returns all the samples in the dataset.
    fn data(self) -> Vec<Self::SampleType>;

    /// Creates a dataset from an collection of samples.
    fn from_data<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self {
        Self::from_batches(vec![Self::collate(data)])
    }

    /// Creates a dataset from an collection of batches.
    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self;

    /// Creates a `DataLoader` from the dataset with the given configuration.
    fn into_loader(self, cfg: DataLoaderConfig) -> DataLoader<Self> {
        DataLoader::new(self.data(), cfg)
    }

    /// Gets the number of samples in the dataset.
    fn size(&self) -> usize;

    /// Defines how to collate a collection of samples into a batch.
    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType;

    /// Maps the dataset to a new dataset using the given mapping function.
    /// 
    /// If you want to chain mapping operations(which is common in image transformation), it is recommended to use the `map` method from the `Iterator` trait(or the `parallel_map` method from the `pariter::IteratorExt` trait) instead, to avoid unnecessary cost.
    /// 
    /// # Examples
    /// ```
    /// let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    /// let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    /// let dataset = TensorDataset::from_tensors(inputs, labels);
    /// let mapped_dataset: TensorDataset = dataset.map(|(x, y): (Arc<Tensor>, Arc<Tensor>)| (Arc::new(x.copy() * 2.0), Arc::new(y.copy() * 2.0)));
    /// ```
    /// 
    /// When chaining mapping operations:
    /// ```
    /// let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    /// let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    /// let dataset = TensorDataset::from_tensors(inputs, labels);
    /// let mapped_dataset: TensorDataset = dataset.into_iter().parallel_map(|(x, y): (Arc<Tensor>, Arc<Tensor>)| (Arc::new(x.copy() * 2.0), Arc::new(y.copy() * 2.0)))
    ///    .parallel_map(|(x, y): (Arc<Tensor>, Arc<Tensor>)| (Arc::new(x.copy() + 1.0), Arc::new(y.copy() + 1.0))).collect();
    /// ```
    fn map<T, F>(self, f: F) -> T
    where
        Self: 'static,
        T: Dataset,
        F: FnMut(Self::SampleType) -> T::SampleType + Send + Clone + 'static,
    {
        self.into_iter().parallel_map(f).collect()
    }

    /// Maps the dataset to a new dataset in batch using the given batch mapping function.
    /// 
    /// If you want to chain mapping operations(which is common in image transformation), it is recommended to make the dataset into [`DataLoader`], and use the `map` method from the `Iterator` trait(or the `parallel_map` method from the `pariter::IteratorExt` trait) instead, to avoid unnecessary cost.
    /// 
    /// # Examples
    /// ```
    /// let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    /// let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    /// let dataset = TensorDataset::from_tensors(inputs, labels);
    /// let mapped_dataset: TensorDataset = dataset.map_batch(|(x, y): (Tensor, Tensor)| (x * 2.0, y * 2.0), 3);
    /// ```
    /// 
    /// When chaining mapping operations:
    /// ```
    /// let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    /// let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    /// let dataset = TensorDataset::from_tensors(inputs, labels);
    /// let mapped_dataset: TensorDataset = dataset.into_loader(
    ///                                         DataLoaderConfigBuilder::default().batch_size(3).build().unwrap()
    ///                                     )
    ///                                     .parallel_map(|(x, y): (Tensor, Tensor)| (x * 2.0, y * 2.0))
    ///                                     .parallel_map(|(x, y): (Tensor, Tensor)| (x + 1, y + 1))
    ///                                     .collect();
    /// ```
    fn map_batch<T, F>(self, f: F, batch_size: usize) -> T
    where
        Self: 'static,
        T: Dataset,
        F: FnMut(Self::BatchType) -> T::BatchType + Send + Clone + 'static,
    {
        self.into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(batch_size)
                .build()
                .unwrap(),
        )
        .parallel_map(f)
        .collect()
    }

    /// Augments the dataset with the given augmentation function.
    /// 
    /// This method is similar to the `map` method, but maps each sample in the dataset to a collection of samples, and then flattens the collection into a new dataset.
    fn augment<T, F, C>(self, f: F) -> T
    where
        Self: 'static,
        T: Dataset,
        C: IntoIterator<Item = T::SampleType> + Send + 'static,
        F: FnMut(Self::SampleType) -> C + Send + Clone + 'static,
    {
        self.into_iter().parallel_map(f).flatten().collect()
    }
}

impl<T: Send + Sync, U: Send + Sync> Dataset for SimpleDataset<T, U> {
    type SampleType = (Arc<T>, Arc<U>);
    type BatchType = (Vec<Arc<T>>, Vec<Arc<U>>);

    fn data(self) -> Vec<Self::SampleType> {
        self.inputs
            .into_iter()
            .zip(self.labels.into_iter())
            .collect()
    }

    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self {
        let mut inputs = Vec::new();
        let mut labels = Vec::new();
        for (batch_inputs, batch_labels) in batches {
            inputs.extend(batch_inputs);
            labels.extend(batch_labels);
        }
        Self { inputs, labels }
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType {
        data.into_iter().map(|x| (x.0, x.1)).unzip()
    }
}

impl<T: Send + Sync> Dataset for UnsupervisedDataset<T> {
    type SampleType = Arc<T>;
    type BatchType = Vec<Arc<T>>;

    fn data(self) -> Vec<Self::SampleType> {
        self.inputs
    }

    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self {
        let mut inputs = Vec::new();
        for batch_inputs in batches {
            inputs.extend(batch_inputs);
        }
        Self { inputs }
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType {
        data.into_iter().collect()
    }
}

impl<V: Send + Sync, U: Send + Sync> SimpleDataset<V, U> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, labels: Vec<Arc<U>>) -> Self {
        Self { inputs, labels }
    }
}

impl<V: Send + Sync> UnsupervisedDataset<V> {
    pub fn from_vectors(inputs: Vec<Arc<V>>) -> Self {
        Self { inputs }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetIterator<T: Dataset> {
    pub data: Vec<T::SampleType>,
    pub index: usize,
}

impl<T: Dataset> DatasetIterator<T> {
    pub fn new(data: Vec<T::SampleType>) -> Self {
        Self { data, index: 0 }
    }
}

impl<T: Dataset> Iterator for DatasetIterator<T> {
    type Item = T::SampleType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let item = self.data[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// The configuration for a `DataLoader`.
#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned")]
pub struct DataLoaderConfig {
    #[builder(default = "1")]
    pub batch_size: usize,

    #[builder(default = "false")]
    pub shuffle: bool,
}

/// A data loader iterates over `Dataset`, which can be used to load data in batches.
#[derive(Debug)]
pub struct DataLoader<T: Dataset> {
    pub data: Vec<T::SampleType>,
    pub cfg: DataLoaderConfig,
    pub index: usize,
}

impl<T: Dataset> DataLoader<T> {
    pub fn new(data: Vec<T::SampleType>, cfg: DataLoaderConfig) -> Self {
        let mut this = Self {
            data,
            cfg,
            index: 0,
        };
        if this.cfg.shuffle {
            this.shuffle();
        }
        this
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        self.data.shuffle(&mut rng);
    }
}

impl<T: Dataset> Clone for DataLoader<T> {
    fn clone(&self) -> Self {
        let mut that = Self {
            data: self.data.clone(),
            cfg: self.cfg.clone(),
            index: self.index,
        };
        if that.cfg.shuffle {
            that.shuffle();
        }
        that
    }
}

impl<T: Dataset> Iterator for DataLoader<T> {
    type Item = T::BatchType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let batch_size = min(self.cfg.batch_size, self.data.len() - self.index);
        let batch = self.data[self.index..self.index + batch_size].to_vec();
        self.index += batch_size;

        Some(T::collate(batch))
    }
}
