use std::{cmp::min, marker::PhantomData, sync::Arc};

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

/// A mapping function used to transform the data in a dataset.
///
/// The mapping has 2 variants:
/// `BatchFunc` is used to transform batched data, which can potentially be more efficient.
/// `SampleFunc` is used to transform individual samples.
pub enum DatasetMapping<
    DatasetFrom: Dataset,
    DatasetTo: Dataset,
    BatchFunc: FnMut(DatasetFrom::BatchType) -> DatasetTo::BatchType + Send + Clone,
    SampleFunc: FnMut(DatasetFrom::SampleType) -> DatasetTo::SampleType + Send + Clone,
> {
    BatchMapping(BatchFunc, usize),
    DataMapping(SampleFunc),
    _Phantom(PhantomData<(DatasetFrom, DatasetTo)>),
}

pub type DatasetBatchMapping<
    DatasetFrom: Dataset,
    DatasetTo: Dataset,
    BatchFunc: FnMut(DatasetFrom::BatchType) -> DatasetTo::BatchType + Send + Clone,
> = DatasetMapping<
    DatasetFrom,
    DatasetTo,
    BatchFunc,
    fn(DatasetFrom::SampleType) -> DatasetTo::SampleType,
>;

pub type DatasetSampleMapping<
    DatasetFrom: Dataset,
    DatasetTo: Dataset,
    DataFunc: FnMut(DatasetFrom::SampleType) -> DatasetTo::SampleType + Send + Clone,
> = DatasetMapping<
    DatasetFrom,
    DatasetTo,
    fn(DatasetFrom::BatchType) -> DatasetTo::BatchType,
    DataFunc,
>;

/// Create a `DatasetBatchMapping` from a batch mapping function.
pub fn batch_mapping<
    T1: Dataset,
    T2: Dataset,
    F: FnMut(T1::BatchType) -> T2::BatchType + Send + Clone,
>(
    batch_func: F,
    batch_size: usize,
) -> DatasetBatchMapping<T1, T2, F> {
    DatasetMapping::BatchMapping(batch_func, batch_size)
}

/// Create a `DatasetSampleMapping` from a sample mapping function.
pub fn sample_mapping<
    T1: Dataset,
    T2: Dataset,
    F: FnMut(T1::SampleType) -> T2::SampleType + Send + Clone,
>(
    sample_func: F,
) -> DatasetSampleMapping<T1, T2, F> {
    DatasetMapping::DataMapping(sample_func)
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
    fn map<T, F1, F2>(self, mapping: DatasetMapping<Self, T, F1, F2>) -> T
    where
        Self: 'static,
        T: Dataset,
        F1: FnMut(Self::BatchType) -> T::BatchType + Send + Clone + 'static,
        F2: FnMut(Self::SampleType) -> T::SampleType + Send + Clone + 'static,
    {
        match mapping {
            DatasetMapping::BatchMapping(f, batch_size) => self
                .into_loader(
                    DataLoaderConfigBuilder::default()
                        .batch_size(batch_size)
                        .build()
                        .unwrap(),
                )
                .parallel_map(f)
                .collect(),
            DatasetMapping::DataMapping(f) => self.into_iter().parallel_map(f).collect(),
            DatasetMapping::_Phantom(..) => unreachable!(),
        }
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
