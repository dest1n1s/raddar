use std::{cmp::min, sync::Arc};

use derive_builder::Builder;
use raddar_derive::{DatasetFromIter, DatasetIntoIter};
use rand::seq::SliceRandom;

#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct SimpleDataset<InputType, LabelType> {
    pub inputs: Vec<Arc<InputType>>,
    pub labels: Vec<Arc<LabelType>>,
}

#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct UnsupervisedDataset<InputType> {
    pub inputs: Vec<Arc<InputType>>,
}

pub enum DatasetMapping<'a, T1: Dataset, T2: Dataset> {
    BatchMapping(&'a mut dyn FnMut(T1::BatchType) -> T2::BatchType, usize),
    DataMapping(&'a mut dyn FnMut(T1::DataType) -> T2::DataType),
}

pub trait Dataset:
    IntoIterator<Item = Self::DataType> + FromIterator<Self::BatchType> + FromIterator<Self::DataType>
where
    Self: Sized,
{
    type DataType: Clone;
    type BatchType;

    fn data(self) -> Vec<Self::DataType>;

    fn from_data<I: IntoIterator<Item = Self::DataType>>(data: I) -> Self {
        Self::from_batches(vec![Self::collate(data)])
    }

    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self;

    fn into_loader(self, cfg: DataLoaderConfig) -> DataLoader<Self> {
        DataLoader::new(self.data(), cfg)
    }

    fn size(&self) -> usize;
    fn collate<I: IntoIterator<Item = Self::DataType>>(data: I) -> Self::BatchType;

    fn map<T: Dataset>(self, mapping: DatasetMapping<Self, T>) -> T {
        match mapping {
            DatasetMapping::BatchMapping(f, batch_size) => {
                self.into_loader(DataLoaderConfigBuilder::default().batch_size(batch_size).build().unwrap()).map(|batch| f(batch)).collect()
            }
            DatasetMapping::DataMapping(f) => self.into_iter().map(|data| f(data)).collect(),
        }
    }
}

impl<T, U> Dataset for SimpleDataset<T, U> {
    type DataType = (Arc<T>, Arc<U>);
    type BatchType = (Vec<Arc<T>>, Vec<Arc<U>>);

    fn data(self) -> Vec<Self::DataType> {
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

    fn collate<I: IntoIterator<Item = Self::DataType>>(data: I) -> Self::BatchType {
        data.into_iter().map(|x| (x.0, x.1)).unzip()
    }
}

impl<T> Dataset for UnsupervisedDataset<T> {
    type DataType = Arc<T>;
    type BatchType = Vec<Arc<T>>;

    fn data(self) -> Vec<Self::DataType> {
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

    fn collate<I: IntoIterator<Item = Self::DataType>>(data: I) -> Self::BatchType {
        data.into_iter().collect()
    }
}

impl<V, U> SimpleDataset<V, U> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, labels: Vec<Arc<U>>) -> Self {
        Self { inputs, labels }
    }
}

impl<V> UnsupervisedDataset<V> {
    pub fn from_vectors(inputs: Vec<Arc<V>>) -> Self {
        Self { inputs }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetIterator<T: Dataset> {
    pub data: Vec<T::DataType>,
    pub index: usize,
}

impl<T: Dataset> DatasetIterator<T> {
    pub fn new(data: Vec<T::DataType>) -> Self {
        Self { data, index: 0 }
    }
}

impl<T: Dataset> Iterator for DatasetIterator<T> {
    type Item = T::DataType;

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

#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned")]
pub struct DataLoaderConfig {
    #[builder(default = "1")]
    pub batch_size: usize,

    #[builder(default = "true")]
    pub shuffle: bool,
}

#[derive(Debug)]
pub struct DataLoader<T: Dataset> {
    pub data: Vec<T::DataType>,
    pub cfg: DataLoaderConfig,
    pub index: usize,
}

impl<T: Dataset> DataLoader<T> {
    pub fn new(data: Vec<T::DataType>, cfg: DataLoaderConfig) -> Self {
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
