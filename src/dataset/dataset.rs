use std::{cmp::min, sync::Arc};

use raddar_derive::IterableDataset;

#[derive(Debug, Clone)]
#[derive(IterableDataset)]
pub struct SimpleDataset<InputType, LabelType> {
    pub inputs: Vec<Arc<InputType>>,
    pub labels: Vec<Arc<LabelType>>,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
#[derive(IterableDataset)]
pub struct UnsupervisedDataset<InputType> {
    pub inputs: Vec<Arc<InputType>>,
    pub batch_size: usize,
}

pub trait Dataset: IntoIterator
where
    Self: Sized,
{
    type DataType: Clone;
    type BatchType;

    fn data(self) -> Vec<Self::DataType>;
    fn size(&self) -> usize;
    fn batch_size(&self) -> usize;
    fn collate(data: Vec<Self::DataType>) -> Self::BatchType;
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

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn collate(data: Vec<Self::DataType>) -> Self::BatchType {
        data.into_iter().map(|x| (x.0, x.1)).unzip()
    }
}

impl<T> Dataset for UnsupervisedDataset<T> {
    type DataType = Arc<T>;
    type BatchType = Vec<Arc<T>>;

    fn data(self) -> Vec<Self::DataType> {
        self.inputs
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn collate(data: Vec<Self::DataType>) -> Self::BatchType {
        data
    }
}

// impl<T, U> IntoIterator for SimpleDataset<T, U> {
//     type Item = <Self as Dataset>::BatchType;
//     type IntoIter = DatasetIterator<Self>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.data().into_iter()
//     }
// }

// impl<T> IntoIterator for UnsupervisedDataset<T> {
//     type Item = <Self as Dataset>::BatchType;
//     type IntoIter = DatasetIterator<Self>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.data().into_iter()
//     }
// }

impl<V, U> SimpleDataset<V, U> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, labels: Vec<Arc<U>>, batch_size: usize) -> Self {
        Self {
            inputs,
            labels,
            batch_size,
        }
    }
}

impl<V> UnsupervisedDataset<V> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, batch_size: usize) -> Self {
        Self {
            inputs,
            batch_size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatasetIterator<T: Dataset> {
    pub data: Vec<T::DataType>,
    pub index: usize,
    pub batch_size: usize,
}

impl<T: Dataset> DatasetIterator<T> {
    pub fn new(data: Vec<T::DataType>, batch_size: usize) -> Self {
        Self { data, index: 0, batch_size }
    }
}

impl<T: Dataset> Iterator for DatasetIterator<T> {
    type Item = T::BatchType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }

        let batch_size = min(self.batch_size, self.data.len() - self.index);
        let batch = self.data[self.index..self.index + batch_size].to_vec();
        self.index += batch_size;

        Some(T::collate(batch))
    }
}
