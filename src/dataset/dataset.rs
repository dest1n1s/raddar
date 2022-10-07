use std::{cmp::min, sync::Arc};

#[derive(Debug)]
pub struct SimpleDataset<InputType, LabelType> {
    pub inputs: Vec<Arc<InputType>>,
    pub labels: Vec<Arc<LabelType>>,
    pub size: usize,
    pub batch_size: usize,
}

pub struct UnsupervisedDataset<InputType> {
    pub inputs: Vec<Arc<InputType>>,
    pub size: usize,
    pub batch_size: usize,
}

pub trait Dataset
where
    Self: Sized,
{
    type InputType;
    type LabelType;
    type BatchType;
    fn iter(&self) -> DatasetIterator<Self>;
    fn get_inputs(&self) -> &Vec<Arc<Self::InputType>>;
    fn get_labels(&self) -> Option<&Vec<Arc<Self::LabelType>>>;
    fn get_size(&self) -> usize;
    fn get_batch_size(&self) -> usize;
    fn process_batch(
        &self,
        inputs: Vec<Arc<Self::InputType>>,
        labels: Option<Vec<Arc<Self::LabelType>>>,
    ) -> Self::BatchType;
}

impl<T, U> Dataset for SimpleDataset<T, U> {
    type InputType = T;
    type LabelType = U;
    type BatchType = (Vec<Arc<T>>, Vec<Arc<U>>);

    fn iter(&self) -> DatasetIterator<Self> {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }

    fn get_inputs(&self) -> &Vec<Arc<Self::InputType>> {
        &self.inputs
    }

    fn get_labels(&self) -> Option<&Vec<Arc<Self::LabelType>>> {
        Some(&self.labels)
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    fn process_batch(
        &self,
        inputs: Vec<Arc<Self::InputType>>,
        labels: Option<Vec<Arc<Self::LabelType>>>,
    ) -> Self::BatchType {
        (inputs, labels.unwrap())
    }
}

impl<T> Dataset for UnsupervisedDataset<T> {
    type InputType = T;
    type LabelType = ();
    type BatchType = Vec<Arc<T>>;

    fn iter(&self) -> DatasetIterator<Self> {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }

    fn get_inputs(&self) -> &Vec<Arc<Self::InputType>> {
        &self.inputs
    }

    fn get_labels(&self) -> Option<&Vec<Arc<Self::LabelType>>> {
        None
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    fn process_batch(
        &self,
        inputs: Vec<Arc<Self::InputType>>,
        _labels: Option<Vec<Arc<Self::LabelType>>>,
    ) -> Self::BatchType {
        inputs
    }
}

impl<V, U> SimpleDataset<V, U> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, labels: Vec<Arc<U>>, batch_size: usize) -> Self {
        let size = inputs.len();
        Self {
            inputs,
            labels,
            size,
            batch_size,
        }
    }
}

impl<V> UnsupervisedDataset<V> {
    pub fn from_vectors(inputs: Vec<Arc<V>>, batch_size: usize) -> Self {
        let size = inputs.len();
        Self {
            inputs,
            size,
            batch_size,
        }
    }
}

pub struct DatasetIterator<'a, T: Dataset> {
    pub dataset: &'a T,
    pub index: usize,
}

impl<'a, T: Dataset> Iterator for DatasetIterator<'a, T> {
    type Item = T::BatchType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.get_size() {
            return None;
        }
        let end = min(
            self.index + self.dataset.get_batch_size(),
            self.dataset.get_size(),
        );
        let batch = self.dataset.get_inputs()[self.index..end].to_vec();
        let batch_labels = if let Some(labels) = self.dataset.get_labels() {
            Some(labels[self.index..end].to_vec())
        } else {
            None
        };
        self.index = end;
        Some(self.dataset.process_batch(batch, batch_labels))
    }
}
