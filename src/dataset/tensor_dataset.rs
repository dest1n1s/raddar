use std::{collections::HashMap, sync::Arc};

use raddar_derive::{DatasetFromIter, DatasetIntoIter};
use tch::Tensor;

use crate::core::TensorIntoIter;

use super::{Dataset, SimpleDataset, UnsupervisedDataset};

/// A tensor dataset where the inputs and labels are all tensors.
///
/// This is the most common type of dataset used in machine learning.
///
/// Where this dataset differs from the `SimpleDataset<Tensor, Tensor>` is that the `BatchType` is a tuple of tensors, rather than a tuple of `Vec<Arc<Tensor>>`.
#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter, Default)]
pub struct TensorDataset {
    pub inputs: Vec<Arc<Tensor>>,
    pub labels: Vec<Arc<Tensor>>,
}

/// An unlabelled tensor dataset where the inputs are all tensors.
///
/// This is the most common type of dataset used in unsupervised machine learning.
///
/// Where this dataset differs from the `UnsupervisedDataset<Tensor>` is that the `BatchType` is `Tensor`, rather than `Vec<Arc<Tensor>>`.
#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct UnsupervisedTensorDataset {
    pub inputs: Vec<Arc<Tensor>>,
}

/// A dataset containing a dictionary of tensors.
#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct DictTensorDataset {
    pub inputs: HashMap<String, Vec<Arc<Tensor>>>,
}

impl Dataset for TensorDataset {
    type SampleType = (Arc<Tensor>, Arc<Tensor>);
    type BatchType = (Tensor, Tensor);

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
            inputs.extend(batch_inputs.into_iter().map(Arc::new));
            labels.extend(batch_labels.into_iter().map(Arc::new));
        }
        Self { inputs, labels }
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType {
        let (inputs, labels): (Vec<_>, Vec<_>) = data.into_iter().unzip();
        let inputs = Tensor::stack(&inputs, 0);
        let labels = Tensor::stack(&labels, 0);
        (inputs, labels)
    }
}

impl Dataset for UnsupervisedTensorDataset {
    type SampleType = Arc<Tensor>;
    type BatchType = Tensor;

    fn data(self) -> Vec<Self::SampleType> {
        self.inputs
    }

    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self {
        let mut inputs = Vec::new();
        for batch_inputs in batches {
            inputs.extend(batch_inputs.into_iter().map(Arc::new));
        }
        Self { inputs }
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType {
        Tensor::stack(&data.into_iter().collect::<Vec<_>>(), 0)
    }
}

impl Dataset for DictTensorDataset {
    type SampleType = HashMap<String, Arc<Tensor>>;
    type BatchType = HashMap<String, Tensor>;

    fn data(self) -> Vec<Self::SampleType> {
        let mut data = Vec::new();
        for i in 0..self.inputs.len() {
            let mut input = HashMap::new();
            for (key, values) in self.inputs.iter() {
                input.insert(key.clone(), values[i].clone());
            }
            data.push(input);
        }
        data
    }

    fn from_batches<I: IntoIterator<Item = Self::BatchType>>(batches: I) -> Self {
        let mut inputs = HashMap::new();
        for batch in batches {
            for (key, value) in batch {
                inputs
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(Arc::new(value));
            }
        }
        Self { inputs }
    }

    fn size(&self) -> usize {
        self.inputs.values().next().unwrap().len()
    }

    fn collate<I: IntoIterator<Item = Self::SampleType>>(data: I) -> Self::BatchType {
        let mut batch = HashMap::new();
        for data in data {
            for (key, value) in data {
                batch.entry(key).or_insert_with(Vec::new).push(value);
            }
        }
        let batch = batch
            .into_iter()
            .map(|(key, values)| {
                (
                    key,
                    Tensor::stack(&values.into_iter().collect::<Vec<_>>(), 0),
                )
            })
            .collect();
        batch
    }
}

impl TensorDataset {
    /// Creates a new `TensorDataset` from the given inputs and labels.
    pub fn from_tensors(inputs: Vec<Arc<Tensor>>, labels: Vec<Arc<Tensor>>) -> Self {
        Self { inputs, labels }
    }
}

impl UnsupervisedTensorDataset {
    /// Creates a new `UnsupervisedTensorDataset` from the given inputs.
    pub fn from_tensors(inputs: Vec<Arc<Tensor>>) -> Self {
        Self { inputs }
    }
}

impl DictTensorDataset {
    /// Creates a new `DictTensorDataset` from the given inputs.
    pub fn from_tensors(inputs: HashMap<String, Vec<Arc<Tensor>>>) -> Self {
        Self { inputs }
    }
}

impl From<SimpleDataset<Tensor, Tensor>> for TensorDataset {
    fn from(dataset: SimpleDataset<Tensor, Tensor>) -> Self {
        dataset.map(|data| data)
    }
}

impl From<UnsupervisedDataset<Tensor>> for UnsupervisedTensorDataset {
    fn from(dataset: UnsupervisedDataset<Tensor>) -> Self {
        dataset.map(|data| data)
    }
}
