use std::{collections::HashMap, sync::Arc};

use raddar_derive::{DatasetFromIter, DatasetIntoIter};
use tch::{Device, Tensor};

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
        // Assert that the inputs and labels have the same length.
        assert_eq!(inputs.len(), labels.len(), "The number of inputs and labels must be the same.");

        // Assert that the inputs and labels are all on the same device.
        let device = inputs[0].device();
        for input in &inputs {
            assert_eq!(input.device(), device, "All inputs and labels must be on the same device.");
        }
        for label in &labels {
            assert_eq!(label.device(), device, "All inputs and labels must be on the same device.");
        }

        // Assert that the inputs and labels are respectively all the same shape.
        let input_shape = inputs[0].size();
        for input in &inputs {
            assert_eq!(input.size(), input_shape, "All inputs must be the same shape.");
        }
        let label_shape = labels[0].size();
        for label in &labels {
            assert_eq!(label.size(), label_shape, "All labels must be the same shape.");
        }
        
        Self { inputs, labels }
    }

    /// Move all tensor to the specific device.
    pub fn to(self, device: Device) -> Self
    where
        Self: Sized,
    {
        self.map(move |(input, label): (Arc<Tensor>, Arc<Tensor>)| {
            (Arc::new(input.to(device)), Arc::new(label.to(device)))
        })
    }
}

impl UnsupervisedTensorDataset {
    /// Creates a new `UnsupervisedTensorDataset` from the given inputs.
    pub fn from_tensors(inputs: Vec<Arc<Tensor>>) -> Self {
        // Assert that all inputs have the same shape.
        let shape = inputs[0].size();
        for input in &inputs {
            assert_eq!(input.size(), shape, "All inputs must have the same shape.");
        }

        // Assert that all inputs are on the same device.
        let device = inputs[0].device();
        for input in &inputs {
            assert_eq!(input.device(), device, "All inputs must be on the same device.");
        }

        Self { inputs }
    }

    /// Move all tensor to the specific device.
    pub fn to(self, device: Device) -> Self
    where
        Self: Sized,
    {
        self.map(move |input: Arc<Tensor>| Arc::new(input.to(device)))
    }
}

impl DictTensorDataset {
    /// Creates a new `DictTensorDataset` from the given inputs.
    pub fn from_tensors(inputs: HashMap<String, Vec<Arc<Tensor>>>) -> Self {
        // Assert that all fields have the same length.
        let length = inputs.values().next().unwrap().len();
        for values in inputs.values() {
            assert_eq!(values.len(), length, "All fields must have the same length.");
        }

        // Assert that all tensors are on the same device.
        let device = inputs.values().next().unwrap()[0].device();
        for values in inputs.values() {
            for value in values {
                assert_eq!(value.device(), device, "All tensors must be on the same device.");
            }
        }

        // Assert that all tensors in a field have the same shape.
        for (key, values) in inputs.iter() {
            let shape = values[0].size();
            for value in values {
                assert_eq!(value.size(), shape, "Field {} has tensors with different shapes.", key);
            }
        }
        
        Self { inputs }
    }

    /// Move all tensor to the specific device.
    pub fn to(self, device: Device) -> Self
    where
        Self: Sized,
    {
        self.map(move |input: HashMap<String, Arc<Tensor>>| {
            input
                .into_iter()
                .map(|(key, value)| (key, Arc::new(value.to(device))))
                .collect()
        })
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
