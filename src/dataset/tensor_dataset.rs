use std::sync::Arc;

use raddar_derive::{DatasetFromIter, DatasetIntoIter};
use tch::Tensor;

use crate::core::TensorIntoIter;

use super::Dataset;

#[derive(Debug, Clone, DatasetIntoIter, DatasetFromIter)]
pub struct TensorDataset {
    pub inputs: Vec<Arc<Tensor>>,
    pub labels: Vec<Arc<Tensor>>,
}

impl Dataset for TensorDataset {
    type DataType = (Arc<Tensor>, Arc<Tensor>);
    type BatchType = (Tensor, Tensor);

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
            inputs.extend(batch_inputs.into_iter().map(Arc::new));
            labels.extend(batch_labels.into_iter().map(Arc::new));
        }
        Self { inputs, labels }
    }

    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn collate<I: IntoIterator<Item = Self::DataType>>(data: I) -> Self::BatchType {
        let (inputs, labels): (Vec<_>, Vec<_>) = data.into_iter().unzip();
        let inputs = Tensor::stack(&inputs, 0);
        let labels = Tensor::stack(&labels, 0);
        (inputs, labels)
    }
}

impl TensorDataset {
    pub fn from_tensors(inputs: Vec<Arc<Tensor>>, labels: Vec<Arc<Tensor>>) -> Self {
        Self { inputs, labels }
    }
}
