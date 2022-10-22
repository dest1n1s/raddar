use std::sync::Arc;

use raddar_derive::IterableDataset;
use tch::Tensor;

use super::Dataset;

#[derive(Debug, Clone)]
#[derive(IterableDataset)]
pub struct TensorDataset {
    pub inputs: Vec<Arc<Tensor>>,
    pub labels: Vec<Arc<Tensor>>,
    pub size: usize,
    pub batch_size: usize,
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

    fn size(&self) -> usize {
        self.size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn collate(data: Vec<Self::DataType>) -> Self::BatchType {
        let (inputs, labels): (Vec<_>, Vec<_>) = data.into_iter().map(|x| (x.0, x.1)).unzip();
        let inputs = Tensor::stack(&inputs, 0);
        let labels = Tensor::stack(&labels, 0);
        (inputs, labels)
    }
}

impl TensorDataset {
    pub fn from_tensors(
        inputs: Vec<Arc<Tensor>>,
        labels: Vec<Arc<Tensor>>,
        batch_size: usize,
    ) -> Self {
        let size = inputs.len();
        Self {
            inputs,
            labels,
            size,
            batch_size,
        }
    }
}
