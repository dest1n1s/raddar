use std::{sync::Arc};

use tch::Tensor;

use super::{Dataset, DatasetIterator};

#[derive(Debug)]
pub struct TensorDataset {
    pub inputs: Vec<Arc<Tensor>>,
    pub labels: Vec<Arc<Tensor>>,
    pub size: usize,
    pub batch_size: usize,
}

impl Dataset for TensorDataset {
    type InputType = Tensor;
    type LabelType = Tensor;
    type BatchType = (Tensor, Tensor);

    fn iter(&self) -> DatasetIterator<Self> {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }

    fn get_inputs(&self) -> &Vec<Arc<Self::InputType>> {
        &self.inputs
    }

    fn get_labels(&self) -> Option<&Vec<Arc<tch::Tensor>>> {
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
        labels: Option<Vec<Arc<tch::Tensor>>>,
    ) -> Self::BatchType {
        let labels = labels.unwrap();
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