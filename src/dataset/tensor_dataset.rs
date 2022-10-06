use std::{sync::Arc, cmp::min};

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
    type DataType = Tensor;
    type BatchType = Tensor;

    fn iter(&self) -> DatasetIterator<Self> {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }

    fn get_inputs(&self) -> &Vec<Arc<Self::DataType>> {
        &self.inputs
    }

    fn get_labels(&self) -> &Vec<Arc<Self::DataType>> {
        &self.labels
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
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

impl<'a> Iterator for DatasetIterator<'a, TensorDataset> {
    type Item = (
        <TensorDataset as Dataset>::BatchType,
        <TensorDataset as Dataset>::BatchType,
    );

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.get_size() {
            return None;
        }
        let end = min(
            self.index + self.dataset.get_batch_size(),
            self.dataset.get_size(),
        );
        let batch = self.dataset.get_inputs()[self.index..end].to_vec();
        let batch_labels = self.dataset.get_labels()[self.index..end].to_vec();
        self.index = end;
        Some((Tensor::stack(&batch, 0), Tensor::stack(&batch_labels, 0)))
    }
}
