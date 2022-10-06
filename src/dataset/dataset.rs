use std::{ops::Index, path::Iter, sync::Arc, cmp::min};

use tch::Tensor;

#[derive(Debug)]
pub struct Dataset{
    pub inputs: Vec<Arc<Tensor>>,
    pub labels: Vec<Arc<Tensor>>,
    pub size: usize,
    pub batch_size: usize,
}

pub struct DatasetIterator<'a> {
    dataset: &'a Dataset,
    index: usize,
}

impl Dataset {
    pub fn from_tensors(inputs: Vec<Arc<Tensor>>, labels: Vec<Arc<Tensor>>, batch_size: usize) -> Self {
        let size = inputs.len();
        Self {
            inputs,
            labels,
            size,
            batch_size,
        }
    }

    pub fn iter(&self) -> DatasetIterator {
        DatasetIterator {
            dataset: self,
            index: 0,
        }
    }
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.size {
            return None;
        }
        let end = min(self.index + self.dataset.batch_size, self.dataset.size);
        let batch = self.dataset.inputs[self.index..end].to_vec();
        let batch_labels = self.dataset.labels[self.index..end].to_vec();
        self.index = end;
        Some((Tensor::stack(&batch, 0), Tensor::stack(&batch_labels, 0)))
    }
}