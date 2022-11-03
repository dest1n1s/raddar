use raddar_derive::{CallableModule, NonParameterModule};
use tch::{no_grad, Device, Kind, Tensor};

use crate::core::{Cellable, TensorCell};

use super::{Module, StateDict, Trainable};

/// A one-hot embedding layer.
///
/// This layer is used to convert a sequence of integers into a sequence of one-hot vectors.
#[derive(Debug, CallableModule, NonParameterModule)]
pub struct OneHot {
    pub num_classes: i64,
}

impl Module for OneHot {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut original_size = input.size();
        original_size.push(self.num_classes as i64);
        let mut one_hot = Tensor::zeros(&original_size, (input.kind(), input.device()));
        let input = input.unsqueeze(-1);
        one_hot.scatter_(-1, &input, &input.ones_like())
    }
}

impl OneHot {
    pub fn new(num_classes: i64) -> Self {
        Self { num_classes }
    }
}

/// An embedding layer.
///
/// See [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
#[derive(Debug, CallableModule)]
pub struct Embedding {
    pub num_embeddings: i64,
    pub embedding_dim: i64,
    pub weight: TensorCell,
    one_hot: OneHot,
}

impl Embedding {
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        let mut weight = Tensor::empty(
            &[num_embeddings, embedding_dim],
            (Kind::Double, Device::Cpu),
        )
        .set_requires_grad(true);

        no_grad(|| {
            weight.init(tch::nn::Init::Uniform { lo: 0., up: 1. });
        });

        Self {
            num_embeddings,
            embedding_dim,
            weight: weight.cell(),
            one_hot: OneHot::new(num_embeddings as i64),
        }
    }
}

impl Trainable for Embedding {
    fn parameters(&self) -> StateDict {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.weight.clone());
        result
    }
}

impl Module for Embedding {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let one_hotted = (self.one_hot)(input);
        let weight = self.weight.lock();
        one_hotted.type_as(&weight).matmul(&weight)
    }
}
