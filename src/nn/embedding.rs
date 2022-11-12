use raddar_derive::Module;

use crate::core::{Cellable, TensorCell, TensorNN};

use super::{Module, StateDict, Trainable};

/// A one-hot embedding layer.
///
/// This layer is used to convert a sequence of integers into a sequence of one-hot vectors.
#[derive(Debug, Module)]
#[module(paramless)]
pub struct OneHot {
    pub num_classes: i64,
}

impl<Ts: TensorNN> Module<Ts> for OneHot {
    fn forward(&self, input: &Ts) -> Ts {
        let mut original_size = input.shape(); //(32,1)
        original_size.push(self.num_classes); //(32,1,10)
        let one_hot = Ts::zeros(&original_size, input.env());
        let input = input.unsqueeze(-1); //(32,1,1)
        let one_hot = one_hot.scatter(-1, &input, &input.ones_like());
        // one_hot.squeeze_dim(-2)
        one_hot
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
#[derive(Debug, Module)]
#[module(tensor_type="Ts")]
pub struct Embedding<Ts: TensorNN> {
    pub num_embeddings: i64,
    pub embedding_dim: i64,
    pub weight: TensorCell<Ts>,
    one_hot: OneHot,
}

impl<Ts: TensorNN> Embedding<Ts> {
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        let mut weight = Ts::empty(
            &[num_embeddings, embedding_dim],
            Ts::Env::default(),
        )
        .set_requires_grad(true);

        weight.no_grad_mut().uniform_(0., 1.);

        Self {
            num_embeddings,
            embedding_dim,
            weight: weight.cell(),
            one_hot: OneHot::new(num_embeddings as i64),
        }
    }
}

impl<Ts: TensorNN> Trainable<Ts> for Embedding<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.weight.clone());
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Embedding<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let one_hotted = (self.one_hot)(input);
        let weight = self.weight.lock();
        one_hotted.type_as(&*weight).matmul(&*weight)
    }
}
