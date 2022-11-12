use raddar_derive::Module;

use super::{module::Module, StateDict, Trainable};
use crate::core::{Cellable, TensorCell, TensorNN};

/// A layer normalization layer.
///
/// See [Layer Normalization](https://arxiv.org/abs/1607.06450).
#[derive(Debug, Module)]
#[module(tensor_type="Ts", builder)]
pub struct LayerNorm<Ts: TensorNN> {
    pub ln_weight: Option<TensorCell<Ts>>,
    pub ln_bias: Option<TensorCell<Ts>>,
    #[builder]
    pub shape: Vec<i64>,
    #[builder(default = "1e-5")]
    pub eps: f64,
    #[builder(default = "true")]
    pub cudnn_enable: bool,
    #[builder(default = "true")]
    pub elementwise_affine: bool,
}

impl<Ts: TensorNN> Trainable<Ts> for LayerNorm<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        if self.elementwise_affine {
            result.insert(
                "weight".to_owned(),
                self.ln_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.ln_bias.as_ref().unwrap().clone());
        }
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for LayerNorm<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let ln_weight = self.ln_weight.as_ref().map(|weight| weight.lock());
        let ln_weight = ln_weight.as_deref();
        let ln_bias = self.ln_bias.as_ref().map(|bias| bias.lock());
        let ln_bias = ln_bias.as_deref();
        input.layer_norm(
            &*self.shape,
            ln_weight,
            ln_bias,
            self.eps,
            self.cudnn_enable,
        )
    }
}

impl<Ts: TensorNN> LayerNorm<Ts> {
    pub fn new(config: LayerNormConfig<Ts>) -> LayerNorm<Ts> {
        let size = &*config.shape;
        let ln_weight = if config.elementwise_affine {
            Some(Ts::ones(size, Ts::Env::default()).cell())
        } else {
            None
        };
        let ln_bias = if config.elementwise_affine {
            Some(Ts::zeros(size, Ts::Env::default()).cell())
        } else {
            None
        };
        LayerNorm {
            ln_weight,
            ln_bias,
            shape: config.shape,
            eps: config.eps,
            cudnn_enable: config.cudnn_enable,
            elementwise_affine: config.elementwise_affine,
        }
    }
}
