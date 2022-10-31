use std::collections::BTreeMap;

use super::{module::Module, Trainable};
use crate::core::{Cellable, StateDict, TensorCell};
use raddar_derive::{ArchitectureBuilder, CallableModule};
use tch::{Device, Kind, Tensor};

/// A layer normalization layer.
/// 
/// See [Layer Normalization](https://arxiv.org/abs/1607.06450).
#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct LayerNorm {
    pub ln_weight: Option<TensorCell>,
    pub ln_bias: Option<TensorCell>,
    #[builder]
    pub shape: Box<[i64]>,
    #[builder(default = "1e-5")]
    pub eps: f64,
    #[builder(default = "true")]
    pub cudnn_enable: bool,
    #[builder(default = "true")]
    pub elementwise_affine: bool,
}

impl Trainable for LayerNorm {
    fn parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        if self.elementwise_affine {
            result.insert(
                "weight".to_owned(),
                self.ln_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.ln_bias.as_ref().unwrap().clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
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

impl LayerNorm {
    pub fn new(config: LayerNormConfig) -> LayerNorm {
        let size = &*config.shape;
        let ln_weight = if config.elementwise_affine {
            Some(Tensor::ones(size, (Kind::Double, Device::Cpu)).cell())
        } else {
            None
        };
        let ln_bias = if config.elementwise_affine {
            Some(Tensor::zeros(size, (Kind::Double, Device::Cpu)).cell())
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
