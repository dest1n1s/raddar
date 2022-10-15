use std::collections::BTreeMap;

use derive_builder::Builder;
use raddar_derive::CallableModule;
use tch::{no_grad, Device, Kind, Tensor};

use crate::core::{Cellable, StateDict, TensorCell};

use super::{module::Module, Trainable};

// A simple fully-connected layer.
#[derive(Debug, CallableModule)]
pub struct Linear {
    pub weight: TensorCell,
    pub bias: Option<TensorCell>,
    pub config: LinearConfig,
}
#[derive(Builder, Debug)]
#[builder(pattern = "owned")]
pub struct LinearConfig {
    pub input_dim: i64,
    pub output_dim: i64,
    #[builder(default = "true")]
    pub bias: bool,
}
impl Trainable for Linear {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.weight.clone());
        if let Some(bias) = &self.bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.weight.lock().unwrap();
        if let Some(bias) = &self.bias {
            let bias = bias.lock().unwrap();
            input.matmul(&weight) + &*bias
        } else {
            input.matmul(&weight)
        }
    }
}

impl Linear {
    pub fn new(config: LinearConfig) -> Linear {
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        let bias = config.bias;
        let mut weight = Tensor::empty(&[input_dim, output_dim], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);

        let mut linear_bias =
            Tensor::empty(&[output_dim], (Kind::Double, Device::Cpu)).set_requires_grad(true);

        no_grad(|| {
            weight.init(tch::nn::Init::KaimingUniform);
            linear_bias.init(tch::nn::Init::KaimingUniform);
        });
        Linear {
            weight: weight.cell(),
            bias: if bias { Some(linear_bias.cell()) } else { None },
            config,
        }
    }
}
