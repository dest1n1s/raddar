use std::collections::BTreeMap;

use raddar_derive::{ArchitectureBuilder, CallableModule};
use tch::{no_grad, Device, Kind, Tensor};

use crate::core::{Cellable, StateDict, TensorCell};

use super::{module::Module, Trainable};

// A simple fully-connected layer.
#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct Linear {
    pub linear_weight: TensorCell,
    pub linear_bias: Option<TensorCell>,
    #[builder]
    pub input_dim: i64,
    #[builder]
    pub output_dim: i64,
    #[builder(default = "true")]
    pub bias: bool,
}

impl Trainable for Linear {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.linear_weight.clone());
        if let Some(bias) = &self.linear_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.linear_weight.lock().unwrap();
        if let Some(bias) = &self.linear_bias {
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
            linear_weight: weight.cell(),
            linear_bias: if bias { Some(linear_bias.cell()) } else { None },
            input_dim,
            output_dim,
            bias,
        }
    }
}
