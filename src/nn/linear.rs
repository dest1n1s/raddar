use std::{sync::{Arc, Mutex}, collections::HashMap};

use raddar_derive::CallableModule;
use tch::{no_grad, Device, Kind, Tensor};

use crate::core::StateDict;

use super::{module::Module, Trainable};

// A simple fully-connected layer.
#[derive(Debug, CallableModule)]
pub struct Linear {
    pub weight: Arc<Mutex<Tensor>>,
    pub bias: Option<Arc<Mutex<Tensor>>>,
}

impl Trainable for Linear {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = HashMap::new();
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
    pub fn new(input_dim: i64, output_dim: i64, bias: bool) -> Linear {
        let mut weight = Tensor::empty(&[input_dim, output_dim], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);

        let mut linear_bias =
            Tensor::empty(&[output_dim], (Kind::Double, Device::Cpu)).set_requires_grad(true);

        no_grad(|| {
            weight.init(tch::nn::Init::KaimingUniform);
            linear_bias.init(tch::nn::Init::KaimingUniform);
        });
        Linear {
            weight: Arc::new(Mutex::new(weight)),
            bias: if bias {
                Some(Arc::new(Mutex::new(linear_bias)))
            } else {
                None
            },
        }
    }
}
