use std::sync::{Arc, Mutex};

use raddar_derive::CallableModule;
use tch::{Device, Kind, no_grad, Tensor};

use super::module::Module;

// A simple fully-connected layer.
#[derive(Debug, CallableModule)]
pub struct Linear {
    pub weight: Arc<Mutex<Tensor>>,
    pub bias: Option<Arc<Mutex<Tensor>>>,
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

    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        if let Some(bias) = &self.bias {
            vec![self.weight.clone(), bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }

    fn set_trainable_parameters(&mut self, parameters: Vec<Arc<Mutex<Tensor>>>) {
        self.weight = parameters[0].clone();
        if let Some(bias) = &mut self.bias {
            *bias = parameters[1].clone();
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
