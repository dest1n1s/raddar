use std::sync::{Arc, Mutex};
use tch::{Device, Kind, Tensor};

// A simple fully-connected layer.
#[derive(Debug)]
pub struct Linear {
    pub weight: Arc<Mutex<Tensor>>,
    pub bias: Option<Arc<Mutex<Tensor>>>,
}

impl super::module::Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.weight.lock().unwrap();
        if let Some(bias) = &self.bias {
            let bias = bias.lock().unwrap();
            input.matmul(&weight.tr()) + &*bias
        } else {
            input.matmul(&weight.tr())
        }
    }

    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        if let Some(bias) = &self.bias {
            vec![self.weight.clone(), bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }
}

impl Linear {
    pub fn new(input_dim: i64, output_dim: i64, bias: bool) -> Linear {
        Linear {
            weight: Arc::new(Mutex::new(Tensor::zeros(&[output_dim, input_dim], (Kind::Double, Device::Cpu)).set_requires_grad(true))),
            bias: if bias {
                Some(Arc::new(Mutex::new(Tensor::zeros(&[output_dim], (Kind::Double, Device::Cpu)).set_requires_grad(true))))
            } else {
                None
            },
        }
    }
}
