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
        if let Some(bias) = &self.bias {
            input.matmul(&self.weight.tr()) + bias
        } else {
            input.matmul(&self.weight.tr())
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
    fn new(input_dim: i64, output_dim: i64, bias: bool) -> Linear {
        Linear {
            weight: Arc::new(Mutex::new(Tensor::zeros(&[output_dim, input_dim], (Kind::Float, Device::Cpu)))),
            bias: if bias {
                Some(Arc::new(Mutex::new(Tensor::zeros(&[output_dim], (Kind::Float, Device::Cpu)))))
            } else {
                None
            },
        }
    }
}
