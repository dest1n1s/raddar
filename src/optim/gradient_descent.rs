use std::sync::{Arc, Mutex};
use tch::{no_grad, Tensor};
use crate::optim::optimizer::OptimizerAlgorithm;

pub struct GradientDescent {
    learning_rate: f64
}

impl OptimizerAlgorithm for GradientDescent {
    fn step(&self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>) {
        for parameter in trainable_parameters {
            let mut parameter = parameter.lock().unwrap();
            let grad = parameter.grad();
            no_grad(|| {
                *parameter -= self.learning_rate * grad
            });
        }
    }
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> GradientDescent {
        GradientDescent {
            learning_rate
        }
    }
}