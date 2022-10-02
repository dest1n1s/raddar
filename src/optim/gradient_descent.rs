use std::sync::{Arc, Mutex};
use tch::Tensor;
use crate::optim::optimizer::OptimizerAlgorithm;

pub struct GradientDescent {
    learning_rate: f64
}

impl OptimizerAlgorithm for GradientDescent {
    fn step(&self, trainable_parameters: &mut Vec<Arc<Mutex<Tensor>>>) {
        for parameter in trainable_parameters {
            let p = parameter.lock().unwrap();
            let g = p.grad();
            *p -= self.learning_rate * g
        }
    }
}

impl GradientDescent {
    fn new(learning_rate: f64) -> GradientDescent {
        GradientDescent {
            learning_rate
        }
    }
}