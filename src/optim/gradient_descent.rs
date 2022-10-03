use std::sync::{Arc, Mutex};
use tch::Tensor;
use crate::optim::optimizer::OptimizerAlgorithm;

pub struct GradientDescent {
    learning_rate: f64
}

impl OptimizerAlgorithm for GradientDescent {
    fn step(&self, trainable_parameters: &mut Vec<Arc<Mutex<Tensor>>>) {
        for parameter in trainable_parameters {
            let mut parameter = parameter.lock().unwrap();
            let grad = parameter.grad();
            *parameter -= self.learning_rate * grad
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