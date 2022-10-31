use tch::no_grad;

use crate::{core::TensorCell, optim::optimizer::OptimizerAlgorithm};

pub struct GradientDescent {
    learning_rate: f64,
}

impl OptimizerAlgorithm for GradientDescent {
    fn step(&mut self, trainable_parameters: &Vec<TensorCell>) {
        for parameter in trainable_parameters {
            let mut parameter = parameter.lock();
            let grad = parameter.grad();
            no_grad(|| *parameter -= self.learning_rate * grad);
        }
    }
    fn init(&mut self, _trainable_parameters: &Vec<TensorCell>) {}
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> GradientDescent {
        GradientDescent { learning_rate }
    }
}
pub fn gradient_descent(learning_rate: f64) -> GradientDescent {
    GradientDescent { learning_rate }
}
