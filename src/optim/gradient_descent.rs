use tch::no_grad;

use crate::{core::{TensorCell, TensorNN}, optim::optimizer::OptimizerAlgorithm};

pub struct GradientDescent {
    learning_rate: f64,
}

impl<Ts: TensorNN> OptimizerAlgorithm<Ts> for GradientDescent {
    fn step(&mut self, trainable_parameters: &Vec<TensorCell<Ts>>) {
        for parameter in trainable_parameters {
            let mut parameter = parameter.lock();
            let grad = parameter.grad();
            no_grad(|| *parameter -= self.learning_rate * grad);
        }
    }
    fn init(&mut self, _trainable_parameters: &Vec<TensorCell<Ts>>) {}
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent { learning_rate }
    }
}
pub fn gradient_descent(learning_rate: f64) -> GradientDescent {
    GradientDescent { learning_rate }
}
