use std::sync::{Arc, Mutex};

use crate::optim::optimizer::OptimizerAlgorithm;
use tch::{no_grad, Tensor};

pub struct RMSProp {
    learning_rate: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    r: Option<Vec<Tensor>>,
    R: Option<Vec<Tensor>>,
}
impl OptimizerAlgorithm for RMSProp {
    fn step(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>) {
        for (i, parameter) in trainable_parameters.iter().enumerate() {
            let mut parameter = parameter.lock().unwrap();
            let mut grad = parameter.grad();
            no_grad(|| {
                let r = &mut self.r.as_mut().unwrap()[i];
                let R = &mut self.R.as_mut().unwrap()[i];
                grad = grad + &*parameter * self.weight_decay;
                *r = (&*r) * self.alpha + (1. - self.alpha) * grad.square();
                *R = self.momentum * (&*R) + &grad / (r.sqrt() + self.eps);
                *parameter -= &*R * self.learning_rate;
            });
        }
    }
    fn init(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>) {
        let mut vector_r: Vec<Tensor> = Vec::new();
        let mut vector_R: Vec<Tensor> = Vec::new();
        for parameter in trainable_parameters {
            let parameter = parameter.lock().unwrap();
            vector_r.push(Tensor::zeros_like(&*&parameter));
            vector_R.push(Tensor::zeros_like(&*&parameter));
        }
        self.r = Some(vector_r);
        self.R = Some(vector_R);
    }
}
impl Default for RMSProp {
    fn default() -> Self {
        RMSProp::new(0.001, None, None, None, None)
    }
}
impl RMSProp {
    pub fn new(
        learning_rate: f64,
        alpha: Option<f64>,
        eps: Option<f64>,
        weight_decay: Option<f64>,
        momentum: Option<f64>,
    ) -> RMSProp {
        RMSProp {
            learning_rate: learning_rate,
            alpha: alpha.unwrap_or(0.99),
            eps: eps.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.),
            momentum: momentum.unwrap_or(0.),
            r: None,
            R: None,
        }
    }
}
