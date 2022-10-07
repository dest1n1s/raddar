use std::sync::{Arc, Mutex};

use crate::optim::optimizer::OptimizerAlgorithm;
use derive_builder::Builder;
use tch::{no_grad, Tensor};
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct RMSProp {
    #[builder(default = "0.001")]
    learning_rate: f64,
    #[builder(default = "0.99")]
    alpha: f64,
    #[builder(default = "1e-8")]
    eps: f64,
    #[builder(default = "0.")]
    weight_decay: f64,
    #[builder(default = "0.")]
    momentum: f64,
    #[builder(default = "None")]
    r: Option<Vec<Tensor>>,
    #[builder(default = "None")]
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
impl RMSProp {
    pub fn new(
        learning_rate: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
    ) -> RMSProp {
        RMSProp {
            learning_rate: learning_rate,
            alpha: alpha,
            eps: eps,
            weight_decay: weight_decay,
            momentum: momentum,
            r: None,
            R: None,
        }
    }
}
