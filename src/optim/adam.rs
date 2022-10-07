use std::sync::{Arc, Mutex};

use crate::optim::optimizer::OptimizerAlgorithm;
use derive_builder::Builder;
use tch::{no_grad, Tensor};
#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct Adam {
    #[builder(default = "0.001")]
    learning_rate: f64,
    #[builder(default = "(0.9,0.999)")]
    betas: (f64, f64),
    #[builder(default = "1e-8")]
    eps: f64,
    #[builder(default = "0.")]
    weight_decay: f64,
    #[builder(default = "0")]
    step: i32,
    #[builder(default = "None")]
    m: Option<Vec<Tensor>>,
    #[builder(default = "None")]
    v: Option<Vec<Tensor>>,
}

impl OptimizerAlgorithm for Adam {
    fn init(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>) {
        let mut vector_m: Vec<Tensor> = Vec::new();
        let mut vector_v: Vec<Tensor> = Vec::new();
        for parameter in trainable_parameters {
            let parameter = parameter.lock().unwrap();
            vector_m.push(Tensor::zeros_like(&*parameter));
            vector_v.push(Tensor::zeros_like(&*parameter));
        }
        self.m = Some(vector_m);
        self.v = Some(vector_v);
    }

    fn step(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>) {
        self.step += 1;
        for (i, parameter) in trainable_parameters.iter().enumerate() {
            let mut parameter = parameter.lock().unwrap();
            let mut grad = parameter.grad();
            no_grad(|| {
                let m = &mut self.m.as_mut().unwrap()[i];
                let v = &mut self.v.as_mut().unwrap()[i];
                grad = grad + &*parameter * self.weight_decay;

                *v = (&*v) * self.betas.1 + (1. - self.betas.1) * grad.square();
                *m = (&*m) * self.betas.0 + (1. - self.betas.0) * &grad;
                let m_hat = &*m / (1. - self.betas.0.powi(self.step));
                let v_hat = &*v / (1. - self.betas.1.powi(self.step));
                *parameter -= self.learning_rate * m_hat / (self.eps + v_hat.sqrt());
            })
        }
    }
}
