use crate::{optim::optimizer::OptimizerAlgorithm, core::TensorCell};
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
    r1: Option<Vec<Tensor>>,
    #[builder(default = "None")]
    r2: Option<Vec<Tensor>>,
}
impl OptimizerAlgorithm for RMSProp {
    fn step(&mut self, trainable_parameters: &Vec<TensorCell>) {
        for (i, parameter) in trainable_parameters.iter().enumerate() {
            let mut parameter = parameter.lock().unwrap();
            let mut grad = parameter.grad();
            no_grad(|| {
                let r1 = &mut self.r1.as_mut().unwrap()[i];
                let r2 = &mut self.r2.as_mut().unwrap()[i];
                grad = grad + &*parameter * self.weight_decay;
                *r1 = (&*r1) * self.alpha + (1. - self.alpha) * grad.square();
                *r2 = self.momentum * (&*r2) + &grad / (r1.sqrt() + self.eps);
                *parameter -= &*r2 * self.learning_rate;
            });
        }
    }
    fn init(&mut self, trainable_parameters: &Vec<TensorCell>) {
        let mut vector_r1: Vec<Tensor> = Vec::new();
        let mut vector_r2: Vec<Tensor> = Vec::new();
        for parameter in trainable_parameters {
            let parameter = parameter.lock().unwrap();
            vector_r1.push(Tensor::zeros_like(&*&parameter));
            vector_r2.push(Tensor::zeros_like(&*&parameter));
        }
        self.r1 = Some(vector_r1);
        self.r2 = Some(vector_r2);
    }
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
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
            learning_rate,
            alpha,
            eps,
            weight_decay,
            momentum,
            r1: None,
            r2: None,
        }
    }
}
