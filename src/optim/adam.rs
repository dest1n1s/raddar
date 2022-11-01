use crate::{core::TensorCell, optim::optimizer::OptimizerAlgorithm};
use raddar_derive::{PartialBuilder};
use tch::{no_grad, Tensor};
#[derive(PartialBuilder)]

pub struct Adam {
    #[builder(default = "0.001")]
    learning_rate: f64,
    #[builder(default = "(0.9,0.999)")]
    betas: (f64, f64),
    #[builder(default = "1e-8")]
    eps: f64,
    #[builder(default = "0.")]
    weight_decay: f64,
    step: i64,
    m: Option<Vec<Tensor>>,
    v: Option<Vec<Tensor>>,
}

impl OptimizerAlgorithm for Adam {
    fn init(&mut self, trainable_parameters: &Vec<TensorCell>) {
        let mut vector_m: Vec<Tensor> = Vec::new();
        let mut vector_v: Vec<Tensor> = Vec::new();
        for parameter in trainable_parameters {
            let parameter = parameter.lock();
            vector_m.push(Tensor::zeros_like(&*parameter));
            vector_v.push(Tensor::zeros_like(&*parameter));
        }
        self.m = Some(vector_m);
        self.v = Some(vector_v);
    }

    fn step(&mut self, trainable_parameters: &Vec<TensorCell>) {
        self.step += 1;
        for (i, parameter) in trainable_parameters.iter().enumerate() {
            let mut parameter = parameter.lock();
            let mut grad = parameter.grad();
            no_grad(|| {
                let m = &mut self.m.as_mut().unwrap()[i];
                let v = &mut self.v.as_mut().unwrap()[i];
                grad = grad + &*parameter * self.weight_decay;
                *v = (&*v) * self.betas.1 + (1. - self.betas.1) * grad.square();
                *m = (&*m) * self.betas.0 + (1. - self.betas.0) * &grad;
                let m_hat = &*m / (1. - self.betas.0.powf(self.step as f64));
                let v_hat = &*v / (1. - self.betas.1.powf(self.step as f64));
                *parameter -= self.learning_rate * m_hat / (self.eps + v_hat.sqrt());
            })
        }
    }
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}
impl Adam {
    pub fn new(config: AdamConfig) -> Adam {
        Adam {
            learning_rate: config.learning_rate,
            betas: config.betas,
            eps: config.eps,
            weight_decay: config.weight_decay,
            step: 0,
            m: None,
            v: None,
        }
    }
}
pub fn adam(learning_rate: f64, betas: (f64, f64)) -> Adam {
    AdamBuilder::default()
        .learning_rate(learning_rate)
        .betas(betas)
        .build()
}
