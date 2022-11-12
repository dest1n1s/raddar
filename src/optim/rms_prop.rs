use std::marker::PhantomData;

use crate::{
    core::{TensorCell, TensorNN},
    optim::optimizer::OptimizerAlgorithm,
};
use raddar_derive::PartialBuilder;
use tch::no_grad;

#[derive(PartialBuilder)]
pub struct RMSProp<Ts: TensorNN> {
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
    r1: Option<Vec<Ts>>,
    r2: Option<Vec<Ts>>,
    #[builder(default = "PhantomData")]
    _marker: PhantomData<Ts>,
}
impl<Ts: TensorNN> OptimizerAlgorithm<Ts> for RMSProp<Ts> {
    fn step(&mut self, trainable_parameters: &Vec<TensorCell<Ts>>) {
        for (i, parameter) in trainable_parameters.iter().enumerate() {
            let mut parameter = parameter.lock();
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
    fn init(&mut self, trainable_parameters: &Vec<TensorCell<Ts>>) {
        let mut vector_r1: Vec<Ts> = Vec::new();
        let mut vector_r2: Vec<Ts> = Vec::new();
        for parameter in trainable_parameters {
            let parameter = parameter.lock();
            vector_r1.push(Ts::zeros_like(&*parameter));
            vector_r2.push(Ts::zeros_like(&*parameter));
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

impl<Ts: TensorNN> RMSProp<Ts> {
    pub fn new(config: RMSPropConfig<Ts>) -> RMSProp<Ts> {
        RMSProp {
            learning_rate: config.learning_rate,
            alpha: config.alpha,
            eps: config.eps,
            weight_decay: config.weight_decay,
            momentum: config.momentum,
            r1: None,
            r2: None,
            _marker: PhantomData,
        }
    }
}
pub fn rmsprop<Ts: TensorNN>(learning_rate: f64, alpha: f64) -> RMSProp<Ts> {
    RMSPropBuilder::default()
        .learning_rate(learning_rate)
        .alpha(alpha)
        .build()
}
