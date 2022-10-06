use std::sync::{Arc, Mutex};

use tch::Tensor;

use crate::nn::module::Module;

pub struct Optimizer<T: OptimizerAlgorithm> {
    opt: T,
    trainable_parameters: Vec<Arc<Mutex<Tensor>>>,
}

pub trait OptimizerAlgorithm {
    fn step(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>);
    fn init(&mut self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>);
}

impl<T: OptimizerAlgorithm> Optimizer<T> {
    pub fn step(&mut self) {
        self.opt.step(&self.trainable_parameters);
    }

    pub fn new(mut opt: T, model: &dyn Module) -> Optimizer<T> {
        opt.init(&model.get_trainable_parameters());
        Optimizer {
            opt,
            trainable_parameters: model.get_trainable_parameters(),
        }
    }
}
