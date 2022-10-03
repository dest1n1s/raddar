use std::sync::{Arc, Mutex};
use tch::Tensor;
use crate::nn::module::Module;

pub struct Optimizer<T: OptimizerAlgorithm> {
    opt: T,
    trainable_parameters: Vec<Arc<Mutex<Tensor>>>
}

pub trait OptimizerAlgorithm{
    fn step(&self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>);
}

impl<T: OptimizerAlgorithm> Optimizer<T> {
    pub fn step(&self) {
        self.opt.step(&self.trainable_parameters);
    }

    pub fn new(opt: T, model: &dyn Module) -> Optimizer<T> {
        Optimizer {
            opt,
            trainable_parameters: model.get_trainable_parameters()
        }
    }
}