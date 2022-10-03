use std::sync::{Arc, Mutex};
use tch::Tensor;
use crate::nn::module::Module;

pub struct Optimizer {
    opt: Box<dyn OptimizerAlgorithm>,
    trainable_parameters: Vec<Arc<Mutex<Tensor>>>
}

pub trait OptimizerAlgorithm{
    fn step(&self, trainable_parameters: &Vec<Arc<Mutex<Tensor>>>);
}

impl Optimizer {
    pub fn step(&self) {
        self.opt.step(&self.trainable_parameters);
    }

    pub fn new(opt: impl OptimizerAlgorithm + 'static, model: &dyn Module) -> Optimizer {
        Optimizer {
            opt: Box::new(opt),
            trainable_parameters: model.get_trainable_parameters()
        }
    }
}