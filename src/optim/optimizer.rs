use std::sync::{Arc, Mutex};
use tch::Tensor;

pub struct Optimizer {
    opt: Box<dyn OptimizerAlgorithm>,
    trainable_parameters: Vec<Arc<Mutex<Tensor>>>
}

pub trait OptimizerAlgorithm{
    fn step(&self, trainable_parameters: &mut Vec<Arc<Mutex<Tensor>>>);
}

impl Optimizer {
    fn step(&mut self) {
        self.opt.step(&mut self.trainable_parameters);
    }
}