use std::sync::{Arc, Mutex};
use tch::Tensor;

pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>;
}

pub trait NonParameterModule: Module {}

default impl<T: NonParameterModule> Module for T {
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        return vec![];
    }
}
