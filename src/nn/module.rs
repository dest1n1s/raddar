use std::sync::{Arc, Mutex};
use tch::Tensor;

pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>;
}

pub trait NonParameterModule: Module {
}

default impl<T: NonParameterModule> Module for T {
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        return vec![];
    }
}

impl Module for Vec<Box<dyn Module>> {
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        let mut result: Vec<Arc<Mutex<Tensor>>> = vec![];
        for module in self.iter(){
            result.append(&mut module.get_trainable_parameters())
        }
        result
    }

    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for module in self.iter(){
            x = module.forward(input)
        }
        x
    }
}