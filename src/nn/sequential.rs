use crate::nn::module::Module;
use std::sync::{Arc,Mutex};
use tch::Tensor;
use std::ops::Deref;

#[derive(Debug)]
pub struct Sequential(Vec<Box<dyn Module>>);

impl Deref for Sequential {
    type Target = Vec<Box<dyn Module>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Box<dyn Module>>> for Sequential {
    fn from(seq: Vec<Box<dyn Module>>) -> Self {
        Sequential(seq)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for module in self.iter(){
            x = module.forward(&x)
        }
        x
    }

    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        let mut result: Vec<Arc<Mutex<Tensor>>> = vec![];
        for module in self.iter(){
            result.append(&mut module.get_trainable_parameters())
        }
        result
    }
}

#[macro_export]
macro_rules! seq {
    ($($x:expr),* $(,)?) => {
        {
            $crate::nn::sequential::Sequential::from(vec![$(Box::new($x) as Box<dyn Module>,)*])
        }
    };
}