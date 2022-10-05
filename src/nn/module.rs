use std::{sync::{Arc, Mutex}, ops::Deref};
use tch::{Tensor, Device, no_grad};

pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>;
    fn to(&self, device: Device) {
        self.get_trainable_parameters().iter().for_each(|param| {
            let mut param = param.lock().unwrap();
            no_grad(|| {
                *param = param.to(device).set_requires_grad(true);
            })
        });
    }
    fn zero_grad(&self) {
        self.get_trainable_parameters().iter().for_each(|param| {
            let mut param = param.lock().unwrap();
            param.zero_grad();
        });
    }
}

pub trait NonParameterModule: Module {}

default impl<T: NonParameterModule> Module for T {
    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        return vec![];
    }
}
