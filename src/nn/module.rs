use std::sync::{Arc, Mutex};

use tch::{no_grad, Device, Tensor};

pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn get_training_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        self.get_trainable_parameters()
            .into_iter()
            .filter(|tensor| tensor.lock().unwrap().requires_grad())
            .collect()
    }

    fn get_trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>;
    
    fn get_all_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>{
        self.get_trainable_parameters()
    }

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
