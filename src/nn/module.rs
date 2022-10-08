use std::sync::{Arc, Mutex};

use tch::{no_grad, Device, Tensor};

pub trait Trainable: std::fmt::Debug + Send {
    fn training_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        self.trainable_parameters()
            .into_iter()
            .filter(|tensor| tensor.lock().unwrap().requires_grad())
            .collect()
    }

    fn trainable_parameter_size(&self) -> usize {
        self.training_parameters().len()
    }

    fn set_trainable_parameters(&mut self, parameters: Vec<Arc<Mutex<Tensor>>>);

    fn trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>;
    
    fn all_parameters(&self) -> Vec<Arc<Mutex<Tensor>>>{
        self.trainable_parameters()
    }

    fn init(&mut self, init: tch::nn::Init) {
        no_grad(|| {
            for parameter in self.trainable_parameters() {
                parameter.lock().unwrap().init(init);
            }
        });
    }

    fn freeze(&mut self) {
        for tensor in self.trainable_parameters() {
            let mut tensor = tensor.lock().unwrap();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(false);
            });
        }
    }

    fn unfreeze(&mut self) {
        for tensor in self.trainable_parameters() {
            let mut tensor = tensor.lock().unwrap();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(true);
            });
        }
    }

    fn to(&self, device: Device) {
        self.all_parameters().iter().for_each(|param| {
            let mut param = param.lock().unwrap();
            let requires_grad = param.requires_grad();
            no_grad(|| {
                *param = param.to(device).set_requires_grad(requires_grad);
            })
        });
    }
    
    fn zero_grad(&self) {
        self.trainable_parameters().iter().for_each(|param| {
            let mut param = param.lock().unwrap();
            param.zero_grad();
        });
    }
}

pub trait Module: Trainable {
    fn forward(&self, input: &Tensor) -> Tensor;
}

pub trait NonParameterModule: Module {}

impl<T: NonParameterModule> Trainable for T {
    fn trainable_parameters(&self) -> Vec<Arc<Mutex<Tensor>>> {
        return vec![];
    }

    fn set_trainable_parameters(&mut self, _parameters: Vec<Arc<Mutex<Tensor>>>) {}
}
