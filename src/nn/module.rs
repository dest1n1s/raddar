use tch::{no_grad, Device, Tensor};

use crate::core::{StateDict, TensorCell};

pub trait Trainable: std::fmt::Debug + Send {
    fn training_parameters(&self) -> Vec<TensorCell> {
        self.trainable_parameters()
            .to_vec()
            .into_iter()
            .filter(|tensor| tensor.lock().requires_grad())
            .collect()
    }

    fn trainable_parameter_size(&self) -> usize {
        self.training_parameters().len()
    }

    fn load_trainable_parameters(&mut self, parameters: StateDict) {
        self.trainable_parameters().load(parameters);
    }

    fn trainable_parameters(&self) -> StateDict;

    fn all_parameters(&self) -> Vec<TensorCell> {
        self.trainable_parameters().to_vec()
    }

    fn init(&mut self, init: tch::nn::Init) {
        no_grad(|| {
            for parameter in self.trainable_parameters().to_vec() {
                parameter.lock().init(init);
            }
        });
    }

    fn freeze(&mut self) {
        for tensor in self.trainable_parameters().to_vec() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(false);
            });
        }
    }

    fn unfreeze(&mut self) {
        for tensor in self.trainable_parameters().to_vec() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(true);
            });
        }
    }

    fn to(&self, device: Device) {
        self.all_parameters().iter().for_each(|param| {
            let mut param = param.lock();
            let requires_grad = param.requires_grad();
            no_grad(|| {
                *param = param.to(device).set_requires_grad(requires_grad);
            })
        });
    }

    fn zero_grad(&self) {
        self.trainable_parameters()
            .to_vec()
            .iter()
            .for_each(|param| {
                let mut param = param.lock();
                param.zero_grad();
            });
    }
}

pub trait Module: Trainable {
    fn forward(&self, input: &Tensor) -> Tensor;
}

pub trait NonParameterModule: Module {}

impl<T: NonParameterModule> Trainable for T {
    fn trainable_parameters(&self) -> StateDict {
        StateDict::new()
    }
}

#[macro_export]
macro_rules! new_module {
    ($type:ty, $config_builder_type:ty, ($($field:ident=$value:expr),*)) => {
        {
            let mut builder = <$config_builder_type>::default();
            $(
                builder = builder.$field($value);
            )*
            <$type>::new(builder.build().unwrap())
        }
    };
}
