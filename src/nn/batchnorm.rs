use std::collections::BTreeMap;

use raddar_derive::{ArchitectureBuilder, CallableModule};
use tch::{nn::Module, Device, Kind, Tensor};

use crate::core::{Cellable, StateDict, TensorCell};

use super::Trainable;

#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct BatchNorm1d {
    #[builder]
    pub num_features: i64,
    #[builder(default = "1e-5")]
    pub eps: f64,
    #[builder(default = "0.1")]
    pub momentum: f64,
    #[builder(default = "true")]
    pub cudnn_enabled: bool,
    #[builder(default = "true")]
    pub bias: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: TensorCell,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}
impl Trainable for BatchNorm1d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.bn_weight.clone());
        if let Some(bias) = &self.bn_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 2 || input.dim() == 3);
        let bn_weight = self.bn_weight.lock();
        // let running_mean = input.mean_dim(&[0], true, Kind::Double).squeeze();
        // let running_var = input.var_dim(&[0], true, true).squeeze();
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            Some(&*bn_weight),
            bn_bias,
            Some(&running_mean),
            Some(&running_var),
            self.training,
            self.momentum,
            self.eps,
            self.cudnn_enabled,
        )
    }
}
impl BatchNorm1d {
    pub fn new(config: BatchNorm1dConfig) -> BatchNorm1d {
        let bn_weight = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let bn_bias = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm1d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            bias: config.bias,
            bn_weight: bn_weight.cell(),
            bn_bias: if config.bias {
                Some(bn_bias.cell())
            } else {
                None
            },
            training: config.training,
            running_mean: running_mean.cell(),
            running_var: running_var.cell(),
        }
    }
}

#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct BatchNorm2d {
    #[builder]
    pub num_features: i64,
    #[builder(default = "1e-5")]
    pub eps: f64,
    #[builder(default = "0.1")]
    pub momentum: f64,
    #[builder(default = "true")]
    pub cudnn_enabled: bool,
    #[builder(default = "true")]
    pub bias: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: TensorCell,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}
impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 4);
        let bn_weight = self.bn_weight.lock();
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            Some(&*bn_weight),
            bn_bias,
            Some(&running_mean),
            Some(&running_var),
            self.training,
            self.momentum,
            self.eps,
            self.cudnn_enabled,
        )
    }
}
impl BatchNorm2d {
    pub fn new(config: BatchNorm2dConfig) -> BatchNorm2d {
        let bn_weight = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let bn_bias = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm2d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            bias: config.bias,
            bn_weight: bn_weight.cell(),
            bn_bias: if config.bias {
                Some(bn_bias.cell())
            } else {
                None
            },
            training: config.training,
            running_mean: running_mean.cell(),
            running_var: running_var.cell(),
        }
    }
}

#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct BatchNorm3d {
    #[builder]
    pub num_features: i64,
    #[builder(default = "1e-5")]
    pub eps: f64,
    #[builder(default = "0.1")]
    pub momentum: f64,
    #[builder(default = "true")]
    pub cudnn_enabled: bool,
    #[builder(default = "true")]
    pub bias: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: TensorCell,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}
impl Module for BatchNorm3d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 5);
        let bn_weight = self.bn_weight.lock();
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            Some(&*bn_weight),
            bn_bias,
            Some(&running_mean),
            Some(&running_var),
            self.training,
            self.momentum,
            self.eps,
            self.cudnn_enabled,
        )
    }
}
impl BatchNorm3d {
    pub fn new(config: BatchNorm3dConfig) -> BatchNorm3d {
        let bn_weight = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let bn_bias = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm3d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            bias: config.bias,
            bn_weight: bn_weight.cell(),
            bn_bias: if config.bias {
                Some(bn_bias.cell())
            } else {
                None
            },
            training: config.training,
            running_mean: running_mean.cell(),
            running_var: running_var.cell(),
        }
    }
}
