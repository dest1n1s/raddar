use std::collections::BTreeMap;

use raddar_derive::{ArchitectureBuilder, CallableModule};
use tch::{Device, Kind, Tensor};

use super::{module::Module, Trainable};
use crate::core::{Cellable, StateDict, TensorCell};

/// A batch normalization layer in 1 dimension.
/// 
/// See [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
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
    pub affine: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: Option<TensorCell>,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}

impl Trainable for BatchNorm1d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 2 || input.dim() == 3);
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_weight = self.bn_weight.as_ref().map(|weight| weight.lock());
        let bn_weight = bn_weight.as_deref();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            bn_weight,
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

/// A batch normalization layer in 2 dimensions.
/// 
/// See [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
impl BatchNorm1d {
    pub fn new(config: BatchNorm1dConfig) -> BatchNorm1d {
        let bn_weight = if config.affine {
            Some(
                Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm1d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            affine: config.affine,
            bn_weight,
            bn_bias,
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
    pub affine: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: Option<TensorCell>,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}

impl Trainable for BatchNorm2d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 4);
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_weight = self.bn_weight.as_ref().map(|weight| weight.lock());
        let bn_weight = bn_weight.as_deref();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            bn_weight,
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
        let bn_weight = if config.affine {
            Some(
                Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm2d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            affine: config.affine,
            bn_weight,
            bn_bias,
            training: config.training,
            running_mean: running_mean.cell(),
            running_var: running_var.cell(),
        }
    }
}

/// A batch normalization layer in 3 dimensions.
/// 
/// See [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
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
    pub affine: bool,
    #[builder(default = "true")]
    pub training: bool,
    pub bn_weight: Option<TensorCell>,
    pub bn_bias: Option<TensorCell>,
    pub running_mean: TensorCell,
    pub running_var: TensorCell,
}

impl Trainable for BatchNorm3d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for BatchNorm3d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.dim() == 5);
        let running_mean = self.running_mean.lock();
        let running_var = self.running_var.lock();
        let bn_weight = self.bn_weight.as_ref().map(|weight| weight.lock());
        let bn_weight = bn_weight.as_deref();
        let bn_bias = self.bn_bias.as_ref().map(|bias| bias.lock());
        let bn_bias = bn_bias.as_deref();
        input.batch_norm(
            bn_weight,
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
        let bn_weight = if config.affine {
            Some(
                Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu))
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Tensor::zeros(&[config.num_features], (Kind::Double, Device::Cpu));
        let running_var = Tensor::ones(&[config.num_features], (Kind::Double, Device::Cpu));
        BatchNorm3d {
            num_features: config.num_features,
            eps: config.eps,
            momentum: config.momentum,
            cudnn_enabled: config.cudnn_enabled,
            affine: config.affine,
            bn_weight,
            bn_bias,
            training: config.training,
            running_mean: running_mean.cell(),
            running_var: running_var.cell(),
        }
    }
}
