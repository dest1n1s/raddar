use raddar_derive::Module;

use super::{module::Module, StateDict, Trainable};
use crate::core::{Cellable, TensorCell, TensorNN};

/// A batch normalization layer in 1 dimension.
///
/// See [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).
#[derive(Debug, Module)]
#[module(tensor_type = "Ts", builder)]
pub struct BatchNorm1d<Ts: TensorNN> {
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
    pub bn_weight: Option<TensorCell<Ts>>,
    pub bn_bias: Option<TensorCell<Ts>>,
    pub running_mean: TensorCell<Ts>,
    pub running_var: TensorCell<Ts>,
}

impl<Ts: TensorNN> Trainable<Ts> for BatchNorm1d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        result
    }
    fn static_tensors(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("running_mean".to_owned(), self.running_mean.clone());
        result.insert("running_var".to_owned(), self.running_var.clone());
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for BatchNorm1d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
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
            Some(&*running_mean),
            Some(&*running_var),
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
impl<Ts: TensorNN> BatchNorm1d<Ts> {
    pub fn new(config: BatchNorm1dConfig<Ts>) -> BatchNorm1d<Ts> {
        let bn_weight = if config.affine {
            Some(
                Ts::ones(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Ts::zeros(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Ts::zeros(&[config.num_features], Ts::Env::default());
        let running_var = Ts::ones(&[config.num_features], Ts::Env::default());
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

#[derive(Debug, Module)]
#[module(tensor_type = "Ts", builder)]
pub struct BatchNorm2d<Ts: TensorNN> {
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
    pub bn_weight: Option<TensorCell<Ts>>,
    pub bn_bias: Option<TensorCell<Ts>>,
    pub running_mean: TensorCell<Ts>,
    pub running_var: TensorCell<Ts>,
}

impl<Ts: TensorNN> Trainable<Ts> for BatchNorm2d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        result
    }
    fn static_tensors(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("running_mean".to_owned(), self.running_mean.clone());
        result.insert("running_var".to_owned(), self.running_var.clone());
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for BatchNorm2d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
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
            Some(&*running_mean),
            Some(&*running_var),
            self.training,
            self.momentum,
            self.eps,
            self.cudnn_enabled,
        )
    }
}

impl<Ts: TensorNN> BatchNorm2d<Ts> {
    pub fn new(config: BatchNorm2dConfig<Ts>) -> BatchNorm2d<Ts> {
        let bn_weight = if config.affine {
            Some(
                Ts::ones(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Ts::zeros(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Ts::zeros(&[config.num_features], Ts::Env::default());
        let running_var = Ts::ones(&[config.num_features], Ts::Env::default());
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
#[derive(Debug, Module)]
#[module(tensor_type = "Ts", builder)]
pub struct BatchNorm3d<Ts: TensorNN> {
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
    pub bn_weight: Option<TensorCell<Ts>>,
    pub bn_bias: Option<TensorCell<Ts>>,
    pub running_mean: TensorCell<Ts>,
    pub running_var: TensorCell<Ts>,
}

impl<Ts: TensorNN> Trainable<Ts> for BatchNorm3d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        if self.affine {
            result.insert(
                "weight".to_owned(),
                self.bn_weight.as_ref().unwrap().clone(),
            );
            result.insert("bias".to_owned(), self.bn_bias.as_ref().unwrap().clone());
        }
        result
    }
    fn static_tensors(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("running_mean".to_owned(), self.running_mean.clone());
        result.insert("running_var".to_owned(), self.running_var.clone());
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for BatchNorm3d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
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
            Some(&*running_mean),
            Some(&*running_var),
            self.training,
            self.momentum,
            self.eps,
            self.cudnn_enabled,
        )
    }
}

impl<Ts: TensorNN> BatchNorm3d<Ts> {
    pub fn new(config: BatchNorm3dConfig<Ts>) -> BatchNorm3d<Ts> {
        let bn_weight = if config.affine {
            Some(
                Ts::ones(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let bn_bias = if config.affine {
            Some(
                Ts::zeros(&[config.num_features], Ts::Env::default())
                    .set_requires_grad(true)
                    .cell(),
            )
        } else {
            None
        };
        let running_mean = Ts::zeros(&[config.num_features], Ts::Env::default());
        let running_var = Ts::ones(&[config.num_features], Ts::Env::default());
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
