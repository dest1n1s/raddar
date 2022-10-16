use raddar_derive::{CallableModule, ArchitectureBuilder};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tch::{no_grad, Device, Kind, Tensor};

use crate::core::{StateDict, TensorCell, Cellable};

use super::{Module, Trainable};
#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct Conv1d {
    pub conv_weight: TensorCell,
    pub conv_bias: Option<TensorCell>,

    #[builder]
    pub in_channel: i64,

    #[builder]
    pub out_channel: i64,

    #[builder]
    pub kernel_size: [i64; 1],

    #[builder(default = "[1]")]
    pub stride: [i64; 1],

    #[builder(default = "[0]")]
    pub padding: [i64; 1],

    #[builder(default = "[1]")]
    pub dilation: [i64; 1],

    #[builder(default = "0")]
    pub groups: i64,

    #[builder(default = "true")]
    pub bias: bool,
}

impl Trainable for Conv1d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.conv_weight.lock().unwrap();
        if let Some(bias) = &self.conv_bias {
            let bias = bias.lock().unwrap();
            input.conv1d(
                &weight,
                Some(&*bias),
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        } else {
            input.conv1d::<&Tensor>(
                &weight,
                None,
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        }
    }
}

impl Conv1d {
    pub fn new(config: Conv1dConfig) -> Conv1d {
        let size: [i64; 3] = [config.out_channel, config.in_channel, config.kernel_size[0]];
        let mut conv_weight =
            Tensor::empty(&size, (Kind::Double, Device::Cpu)).set_requires_grad(true);
        let mut conv_bias = Tensor::empty(&[config.out_channel], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);
        no_grad(|| {
            conv_weight.init(tch::nn::Init::KaimingUniform);
            conv_bias.init(tch::nn::Init::KaimingUniform);
        });
        Conv1d {
            conv_weight: conv_weight.cell(),
            conv_bias: if config.bias {
                Some(conv_bias.cell())
            } else {
                None
            },
            in_channel: config.in_channel,
            out_channel: config.out_channel,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            bias: config.bias,
        }
    }
}

#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct Conv2d {
    pub conv_weight: TensorCell,
    pub conv_bias: Option<TensorCell>,
    
    #[builder]
    pub in_channel: i64,

    #[builder]
    pub out_channel: i64,

    #[builder]
    pub kernel_size: [i64; 2],

    #[builder(default = "[1, 1]")]

    pub stride: [i64; 2],
    #[builder(default = "[0, 0]")]

    pub padding: [i64; 2],
    #[builder(default = "[1, 1]")]

    pub dilation: [i64; 2],
    #[builder(default = "0")]

    pub groups: i64,
    #[builder(default = "true")]

    pub bias: bool,
}

impl Trainable for Conv2d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.conv_weight.lock().unwrap();
        if let Some(bias) = &self.conv_bias {
            let bias = bias.lock().unwrap();
            input.conv2d(
                &weight,
                Some(&*bias),
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        } else {
            input.conv2d::<&Tensor>(
                &weight,
                None,
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        }
    }
}

impl Conv2d {
    pub fn new(config: Conv2dConfig) -> Conv2d {
        let size: [i64; 4] = [
            config.out_channel,
            config.in_channel,
            config.kernel_size[0],
            config.kernel_size[1],
        ];
        let mut conv_weight =
            Tensor::empty(&size, (Kind::Double, Device::Cpu)).set_requires_grad(true);
        let mut conv_bias = Tensor::empty(&[config.out_channel], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);

        no_grad(|| {
            conv_weight.init(tch::nn::Init::KaimingUniform);
            conv_bias.init(tch::nn::Init::KaimingUniform);
        });
        
        Conv2d {
            conv_weight: conv_weight.cell(),
            conv_bias: if config.bias {
                Some(conv_bias.cell())
            } else {
                None
            },
            in_channel: config.in_channel,
            out_channel: config.out_channel,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            bias: config.bias,
        }
    }
}

#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct Conv3d {
    pub conv_weight: TensorCell,
    pub conv_bias: Option<TensorCell>,
    
    #[builder]
    pub in_channel: i64,

    #[builder]
    pub out_channel: i64,

    #[builder]
    pub kernel_size: [i64; 3],

    #[builder(default = "[1, 1, 1]")]
    pub stride: [i64; 3],

    #[builder(default = "[0, 0, 0]")]
    pub padding: [i64; 3],

    #[builder(default = "[1, 1, 1]")]
    pub dilation: [i64; 3],

    #[builder(default = "0")]
    pub groups: i64,

    #[builder(default = "true")]
    pub bias: bool,
}

impl Trainable for Conv3d {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = BTreeMap::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        StateDict::from_map(result)
    }
}

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = &self.conv_weight.lock().unwrap();
        if let Some(bias) = &self.conv_bias {
            let bias = bias.lock().unwrap();
            input.conv3d(
                &weight,
                Some(&*bias),
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        } else {
            input.conv3d::<&Tensor>(
                &weight,
                None,
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups,
            )
        }
    }
}

impl Conv3d {
    pub fn new(config: Conv3dConfig) -> Conv3d {
        let size: [i64; 5] = [
            config.out_channel,
            config.in_channel,
            config.kernel_size[0],
            config.kernel_size[1],
            config.kernel_size[2],
        ];
        let mut conv_weight =
            Tensor::empty(&size, (Kind::Double, Device::Cpu)).set_requires_grad(true);
        let mut conv_bias = Tensor::empty(&[config.out_channel], (Kind::Double, Device::Cpu))
            .set_requires_grad(true);

        no_grad(|| {
            conv_weight.init(tch::nn::Init::KaimingUniform);
            conv_bias.init(tch::nn::Init::KaimingUniform);
        });

        Conv3d {
            conv_weight: conv_weight.cell(),
            conv_bias: if config.bias {
                Some(conv_bias.cell())
            } else {
                None
            },
            in_channel: config.in_channel,
            out_channel: config.out_channel,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            bias: config.bias,
        }
    }
}
