use raddar_derive::{ArchitectureBuilder, CallableModule};

use crate::core::{Cellable, TensorCell, TensorNN};

use super::{Module, StateDict, Trainable};

/// A Convolution layer in 1 dimension.
///
/// See [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).
#[derive(Debug, CallableModule, ArchitectureBuilder)]
#[module(tensor_type="Ts")]
pub struct Conv1d<Ts: TensorNN> {
    pub conv_weight: TensorCell<Ts>,
    pub conv_bias: Option<TensorCell<Ts>>,

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

    #[builder(default = "1")]
    pub groups: i64,

    #[builder(default = "true")]
    pub bias: bool,
}

impl<Ts: TensorNN> Trainable<Ts> for Conv1d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Conv1d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let weight = self.conv_weight.lock();
        let bias = self.conv_bias.as_ref().map(|bias| bias.lock());
        let bias = bias.as_deref();
        input.conv1d(
            weight.as_ref(),
            bias,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )
    }
}

impl<Ts: TensorNN> Conv1d<Ts> {
    pub fn new(config: Conv1dConfig<Ts>) -> Conv1d<Ts> {
        let size: [i64; 3] = [config.out_channel, config.in_channel, config.kernel_size[0]];
        let mut conv_weight =
            Ts::empty(&size, Ts::Env::default()).set_requires_grad(true);
        let mut conv_bias = Ts::empty(&[config.out_channel], Ts::Env::default())
            .set_requires_grad(true);
        conv_weight.no_grad_mut().kaiming_uniform_();
        conv_bias.no_grad_mut().kaiming_uniform_();
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

/// A Convolution layer in 2 dimensions.
#[derive(Debug, CallableModule, ArchitectureBuilder)]
#[module(tensor_type="Ts")]
pub struct Conv2d<Ts: TensorNN> {
    pub conv_weight: TensorCell<Ts>,
    pub conv_bias: Option<TensorCell<Ts>>,

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
    #[builder(default = "1")]
    pub groups: i64,
    #[builder(default = "true")]
    pub bias: bool,
}

impl<Ts: TensorNN> Trainable<Ts> for Conv2d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Conv2d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let weight = &self.conv_weight.lock();
        let bias = self.conv_bias.as_ref().map(|bias| bias.lock());
        let bias = bias.as_deref();
        input.conv2d(
            weight.as_ref(),
            bias,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )
    }
}

impl<Ts: TensorNN> Conv2d<Ts> {
    pub fn new(config: Conv2dConfig<Ts>) -> Conv2d<Ts> {
        let size: [i64; 4] = [
            config.out_channel,
            config.in_channel,
            config.kernel_size[0],
            config.kernel_size[1],
        ];
        let mut conv_weight =
            Ts::empty(&size, Ts::Env::default()).set_requires_grad(true);
        let mut conv_bias = Ts::empty(&[config.out_channel], Ts::Env::default())
            .set_requires_grad(true);

        conv_weight.no_grad_mut().kaiming_uniform_();
        conv_bias.no_grad_mut().kaiming_uniform_();

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

/// A convolution layer in 3 dimensions.
#[derive(Debug, CallableModule, ArchitectureBuilder)]
#[module(tensor_type="Ts")]
pub struct Conv3d<Ts: TensorNN> {
    pub conv_weight: TensorCell<Ts>,
    pub conv_bias: Option<TensorCell<Ts>>,

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

    #[builder(default = "1")]
    pub groups: i64,

    #[builder(default = "true")]
    pub bias: bool,
}

impl<Ts: TensorNN> Trainable<Ts> for Conv3d<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.conv_weight.clone());
        if let Some(bias) = &self.conv_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Conv3d<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let weight = &self.conv_weight.lock();
        let bias = self.conv_bias.as_ref().map(|bias| bias.lock());
        let bias = bias.as_deref();
        input.conv3d(
            weight.as_ref(),
            bias,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )
    }
}

impl<Ts: TensorNN> Conv3d<Ts> {
    pub fn new(config: Conv3dConfig<Ts>) -> Conv3d<Ts> {
        let size: [i64; 5] = [
            config.out_channel,
            config.in_channel,
            config.kernel_size[0],
            config.kernel_size[1],
            config.kernel_size[2],
        ];
        let mut conv_weight =
            Ts::empty(&size, Ts::Env::default()).set_requires_grad(true);
        let mut conv_bias = Ts::empty(&[config.out_channel], Ts::Env::default())
            .set_requires_grad(true);

        conv_weight.no_grad_mut().kaiming_uniform_();
        conv_bias.no_grad_mut().kaiming_uniform_();

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
