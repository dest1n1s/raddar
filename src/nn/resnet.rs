use std::marker::PhantomData;

use raddar_derive::{ArchitectureBuilder, CallableModule};
use tch::Tensor;

use crate::{core::StateDict, nn::ReLU, seq};

use super::{
    AdaptiveAveragePooling2DBuilder, BatchNorm2dBuilder, Conv2d, Conv2dBuilder, LinearBuilder,
    MaxPooling2DBuilder, Module, Sequential, Trainable,
};

pub trait Block {
    fn expansion() -> i64;
    fn new_block(
        inplanes: i64,
        planes: i64,
        stride: [i64; 2],
        groups: i64,
        base_width: i64,
        dilation: [i64; 2],
        downsample: Option<Sequential>,
        norm_layer: Option<fn(i64) -> Sequential>,
    ) -> Self;
}
pub fn conv3x3(
    in_planes: i64,
    out_planes: i64,
    stride: [i64; 2],
    groups: i64,
    dilation: [i64; 2],
) -> Conv2d {
    Conv2dBuilder::default()
        .kernel_size([3, 3])
        .in_channel(in_planes)
        .out_channel(out_planes)
        .stride(stride)
        .groups(groups)
        .dilation(dilation)
        .bias(false)
        .padding(dilation)
        .build()
}

pub fn conv1x1(in_planes: i64, out_planes: i64, stride: [i64; 2]) -> Conv2d {
    Conv2dBuilder::default()
        .kernel_size([1, 1])
        .in_channel(in_planes)
        .out_channel(out_planes)
        .stride(stride)
        .bias(false)
        .build()
}
pub fn batchnorm2d(num_features: i64) -> Sequential {
    seq!(BatchNorm2dBuilder::default()
        .num_features(num_features)
        .build())
}
#[derive(Debug, CallableModule)]
pub struct BasicBlock {
    pub block: Sequential,
    pub downsample: Option<Sequential>,
}
impl Trainable for BasicBlock {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = StateDict::new();
        // result.append_child("features".to_owned(), self.features.trainable_parameters());

        result
    }
}
impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut identity = input.copy();
        let mut output = (self.block)(input);
        if let Some(downsample) = &self.downsample {
            identity = (*downsample)(&identity);
        }
        output += identity;
        let relu = seq!(ReLU);
        (relu)(&output)
    }
}
impl Block for BasicBlock {
    fn expansion() -> i64 {
        1
    }
    fn new_block(
        in_planes: i64,
        planes: i64,
        stride: [i64; 2],
        groups: i64,
        base_width: i64,
        dilation: [i64; 2],
        downsample: Option<Sequential>,
        norm_layer: Option<fn(i64) -> Sequential>,
    ) -> Self {
        assert!(groups == 1 && base_width == 64 && dilation == [1, 1]);
        let normlayer = if let Some(temp) = norm_layer {
            temp
        } else {
            batchnorm2d
        };
        let mut block = seq!();
        block.push(Box::new(conv3x3(
            in_planes, planes, stride, groups, dilation,
        )));
        block.push(Box::new(normlayer(planes)));
        block.push(Box::new(ReLU));
        block.push(Box::new(conv3x3(planes, planes, [1, 1], groups, dilation)));

        block.push(Box::new(normlayer(planes)));
        Self { block, downsample }
    }
}
#[derive(Debug, CallableModule)]
pub struct BottleNeck {
    pub block: Sequential,
    pub downsample: Option<Sequential>,
}
impl Trainable for BottleNeck {
    fn trainable_parameters(&self) -> StateDict {
        let mut result = StateDict::new();
        // result.append_child("features".to_owned(), self.features.trainable_parameters());
        result
    }
}
impl Module for BottleNeck {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut identity = input.copy();
        let mut output = (self.block)(input);
        if let Some(downsample) = &self.downsample {
            identity = (*downsample)(&identity);
        }
        output += identity;
        let relu = seq!(ReLU);
        (relu)(&output)
    }
}
impl Block for BottleNeck {
    fn expansion() -> i64 {
        4
    }
    fn new_block(
        inplanes: i64,
        planes: i64,
        stride: [i64; 2],
        groups: i64,
        base_width: i64,
        dilation: [i64; 2],
        downsample: Option<Sequential>,
        norm_layer: Option<fn(i64) -> Sequential>,
    ) -> Self {
        let normlayer = if let Some(temp) = norm_layer {
            temp
        } else {
            batchnorm2d
        };
        let width = (((planes as f64) * (base_width as f64) / 64.0) as i64) * groups;
        let mut block = seq!();
        block.push(Box::new(conv1x1(inplanes, width, [1, 1])));
        block.push(Box::new(normlayer(width)));
        block.push(Box::new(ReLU));
        block.push(Box::new(conv3x3(width, width, stride, groups, dilation)));
        block.push(Box::new(normlayer(width)));
        block.push(Box::new(ReLU));
        block.push(Box::new(conv1x1(
            width,
            planes * <BottleNeck as Block>::expansion(),
            [1, 1],
        )));
        block.push(Box::new(normlayer(
            planes * <BottleNeck as Block>::expansion(),
        )));
        Self { block, downsample }
    }
}
#[derive(Debug, CallableModule, ArchitectureBuilder)]
pub struct ResNet<T: Block> {
    #[builder(default = "64")]
    pub base_width: i64,
    #[builder(default = "1000")]
    pub num_classes: i64,
    #[builder]
    pub layers: [i64; 4],
    pub net: Sequential,
    #[builder(default = "[false, false, false]")]
    pub replace_stride_with_dilation: [bool; 3],
    #[builder(default = "1")]
    pub groups: i64,
    #[builder(default = "Some(batchnorm2d)")]
    pub norm_layer: Option<fn(i64) -> Sequential>,
    #[builder(default = "1")]
    pub dilation: i64,
    #[builder(default = "64")]
    pub inplanes: i64,
    #[builder]
    pub _phantom: PhantomData<T>,
}

impl<T: Block> ResNet<T> {
    fn new(config: ResNetConfig<T>) -> ResNet<T> {
        let normlayer = if let Some(temp) = config.norm_layer {
            temp
        } else {
            batchnorm2d
        };
        let net = seq!();
        net.push(Box::new(
            Conv2dBuilder::default()
                .kernel_size([7, 7])
                .in_channel(3)
                .out_channel(64)
                .stride([2, 2])
                .padding([3, 3])
                .bias(false)
                .build(),
        ));
        net.push(Box::new(normlayer(64)));
        net.push(Box::new(ReLU));
        net.push(Box::new(
            MaxPooling2DBuilder::default()
                .kernel_size([3, 3])
                .stride([2, 2])
                .padding([1, 1])
                .build(),
        ));
        net.push(Box::new(make_layer(
            normlayer,
            PhantomData::<T>,
            64,
            config.layers[0],
            [1, 1],
            false,
            &config.dilation,
            &config.inplanes,
            config.groups,
            config.base_width,
        )));
        net.push(Box::new(make_layer(
            normlayer,
            PhantomData::<T>,
            128,
            config.layers[1],
            [2, 2],
            config.replace_stride_with_dilation[0],
            &config.dilation,
            &config.inplanes,
            config.groups,
            config.base_width,
        )));
        net.push(Box::new(make_layer(
            normlayer,
            PhantomData::<T>,
            256,
            config.layers[2],
            [2, 2],
            config.replace_stride_with_dilation[1],
            &config.dilation,
            &config.inplanes,
            config.groups,
            config.base_width,
        )));
        net.push(Box::new(make_layer(
            normlayer,
            PhantomData::<T>,
            512,
            config.layers[3],
            [2, 2],
            config.replace_stride_with_dilation[2],
            &config.dilation,
            &config.inplanes,
            config.groups,
            config.base_width,
        )));
        net.push(Box::new(
            AdaptiveAveragePooling2DBuilder::default()
                .output_size([1, 1])
                .build(),
        ));
        net.push(Box::new(
            LinearBuilder::default()
                .input_dim(T::expansion() * 512)
                .output_dim(config.num_classes)
                .build(),
        ));
        ResNet {
            base_width: config.base_width,
            num_classes: config.num_classes,
            layers: config.layers,
            net,
            replace_stride_with_dilation: config.replace_stride_with_dilation,
            groups: config.groups,
            norm_layer: Some(normlayer),
            _phantom: PhantomData::<T>,
            dilation: config.dilation,
            inplanes: config.inplanes,
        }
    }
}

fn make_layer<T: Block>(
    normlayer: fn(i64) -> Sequential,
    _: PhantomData<T>,
    planes: i64,
    block_num: i64,
    stride: [i64; 2],
    dilate: bool,
    config_dilation: &[i64; 2],
    config_inplanes: &i64,
    groups: i64,
    base_width: i64,
) -> T {
    let downsample = None;
    let previous_dilation = *config_dilation;
    if dilate == true {
        *config_dilation *= stride;
        stride = [1, 1];
    }
    if stride != [1, 1] || *config_inplanes != planes * T::expansion() {
        downsample = Some(seq!(
            conv1x1(*config_inplanes, planes * T::expansion(), stride),
            normlayer(planes * T::expansion()),
        ));
    }
    let layers = seq!();
    layers.push(Box::new(T::new_block(
        *config_inplanes,
        planes,
        stride,
        groups,
        base_width,
        previous_dilation,
        downsample,
        Some(normlayer),
    )));
    *config_inplanes = planes * T::expansion();
    for _ in 1..=block_num - 1 {
        layers.push(Box::new(T::new_block(
            *config_inplanes,
            planes,
            stride,
            groups,
            base_width,
            [1, 1],
            None,
            Some(normlayer),
        )));
    }
    layers
}
