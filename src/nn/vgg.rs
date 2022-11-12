use std::vec;

use raddar_derive::Module;

use crate::{seq, core::TensorNN};

use super::{
    AdaptiveAveragePooling2D, AdaptiveAveragePooling2DBuilder, BatchNorm2dBuilder, Conv2dBuilder,
    DropoutBuilder, LinearBuilder, MaxPooling2DBuilder, Mod, Module, ReLU, Sequential, Trainable,
    TrainableDict,
};
#[derive(Clone, Debug)]
pub enum VggType {
    Vgg11,
    Vgg11Bn,
    Vgg13,
    Vgg13Bn,
    Vgg16,
    Vgg16Bn,
    Vgg19,
    Vgg19Bn,
}

/// VGG model
///
/// The VGG model is a convolutional neural network that is 16 or 19 layers deep. See [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) for more details.
#[derive(Debug, Module)]
#[module(tensor_type="Ts", builder)]
pub struct Vgg<Ts: TensorNN> {
    pub features: Mod<Sequential<Ts>, Ts>,
    pub avgpool: Mod<AdaptiveAveragePooling2D, Ts>,
    pub classifier: Mod<Sequential<Ts>, Ts>,

    #[builder(default = "1000")]
    pub num_classes: i64,

    #[builder(default = "0.5")]
    pub dropout: f64,
    #[builder]
    pub vgg_type: VggType,
}

fn make_layer<Ts: TensorNN>(layer_type: Vec<i64>, batchnorm: bool) -> Mod<Sequential<Ts>, Ts> {
    let mut in_channel = 3;
    let mut features = Sequential::default();
    for i in &layer_type {
        match i {
            0 => {
                features.push(
                    MaxPooling2DBuilder::default()
                        .kernel_size([2, 2])
                        .stride([2, 2])
                        .build(),
                );
            }
            _ => {
                features.push(
                    Conv2dBuilder::default()
                        .in_channel(in_channel)
                        .out_channel(*i)
                        .kernel_size([3, 3])
                        .padding([1, 1])
                        .build(),
                );
                if batchnorm == true {
                    features.push(BatchNorm2dBuilder::default().num_features(*i).build())
                };
                in_channel = *i;
            }
        }
        features.push(Mod::new(ReLU));
    }
    Mod::new(features)
}

impl<Ts: TensorNN> Trainable<Ts> for Vgg<Ts> {
    fn child_modules(&self) -> TrainableDict<Ts> {
        let mut result = TrainableDict::new();
        result.insert("features".to_owned(), self.features.clone());
        result.insert("classifier".to_owned(), self.classifier.clone());
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Vgg<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let mut output = (self.features)(input);
        output = (self.avgpool)(&output);
        output = output.flatten(1, 3);
        output = (self.classifier)(&output);
        output
    }
}

impl<Ts: TensorNN> Vgg<Ts> {
    pub fn new(config: VggConfig<Ts>) -> Vgg<Ts> {
        let (layer_type, batchnorm) = match config.vgg_type {
            VggType::Vgg11 => (
                vec![64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0],
                false,
            ),
            VggType::Vgg11Bn => (
                vec![64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0],
                true,
            ),
            VggType::Vgg13 => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0,
                ],
                false,
            ),
            VggType::Vgg13Bn => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0,
                ],
                true,
            ),
            VggType::Vgg16 => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0,
                ],
                false,
            ),
            VggType::Vgg16Bn => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0,
                ],
                true,
            ),
            VggType::Vgg19 => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512,
                    512, 512, 0,
                ],
                false,
            ),
            VggType::Vgg19Bn => (
                vec![
                    64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512,
                    512, 512, 0,
                ],
                true,
            ),
        };
        let features = make_layer(layer_type, batchnorm);
        let avgpool = AdaptiveAveragePooling2DBuilder::default()
            .output_size([7, 7])
            .build();
        let classifier = seq!(
            LinearBuilder::default()
                .input_dim(512 * 7 * 7)
                .output_dim(4096)
                .build(),
            Mod::new(ReLU),
            DropoutBuilder::default().p(config.dropout).build(),
            LinearBuilder::default()
                .input_dim(4096)
                .output_dim(4096)
                .build(),
            Mod::new(ReLU),
            DropoutBuilder::default().p(config.dropout).build(),
            LinearBuilder::default()
                .input_dim(4096)
                .output_dim(config.num_classes)
                .build(),
        );
        Vgg {
            features,
            avgpool,
            classifier,
            num_classes: config.num_classes,
            dropout: config.dropout,
            vgg_type: config.vgg_type,
        }
    }
}

pub fn vgg<Ts: TensorNN>(vgg_type: VggType, num_classes: i64, dropout: f64, _pretrained: bool) -> Mod<Vgg<Ts>, Ts> {
    VggBuilder::default()
        .num_classes(num_classes)
        .dropout(dropout)
        .vgg_type(vgg_type)
        .build()
}
