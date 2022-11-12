use raddar_derive::Module;

use crate::{seq, core::TensorNN};

use super::{
    AdaptiveAveragePooling2DBuilder, AveragePooling2DBuilder, BatchNorm2dBuilder, Conv2dBuilder,
    DropoutBuilder, Linear, LinearBuilder, MaxPooling2DBuilder, Mod, Module, ModuleDict,
    NamedSequential, ReLU, Trainable, TrainableDict,
};

pub fn transition<Ts: TensorNN>(num_input_features: i64, num_output_features: i64) -> Mod<NamedSequential<Ts>, Ts> {
    let mut res = NamedSequential::default();
    res.push((
        "norm".to_owned(),
        BatchNorm2dBuilder::default()
            .num_features(num_input_features)
            .build(),
    ));
    res.push(("relu".to_owned(), Mod::new(ReLU)));
    res.push((
        "conv".to_owned(),
        Conv2dBuilder::default()
            .in_channel(num_input_features)
            .out_channel(num_output_features)
            .kernel_size([1, 1])
            .stride([1, 1])
            .bias(false)
            .build(),
    ));
    res.push((
        "pool".to_owned(),
        AveragePooling2DBuilder::default()
            .kernel_size([2, 2])
            .stride([2, 2])
            .build(),
    ));
    Mod::new(res)
}

#[derive(Debug, Module)]
#[module(tensor_type = "Ts")]
pub struct DenseLayer<Ts: TensorNN> {
    modules: ModuleDict<Ts>,
    drop_rate: f64,
}
impl<Ts: TensorNN> Module<Ts> for DenseLayer<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        // println!("denselayer input {:?}", input.size());
        let norm1 = &self.modules["norm1"];
        let norm2 = &self.modules["norm2"];
        let relu1 = &self.modules["relu1"];
        let relu2 = &self.modules["relu2"];
        let conv1 = &self.modules["conv1"];
        let conv2 = &self.modules["conv2"];
        let bottleneck_output = conv1(&relu1(&norm1(input)));
        let new_features = conv2(&relu2(&norm2(&bottleneck_output)));
        // println!("denselayer output {:?}", new_features.size());
        if self.drop_rate > 0. {
            let dropout = DropoutBuilder::default().p(self.drop_rate).build();
            dropout(&new_features)
        } else {
            new_features
        }
    }
}
impl<Ts: TensorNN> Trainable<Ts> for DenseLayer<Ts> {
    fn child_modules(&self) -> TrainableDict<Ts> {
        self.modules
            .iter()
            .map(|(key, value)| (key.to_owned(), value.clone() as Mod<dyn Trainable<Ts>, Ts>))
            .collect()
    }
}
impl<Ts: TensorNN> DenseLayer<Ts> {
    pub fn new(
        num_input_features: i64,
        growth_rate: i64,
        bn_size: i64,
        drop_rate: f64,
    ) -> DenseLayer<Ts> {
        let mut modules = ModuleDict::new();
        modules.insert(
            "norm1".to_owned(),
            BatchNorm2dBuilder::default()
                .num_features(num_input_features)
                .build(),
        );
        modules.insert("relu1".to_owned(), Mod::new(ReLU));
        modules.insert(
            "conv1".to_owned(),
            Conv2dBuilder::default()
                .in_channel(num_input_features)
                .out_channel(bn_size * growth_rate)
                .kernel_size([1, 1])
                .stride([1, 1])
                .bias(false)
                .build(),
        );
        modules.insert(
            "norm2".to_owned(),
            BatchNorm2dBuilder::default()
                .num_features(bn_size * growth_rate)
                .build(),
        );
        modules.insert("relu2".to_owned(), Mod::new(ReLU));
        modules.insert(
            "conv2".to_owned(),
            Conv2dBuilder::default()
                .in_channel(bn_size * growth_rate)
                .out_channel(growth_rate)
                .kernel_size([3, 3])
                .stride([1, 1])
                .padding([1, 1])
                .bias(false)
                .build(),
        );
        DenseLayer { modules, drop_rate }
    }
}

pub fn denselayer<Ts: TensorNN>(
    num_input_features: i64,
    growth_rate: i64,
    bn_size: i64,
    drop_rate: f64,
) -> Mod<DenseLayer<Ts>, Ts> {
    Mod::new(DenseLayer::new(
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
    ))
}
#[derive(Debug, Module)]
#[module(tensor_type="Ts", builder)]
pub struct DenseBlock<Ts: TensorNN> {
    #[builder]
    pub num_layers: i64,
    #[builder]
    pub num_input_features: i64,
    #[builder]
    pub bn_size: i64,
    #[builder]
    pub growth_rate: i64,
    #[builder]
    pub drop_rate: f64,
    pub layers: ModuleDict<Ts>,
}
impl<Ts: TensorNN> Module<Ts> for DenseBlock<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let mut output = input.copy();
        for (_, layer) in &self.layers {
            let new_features = layer(&output);
            output = Ts::concat(&[output, new_features], 1);
        }
        output
    }
}
impl<Ts: TensorNN> Trainable<Ts> for DenseBlock<Ts> {
    fn child_modules(&self) -> TrainableDict<Ts> {
        self.layers
            .iter()
            .map(|(key, value)| (key.to_owned(), value.clone() as Mod<dyn Trainable<Ts>, Ts>))
            .collect()
    }
}
impl<Ts: TensorNN> DenseBlock<Ts> {
    pub fn new(config: DenseBlockConfig<Ts>) -> DenseBlock<Ts> {
        let mut layers = ModuleDict::new();
        for i in 0..config.num_layers {
            layers.insert(
                format!("denselayer{}", i + 1),
                denselayer(
                    config.num_input_features + i * config.growth_rate,
                    config.growth_rate,
                    config.bn_size,
                    config.drop_rate,
                ),
            );
        }
        DenseBlock {
            num_layers: config.num_layers,
            num_input_features: config.num_input_features,
            bn_size: config.bn_size,
            growth_rate: config.growth_rate,
            drop_rate: config.drop_rate,
            layers,
        }
    }
}
#[derive(Debug, Module)]
#[module(tensor_type="Ts", builder)]
pub struct DenseNet<Ts: TensorNN> {
    pub features: NamedSequential<Ts>,
    pub classifier: Mod<Linear<Ts>, Ts>,
    #[builder(default = "32")]
    pub growth_rate: i64,
    #[builder(default = "vec![6,12,24,16]")]
    pub block_config: Vec<i64>,
    #[builder(default = "64")]
    pub num_init_features: i64,
    #[builder(default = "4")]
    pub bn_size: i64,
    #[builder(default = "0.5")]
    pub drop_rate: f64,
    #[builder]
    pub num_classes: i64,
}
impl<Ts: TensorNN> Module<Ts> for DenseNet<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let features = (self.features)(input);
        let temp_relu = seq!(Mod::new(ReLU));
        let temp_avgpool = AdaptiveAveragePooling2DBuilder::default()
            .output_size([1, 1])
            .build();
        let mut out = temp_avgpool(&temp_relu(&features));
        out = out.flatten(1, 3);
        out = (self.classifier)(&out);
        out
    }
}
impl<Ts: TensorNN> Trainable<Ts> for DenseNet<Ts> {}
impl<Ts: TensorNN> DenseNet<Ts> {
    pub fn new(config: DenseNetConfig<Ts>) -> DenseNet<Ts> {
        let mut features = NamedSequential::default();
        features.push((
            "conv0".to_owned(),
            Conv2dBuilder::default()
                .in_channel(3)
                .out_channel(config.num_init_features)
                .kernel_size([7, 7])
                .stride([2, 2])
                .padding([3, 3])
                .bias(false)
                .build(),
        ));
        features.push((
            "norm0".to_owned(),
            BatchNorm2dBuilder::default()
                .num_features(config.num_init_features)
                .build(),
        ));
        features.push(("relu0".to_owned(), Mod::new(ReLU)));
        features.push((
            "pool0".to_owned(),
            MaxPooling2DBuilder::default()
                .kernel_size([3, 3])
                .stride([2, 2])
                .padding([1, 1])
                .build(),
        ));
        let mut num_features = config.num_init_features;
        for (i, num_layers) in config.block_config.iter().enumerate() {
            // println!("{}  {}  {}", i, num_layers, num_features);
            features.push((
                format!("denseblock{}", i + 1),
                DenseBlockBuilder::default()
                    .num_layers(*num_layers)
                    .num_input_features(num_features)
                    .bn_size(config.bn_size)
                    .growth_rate(config.growth_rate)
                    .drop_rate(config.drop_rate)
                    .build(),
            ));
            num_features += num_layers * config.growth_rate;
            if i != config.block_config.len() - 1 {
                features.push((
                    format!("transition{}", i + 1),
                    transition(num_features, num_features / 2),
                ));
                num_features /= 2;
            }
        }
        // let keys: Vec<String> = features.iter().map(|(key, _)| key.to_owned()).collect();
        // println!("{:#?}", keys);
        features.push((
            "norm5".to_owned(),
            BatchNorm2dBuilder::default()
                .num_features(num_features)
                .build(),
        ));
        let classifier = LinearBuilder::default()
            .input_dim(num_features)
            .output_dim(config.num_classes)
            .build();
        DenseNet {
            features,
            classifier,
            growth_rate: config.growth_rate,
            block_config: config.block_config,
            num_init_features: config.num_init_features,
            bn_size: config.bn_size,
            drop_rate: config.drop_rate,
            num_classes: config.num_classes,
        }
    }
}
pub fn densenet121<Ts: TensorNN>(num_classes: i64, drop_rate: f64) -> Mod<DenseNet<Ts>, Ts> {
    DenseNetBuilder::default()
        .num_classes(num_classes)
        .drop_rate(drop_rate)
        .block_config(vec![6, 12, 24, 16])
        .growth_rate(32)
        .num_init_features(64)
        .build()
}

pub fn densenet161<Ts: TensorNN>(num_classes: i64, drop_rate: f64) -> Mod<DenseNet<Ts>, Ts> {
    DenseNetBuilder::default()
        .num_classes(num_classes)
        .drop_rate(drop_rate)
        .block_config(vec![6, 12, 36, 24])
        .growth_rate(48)
        .num_init_features(96)
        .build()
}

pub fn densenet169<Ts: TensorNN>(num_classes: i64, drop_rate: f64) -> Mod<DenseNet<Ts>, Ts> {
    DenseNetBuilder::default()
        .num_classes(num_classes)
        .drop_rate(drop_rate)
        .block_config(vec![6, 12, 32, 32])
        .growth_rate(32)
        .num_init_features(64)
        .build()
}

pub fn densenet201<Ts: TensorNN>(num_classes: i64, drop_rate: f64) -> Mod<DenseNet<Ts>, Ts> {
    DenseNetBuilder::default()
        .num_classes(num_classes)
        .drop_rate(drop_rate)
        .block_config(vec![6, 12, 48, 32])
        .growth_rate(32)
        .num_init_features(64)
        .build()
}
