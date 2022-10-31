use std::collections::BTreeMap;
use std::sync::Arc;

use image::DynamicImage;
use raddar::dataset::{
    image_mappings, DataLoaderConfigBuilder, Dataset, DynImageDataset, LoadFromImageFolder,
    TensorDataset, UnsupervisedTensorDataset,
};
use raddar::nn::embedding::{Embedding, OneHot};
use raddar::nn::{
    alexnet, resnet50, vgg, BatchNorm1dBuilder, BatchNorm2dBuilder, BatchNorm3dBuilder,
    LayerNormBuilder, LinearBuilder, MaxPooling1DBuilder, Trainable, VggType,
};
use raddar::optim::{
    AdamBuilder, CosineAnnealingLRBuilder, Optimizer, RMSPropBuilder, StepLRBuilder,
};
use raddar::{assert_tensor_eq, seq, tensor};

use tch::{Device, Kind, Reduction, Tensor};

#[test]
fn sequential_test() {
    let inputs =
        tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]).to(tch::Device::Cuda(0));
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]])
        .to(tch::Device::Cuda(0));

    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        // LeakyReLU::new(0.01),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    model.to(tch::Device::Cuda(0));
    let mut optimizer = Optimizer::new(
        &model,
        RMSPropBuilder::default().build(),
        Some(StepLRBuilder::default().build()),
    );
    for _epoch in 1..=5000 {
        model.zero_grad();
        let loss = model(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
    }
    model(&inputs).print();
}

#[test]
fn embedding_test() {
    let inputs = tensor!([1i64, 2, 3, 4, 5]);
    let one_hot = OneHot::new(6);
    one_hot(&inputs).print();

    let embedding = Embedding::new(6, 3);
    embedding(&inputs).print();
}

#[test]
fn pooling_test() {
    let inputs = tensor!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let model = MaxPooling1DBuilder::default()
        .kernel_size([2])
        .stride([1])
        .build();
    let output = model(&inputs);
    assert_tensor_eq!(output, tensor!([[2., 3.], [5., 6.], [8., 9.]]));
}

#[test]
fn alexnet_test() {
    let num_classes = 100;
    let inputs = Tensor::rand(&[1, 3, 224, 224], (Kind::Double, Device::Cpu));
    let net = alexnet(num_classes, 0.5, true);
    let output = net(&inputs);
    assert!(output.size2().unwrap().1 == num_classes);
}

#[test]
fn batchnorm_test() {
    let bn1d_2 = BatchNorm1dBuilder::default()
        .num_features(4)
        .affine(false)
        .build();
    let input1 = tensor!([[2., 3., 4., 5.], [3., 4., 5., 6.], [4., 5., 6., 7.]]);
    let _output1 = bn1d_2(&input1);

    let bn1d_3 = BatchNorm1dBuilder::default().num_features(2).build();
    let input2 = tensor!([
        [[2., 3., 4., 5.], [2., 3., 4., 5.]],
        [[2., 3., 4., 5.], [2., 3., 4., 5.]]
    ]);
    let _output2 = bn1d_3(&input2);

    let bn2d = BatchNorm2dBuilder::default().num_features(3).build();
    let input3 = Tensor::ones(&[6, 3, 5, 14], (Kind::Double, Device::Cpu));
    let _output3 = bn2d(&input3);

    let bn3d = BatchNorm3dBuilder::default().num_features(8).build();
    let input4 = Tensor::ones(&[6, 8, 5, 14, 11], (Kind::Double, Device::Cpu));
    let _output4 = bn3d(&input4);
}
#[test]
fn layernorm() {
    let ln = LayerNormBuilder::default()
        .shape(Box::new([3, 5, 2]))
        .build();
    let input = Tensor::ones(&[6, 3, 5, 2], (Kind::Double, Device::Cpu));
    ln(&input).print();
}

#[test]
fn vgg_test() {
    let num_classes = 100;
    let inputs = Tensor::rand(&[1, 3, 224, 224], (Kind::Double, Device::Cpu));
    let net = vgg(VggType::Vgg11, num_classes, 0.5, true);
    let output = net(&inputs);
    assert!(output.size2().unwrap().1 == num_classes);
}

#[test]
fn resnet_test() {
    let num_classes = 100;
    let inputs = Tensor::rand(&[1, 3, 224, 224], (Kind::Double, Device::Cpu));
    let net = resnet50(num_classes);
    let output = net(&inputs);
    assert!(output.size2().unwrap().1 == num_classes);
}
#[test]
fn cifar10_test() {
    let num_classes = 10;
    let model = resnet50(num_classes);
    let mut optimizer = Optimizer::new(
        &model,
        AdamBuilder::default().learning_rate(0.01).build(),
        Some(CosineAnnealingLRBuilder::default().build()),
    );
    let classes_vec = vec![
        (0, "airplane".to_string()),
        (1, "automobile".to_string()),
        (2, "bird".to_string()),
        (3, "cat".to_string()),
        (4, "deer".to_string()),
        (5, "dog".to_string()),
        (6, "frog".to_string()),
        (7, "horse".to_string()),
        (8, "ship".to_string()),
        (9, "truck".to_string()),
    ];
    let classes_map: BTreeMap<_, _> = classes_vec.into_iter().collect();
    let mut cifar_dataset = TensorDataset::default();
    for (id, class) in &classes_map {
        let root_path = "dataset/cifar10/train/";
        let id = id.to_owned();

        let temp_dataset: TensorDataset =
            DynImageDataset::from_image_folder(&(root_path.to_owned() + class), ())
                .map::<DynImageDataset, _>(image_mappings::resize(224, 224))
                .map::<UnsupervisedTensorDataset, _>(image_mappings::to_tensor(
                    DynamicImage::into_rgb32f,
                ))
                .map(move |inputs: Arc<Tensor>| {
                    let new_inputs = inputs.permute(&[2, 0, 1]);
                    (Arc::new(new_inputs), Arc::new(tensor!([id])))
                });

        cifar_dataset = cifar_dataset
            .into_iter()
            .chain(temp_dataset.into_iter())
            .collect();
        println!("Class {} is loaded.", *class);
    }
    let cifar_dataloader = cifar_dataset.into_loader(
        DataLoaderConfigBuilder::default()
            .batch_size(32)
            .shuffle(true)
            .build()
            .unwrap(),
    );
    println!("DataLoader is prepared.");
    let onehot = OneHot::new(10);
    for _ in 1..5 {
        let epoch_loader = cifar_dataloader.clone();
        for (img, label) in epoch_loader {
            // label.print();
            model.zero_grad();
            let loss = model(&img.to_kind(Kind::Double)).mse_loss(
                &(onehot)(&label.to_kind(Kind::Int64)).to_kind(Kind::Double),
                Reduction::Mean,
            );
            loss.backward();
            loss.print();
            optimizer.step();
        }
    }
}
