# Raddar

A Rust-native approach to deep learning.

Currently since there's no mature solution for n-dimensional array computing on gpu in rust, we temporarily use the `Tensor` and other CUDA toolkit from `tch`, which provides Rust bindings for `libtorch`. But we won't use high-level parts of it.

## Getting-Started

This crate requires CUDA and `libtorch` support. You need to:

- Install CUDA 11.3 from [NVIDIA](https://developer.nvidia.com/cuda-11-3-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local). After the installation, you need to set the `TORCH_CUDA_VERSION` environment variable to `11.3`, or `libtorch` cannot find your CUDA.

- Install `libtorch`. You can follow the instructions in [tch-rs](https://github.com/LaurentMazare/tch-rs/blob/main/README.md#getting-started). 

- (On Windows) Check if your rust use a MSVC-based toolchain. The GNU toolchain could not successfully compile `torch-sys`. You can check the current toolchain by running

  ```shell
  rustup toolchain list
  ```

  If not, run

  ```shell
  rustup toolchain install nightly-x86_64-pc-windows-msvc
  rustup toolchain default nightly-x86_64-pc-windows-msvc
  ```

  to switch the toolchain.

- You should now be able to run the project. Try ```device_test``` in ```tests/tch_test.rs``` to see if the all settings are correct.

## Examples

### Basic Tensor Operations

### Training a Model via Gradient Descent

Raddar provides automatic differentiation for most tensor operations it supports. This is commonly used to train models using gradient descent. The optimization is performed over variables which are created via a declarative macro `tensor!(...)` by defining their initializations in this example below. The `model` contains only one linear layer, which has `input_dim` and `output_dim` of one. Once the model has been generated, a optimizer with optimization algorithm of `GradientDescent` and scheduler of `StepLR` is created. Then on each step of the training loop:

- The forward pass is applied to the data.
- A loss is computed as the mean square error between the model output and the ground truth.
- Finally an optimization step is performed: gradients are computed and variables of `model` are modified accordingly.

``` rust
use raddar::nn::{LinearBuilder, Trainable};
use raddar::optim::{
    GradientDescent, Optimizer,
};
use raddar::tensor;
use tch::Reduction;
fn gradient_descent() {
    let inputs = tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]]);
    let model = LinearBuilder::default().input_dim(1).output_dim(1).build();
    let mut optimizer = Optimizer::new(
        model.training_parameters(),
        GradientDescent::new(0.01),
        Some(StepLRBuilder::default().build()),
    );
    for epoch in 1..=5000 {
        model.zero_grad();
        let loss = model(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
        println!("epoch: {}, loss: {}", epoch, f64::from(loss));
    }
    println!("final model:");
    println!(
        "weight: {}, bias: {}",
        f64::from(&*model.module().linear_weight.lock()),
        f64::from(&*model.module().linear_bias.as_ref().unwrap().lock())
    );
}
```

### Writing a Simple Neural Network

The declarative macro `seq!(...)` can be used to create neural network architectures. The following code defines a simple model with three hidden layers (one linear layer, one LeakyReLU layer, and one linear layer respectively) and a optimizer with optimization algorithm of `RMSProp` and scheduler of `StepLR`.

``` rust
use raddar::nn::{LinearBuilder, LeakyReLU, Mod};
use raddar::optim::{
    Optimizer, RMSPropBuilder, StepLRBuilder,
};
use raddar::{seq, tensor};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
fn run() {
    let inputs = tensor!([[1.0f32], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]])
        .to(tch::Device::Cuda(0));
    let labels = tensor!([[4.0f32], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]])
        .to(tch::Device::Cuda(0));
    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        Mod::new(LeakyReLU::new(0.01)),
    )
    .to(tch::Device::Cuda(0));
    model
        .module_mut()
        .push(LinearBuilder::default().input_dim(1).output_dim(1).build());
    let mut optimizer = Optimizer::new(
        model.training_parameters(),
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
```

### Creating a Tensor-Dataset

`TensorDataset::from_tensors` can be used to create a tensor-dataset. Use `into_loader` to  create a `DataLoader` from the corresponding dataset.

``` rust
use std::sync::Arc;
use raddar::{
    dataset::{DataLoaderConfigBuilder, TensorDataset},
    tensor, tensor_vec,
};
use tch::Tensor;
fn create_dataset() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = TensorDataset::from_tensors(inputs, labels);
    let mut iter = dataset
        .into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(3)
                .build()
                .unwrap(),
        )
        .cycle();
    iter.next();
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [2, 1]);
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
}
```

### Using Some Built-in Models

In the following code, the built-in `resnet50` model is trained on the CIFAR-10 dataset using `RMSProp` optimizer with scheduler algorithm of `CosineAnnealingLR`. There are also many other built-in model in Raddar, including VGG, AlexNet and DenseNet.

``` rust
use std::sync::Arc;
use image::DynamicImage;
use linked_hash_map::LinkedHashMap;
use raddar::dataset::{
    image_mappings, DataLoaderConfigBuilder, Dataset, DynImageDataset, LoadFromImageFolder,
    TensorDataset, UnsupervisedTensorDataset,
};
use raddar::nn::embedding::{Embedding, OneHot};
use raddar::nn::{
    resnet50, LeakyReLU, LinearBuilder, Mod, Trainable
};
use raddar::optim::{
    cosine_annealing_lr, Optimizer, rmsprop,
};
use raddar::{seq, tensor};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
fn cifar10() {
    let num_classes = 10;
    let batch_size = 32;
    let device = Device::Cuda(0);
    let model = resnet50(num_classes).to(device);
    let mut optimizer = opt_with_sched(
        model.training_parameters(),
        rmsprop(0.03, 0.99),
        cosine_annealing_lr(200, 100, 0.01),
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
    let classes_map: LinkedHashMap<_, _> = classes_vec.into_iter().collect();
    let mut cifar_dataset = TensorDataset::default();
    let mut valid_dataset = TensorDataset::default();
    for (id, class) in &classes_map {
        let train_path = "dataset/cifar10/train/";
        let valid_path = "dataset/cifar10/val/";
        let id = id.to_owned();
        let train_temp_dataset =
            DynImageDataset::from_image_folder(&(train_path.to_owned() + class), ())
                .map::<DynImageDataset, _>(image_mappings::resize(224, 224))
                .map::<UnsupervisedTensorDataset, _>(image_mappings::to_tensor(
                    DynamicImage::into_rgb32f,
                ))
                .map::<TensorDataset, _>(move |inputs: Arc<Tensor>| {
                    let new_inputs = inputs.permute(&[2, 0, 1]);
                    (Arc::new(new_inputs), Arc::new(Tensor::from(id)))
                })
                .to(device);
        let valid_temp_dataset =
            DynImageDataset::from_image_folder(&(valid_path.to_owned() + class), ())
                .map::<DynImageDataset, _>(image_mappings::resize(224, 224))
                .map::<UnsupervisedTensorDataset, _>(image_mappings::to_tensor(
                    DynamicImage::into_rgb32f,
                ))
                .map::<TensorDataset, _>(move |inputs: Arc<Tensor>| {
                    let new_inputs = inputs.permute(&[2, 0, 1]);
                    (Arc::new(new_inputs), Arc::new(Tensor::from(id)))
                })
                .to(device);
        cifar_dataset = cifar_dataset
            .into_iter()
            .chain(train_temp_dataset.into_iter())
            .collect();
        valid_dataset = valid_dataset
            .into_iter()
            .chain(valid_temp_dataset.into_iter())
            .collect();
        println!("Class {} is loaded.", *class);
    }
    let cifar_dataloader = cifar_dataset.into_loader(
        DataLoaderConfigBuilder::default()
            .batch_size(batch_size)
            .shuffle(true)
            .build()
            .unwrap(),
    );
    let valid_dataloader = valid_dataset.into_loader(
        DataLoaderConfigBuilder::default()
            .batch_size(batch_size)
            .shuffle(true)
            .build()
            .unwrap(),
    );
    println!("DataLoader is prepared.");
    let onehot = OneHot::new(num_classes);
    for _ in 1..50 {
        let epoch_loader = cifar_dataloader.clone();
        let valid_loader = valid_dataloader.clone();
        for (img, label) in epoch_loader {
            model.zero_grad();
            let output = model(&img.to_kind(Kind::Float));
            let loss = output.mse_loss(
                &(onehot)(&label.to_kind(Kind::Int64)).to_kind(Kind::Float),
                Reduction::Mean,
            );
            loss.backward();
            optimizer.step();
        }
        model.eval(true);
        let mut acc = Tensor::zeros(&[1], (Kind::Float, device));
        let mut now_bnum = 0;
        for (img, label) in valid_loader {
            no_grad(|| {
                let output = model(&img.to_kind(Kind::Float));
                let output = output.argmax(1, false);
                let bacc = output.eq_tensor(&label).mean(Kind::Float);
                acc += bacc;
                now_bnum += 1;
            });
        }
        acc = acc / now_bnum;
        acc.print();
        model.train(true);
    }
}
```