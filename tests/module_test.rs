use std::sync::{Arc, Mutex};

use raddar::core::StateDict;
use raddar::nn::embedding::{Embedding, OneHot};
use raddar::nn::{Linear, Module, LeakyReLU, Trainable};
use raddar::optim::{Optimizer, RMSPropBuilder};
use raddar::{seq, assert_tensor_eq};
use tch::{Reduction, Tensor};

#[test]
fn sequential_test() {
    let inputs = Tensor::of_slice2(&[[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]])
        .to(tch::Device::Cuda(0));
    let labels = Tensor::of_slice2(&[[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]])
        .to(tch::Device::Cuda(0));

    let model = seq!(
        Linear::new(1, 1, true),
        LeakyReLU::new(0.01),
        Linear::new(1, 1, true),
    );
    model.to(tch::Device::Cuda(0));
    let mut optimizer = Optimizer::new(RMSPropBuilder::default().build().unwrap(), &model);
    for _epoch in 1..=5000 {
        model.zero_grad();
        let loss = model(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
    }
    model.forward(&inputs).print();
}

#[test]
fn embedding_test() {
    let inputs = Tensor::of_slice(&[1i64, 2, 3, 4, 5]);
    let one_hot = OneHot::new(6);
    one_hot(&inputs).print();

    let embedding = Embedding::new(6, 3);
    embedding(&inputs).print();
}

#[test]
fn set_parameter_test() {
    let mut model = seq!(Linear::new(1, 1, true), Linear::new(1, 1, true),);
    let parameters = vec![
        ("0.weight".to_owned(), Arc::new(Mutex::new(Tensor::of_slice2(&[[1.0]])))),
        ("0.bias".to_owned(), Arc::new(Mutex::new(Tensor::of_slice(&[2.0])))),
        ("1.weight".to_owned(), Arc::new(Mutex::new(Tensor::of_slice2(&[[3.0]])))),
        ("1.bias".to_owned(), Arc::new(Mutex::new(Tensor::of_slice(&[2.0])))),
    ].into_iter().collect();
    let state_dict = StateDict::from_map(parameters);
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&Tensor::of_slice(&[1.0]));
    assert_tensor_eq!(&output, &Tensor::of_slice(&[11.0]));
}
