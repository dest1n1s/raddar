use raddar::nn::embedding::{Embedding, OneHot};
use raddar::nn::{Linear, MaxPooling1D, MaxPooling2D, Trainable};
use raddar::optim::{Optimizer, RMSPropBuilder, StepLRBuilder};
use raddar::{assert_tensor_eq, seq, tensor};
use tch::Reduction;

#[test]
fn sequential_test() {
    let inputs = tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]])
        .to(tch::Device::Cuda(0));
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]])
        .to(tch::Device::Cuda(0));

    let model = seq!(
        Linear::new(1, 1, true),
        // LeakyReLU::new(0.01),
        Linear::new(1, 1, true),
    );
    model.to(tch::Device::Cuda(0));
    let mut optimizer = Optimizer::new(
        RMSPropBuilder::default().build().unwrap(),
        &model,
        Some(StepLRBuilder::default().build().unwrap()),
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
    let model = MaxPooling1D::new([2], [1], [0], [1], false);
    let output = model(&inputs);
    assert_tensor_eq!(output, tensor!([[2., 3.], [5., 6.], [8., 9.]]));

    let inputs = tensor!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let model = MaxPooling1D::new([2], [2], [0], [1], true);
    let output = model(&inputs);
    assert_tensor_eq!(output, tensor!([[2., 3.], [5., 6.], [8., 9.]]));

    let inputs = tensor!([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]);
    let model = MaxPooling2D::new([2, 2], [1, 1], [0, 0], [1, 1], false);
    let output = model(&inputs);
    assert_tensor_eq!(output, tensor!([[[5., 6.], [8., 9.]]]));
}
