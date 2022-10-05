use raddar::{
    nn::{
        act_funcs::{GeLU, LeakyReLU},
        linear::Linear,
        module::Module,
        sequential::Sequential
    },
    optim::{gradient_descent::GradientDescent, optimizer::Optimizer},
    seq,
};
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
    let optimizer = Optimizer::new(GradientDescent::new(0.0001), &model);
    for epoch in 1..=50000 {
        model.zero_grad();
        let loss = model.forward(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
    }
    model.forward(&inputs).print();
}
