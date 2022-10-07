use raddar::nn::{LeakyReLU, Linear, Module};
use raddar::optim::{GradientDescent, Optimizer, RMSProp, RMSPropBuilder};
use raddar::seq;
use tch::{Reduction, Tensor};

#[test]
fn sequential_test() {
    let inputs = Tensor::of_slice2(&[[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]])
        .to(tch::Device::Cuda(0));
    let labels = Tensor::of_slice2(&[[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]])
        .to(tch::Device::Cuda(0));

    let model = seq!(
        Linear::new(1, 1, true),
        // LeakyReLU::new(0.01),
        Linear::new(1, 1, true),
    );
    model.to(tch::Device::Cuda(0));
    let mut optimizer = Optimizer::new(RMSPropBuilder::default().build().unwrap(), &model);
    for epoch in 1..=5000 {
        model.zero_grad();
        let loss: Tensor = model(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
    }
    model.forward(&inputs).print();
}
