use tch::{Reduction, Tensor};
use raddar::nn::linear::Linear;
use raddar::nn::module::Module;
use raddar::optim::gradient_descent::GradientDescent;
use raddar::optim::optimizer::Optimizer;

#[test]
fn gradient_descent_test() {
    let inputs = Tensor::of_slice2(&[[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
    let labels = Tensor::of_slice2(&[[3.0], [7.1], [10.5], [9.0], [16.6], [20.8], [5.6], [13.0]]);

    let model = Linear::new(1, 1, true);
    let optimizer = Optimizer::new(GradientDescent::new(0.001), &model);
    for epoch in 1..=500 {
        let loss = model.forward(&inputs).mse_loss(&labels, Reduction::Mean);
        loss.backward();
        optimizer.step();
        println!("epoch: {}, loss: {}", epoch, f64::from(loss));
    }
    println!("final model:");
    println!("weight: {}, bias: {}", f64::from(&*model.weight.lock().unwrap()), f64::from(&*model.bias.unwrap().lock().unwrap()));
}