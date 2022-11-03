use raddar::nn::{LinearBuilder, Trainable};
use raddar::optim::{
    AdamBuilder, CosineAnnealingLRBuilder, GradientDescent, Optimizer, StepLRBuilder,
};
use raddar::tensor;
use tch::Reduction;

#[test]
fn gradient_descent_test() {
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

#[test]
fn rmsprop_test() {
    let inputs = tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]]);

    let model = LinearBuilder::default().input_dim(1).output_dim(1).build();
    let mut optimizer = Optimizer::new(
        model.training_parameters(),
        AdamBuilder::default().learning_rate(0.01).build(),
        Some(CosineAnnealingLRBuilder::default().build()),
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
