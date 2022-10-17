use raddar::nn::{LinearBuilder, Trainable};
use raddar::optim::{AdamBuilder, CosineAnnealingLRBuilder, GradientDescent, Optimizer};
use raddar::tensor;
use tch::Reduction;

#[test]
fn gradient_descent_test() {
    let inputs = tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]]);

    let model = LinearBuilder::default().input_dim(1).output_dim(1).build();
    let mut optimizer = Optimizer::new(
        GradientDescent::new(0.01),
        &model,
        Some(CosineAnnealingLRBuilder::default().build().unwrap()),
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
        f64::from(&*model.linear_weight.lock().unwrap()),
        f64::from(&*model.linear_bias.unwrap().lock().unwrap())
    );
}

#[test]
fn rmsprop_test() {
    let inputs = tensor!([[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
    let labels = tensor!([[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]]);

    let model = LinearBuilder::default().input_dim(1).output_dim(1).build();
    let mut optimizer = Optimizer::new(
        AdamBuilder::default().learning_rate(0.01).build().unwrap(),
        &model,
        Some(CosineAnnealingLRBuilder::default().build().unwrap()),
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
        f64::from(&*model.linear_weight.lock().unwrap()),
        f64::from(&*model.linear_bias.unwrap().lock().unwrap())
    );
}
