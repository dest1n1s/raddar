use tch::{Device, Tensor};

#[test]
fn device_test() {
    tch::maybe_init_cuda();
    println!("Cuda: {}", tch::Cuda::is_available());
    println!("Cudnn: {}", tch::Cuda::cudnn_is_available());
}

#[test]
fn grad_test() {
    let t = Tensor::of_slice(&[3.0, 1.0]).to(Device::Cuda(0)).set_requires_grad(true);
    let p = t.get(0) * t.get(1) + t.get(0) + t.get(1);
    p.backward();
    let dp_over_dt = t.grad();
    assert_eq!(Vec::<f64>::from(&dp_over_dt), [2.0, 4.0]);
}