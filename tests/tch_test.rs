use raddar::tensor;

#[test]
fn device_test() {
    tch::maybe_init_cuda();
    println!("Cuda: {}", tch::Cuda::is_available());
    println!("Cudnn: {}", tch::Cuda::cudnn_is_available());
}

#[test]
fn grad_test() {
    let t = tensor!([3.0, 1.0]);
    let mut t0 = t.get(0);
    t0 += 1;
    println!("t: {:#?}", t);
}
