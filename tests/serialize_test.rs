use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use raddar::{
    assert_tensor_eq,
    core::StateDict,
    nn::{Linear, Trainable},
    seq,
};
use tch::Tensor;

#[test]
fn load_parameter_test() {
    let mut model = seq!(Linear::new(1, 1, true), Linear::new(1, 1, true),);
    let parameters = vec![
        (
            "0.weight".to_owned(),
            Arc::new(Mutex::new(Tensor::of_slice2(&[[1.0]]))),
        ),
        (
            "0.bias".to_owned(),
            Arc::new(Mutex::new(Tensor::of_slice(&[2.0]))),
        ),
        (
            "1.weight".to_owned(),
            Arc::new(Mutex::new(Tensor::of_slice2(&[[3.0]]))),
        ),
        (
            "1.bias".to_owned(),
            Arc::new(Mutex::new(Tensor::of_slice(&[2.0]))),
        ),
    ]
    .into_iter()
    .collect();
    let state_dict = StateDict::from_map(parameters);
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&Tensor::of_slice(&[1.0]));
    assert_tensor_eq!(&output, &Tensor::of_slice(&[11.0]));
}

#[test]
fn load_npz_test() {
    let mut model = seq!(Linear::new(1, 1, true), Linear::new(1, 1, true),);
    let state_dict = StateDict::from_npz(Path::new("./tests/serialize_test.npz")).unwrap();
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&Tensor::of_slice(&[2.0]));
    assert_tensor_eq!(&output, &Tensor::of_slice(&[0.1818]));
}

#[test]
fn load_ot_test() {
    let mut model = seq!(Linear::new(1, 1, true), Linear::new(1, 1, true),);
    let state_dict = StateDict::from_ot(Path::new("./tests/serialize_test.ot")).unwrap();
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&Tensor::of_slice(&[2.0]));
    assert_tensor_eq!(&output, &Tensor::of_slice(&[0.1818]));
}
