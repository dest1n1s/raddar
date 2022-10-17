use std::path::Path;

use raddar::{
    assert_tensor_eq,
    core::{Cellable, StateDict},
    nn::{LinearBuilder, Trainable},
    seq, tensor,
};

#[test]
fn load_parameter_test() {
    let mut model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    let parameters = vec![
        ("0.weight".to_owned(), tensor!([[1.0]]).cell()),
        ("0.bias".to_owned(), tensor!([2.0]).cell()),
        ("1.weight".to_owned(), tensor!([[3.0]]).cell()),
        ("1.bias".to_owned(), tensor!([2.0]).cell()),
    ]
    .into_iter()
    .collect();
    let state_dict = StateDict::from_map(parameters);
    model.load_trainable_parameters(state_dict.clone());
    println!("model: {}", model.trainable_parameters());
    let output = model(&tensor!([1.0]));
    assert_tensor_eq!(&output, &tensor!([11.0]));
}

#[test]
fn read_nonexistent_parameter_test() {
    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    model
        .trainable_parameters()
        .child_state_dict("0")
        .unwrap()
        .tensor("nonexistent")
        .expect_err("Exist tensor named nonexistent in state dict");
}

#[test]
fn load_npz_test() {
    let mut model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    let state_dict = StateDict::from_npz(Path::new("./tests/serialize_test.npz")).unwrap();
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&tensor!([2.0f32]));
    assert_tensor_eq!(&output, &tensor!([0.1818f32]));
}

#[test]
fn load_ot_test() {
    let mut model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    let state_dict = StateDict::from_ot(Path::new("./tests/serialize_test.ot")).unwrap();
    model.load_trainable_parameters(state_dict.clone());
    let output = model(&tensor!([2.0f32]));
    assert_tensor_eq!(&output, &tensor!([0.1818f32]));
}
