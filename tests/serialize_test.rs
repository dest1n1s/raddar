use std::path::Path;

use raddar::{
    assert_tensor_eq,
    core::Cellable,
    nn::{LinearBuilder, Trainable},
    seq, tensor,
};

#[test]
fn load_parameter_test() {
    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    let state_dict = vec![
        ("0.weight".to_owned(), tensor!([[1.0]]).cell()),
        ("0.bias".to_owned(), tensor!([2.0]).cell()),
        ("1.weight".to_owned(), tensor!([[3.0]]).cell()),
        ("1.bias".to_owned(), tensor!([2.0]).cell()),
    ]
    .into_iter()
    .collect();
    model.load(state_dict);
    println!("model: {:#?}", model.parameters());
    let output = model(&tensor!([1.0]));
    assert_tensor_eq!(&output, &tensor!([11.0]));
}

#[test]
fn load_npz_test() {
    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    model.load_npz(Path::new("./tests/serialize_test.npz")).unwrap();
    let output = model(&tensor!([2.0f32]));
    assert_tensor_eq!(&output, &tensor!([0.1818f32]));
}

#[test]
fn load_ot_test() {
    let model = seq!(
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
        LinearBuilder::default().input_dim(1).output_dim(1).build(),
    );
    model.load_ot("./tests/serialize_test.ot").unwrap();
    let output = model(&tensor!([2.0f32]));
    assert_tensor_eq!(&output, &tensor!([0.1818f32]));
}
