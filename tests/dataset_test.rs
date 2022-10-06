use raddar::{dataset::Dataset, tensor_vec};

#[test]
fn dataset_test() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = Dataset::from_tensors(inputs, labels, 3);

    let mut iter = dataset.iter();
    iter.next();
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [2, 1]);

    let mut iter = dataset.iter();
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
}
