use std::sync::Arc;

use raddar::{
    assert_tensor_eq,
    dataset::{DataLoaderConfigBuilder, Dataset, TensorDataset},
    tensor, tensor_vec,
};
use tch::Tensor;

#[test]
fn dataset_test() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = TensorDataset::from_tensors(inputs, labels);

    let mut iter = dataset
        .into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(3)
                .build()
                .unwrap(),
        )
        .cycle();
    iter.next();
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [2, 1]);

    let (batch, _) = iter.next().unwrap();
    assert_eq!(batch.size(), [3, 1]);
}

#[test]
fn dataset_mapping_test() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = TensorDataset::from_tensors(inputs, labels);

    let mapped_dataset: TensorDataset = dataset.map(|(x, y): (Arc<Tensor>, Arc<Tensor>)| {
        (Arc::new(x.copy() * 2.0), Arc::new(y.copy() * 2.0))
    });
    let mapped_dataset: TensorDataset =
        mapped_dataset.map_batch(|(x, y): (Tensor, Tensor)| (x * 2.0, y * 2.0), 3);
    let mut iter = mapped_dataset
        .into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(3)
                .build()
                .unwrap(),
        )
        .cycle();
    let (batch, _) = iter.next().unwrap();
    assert_tensor_eq!(batch, tensor!([[4.0], [12.0], [20.0]]));
    let (batch, _) = iter.next().unwrap();
    assert_tensor_eq!(batch, tensor!([[16.0], [32.0], [40.0]]));
    let (batch, _) = iter.next().unwrap();
    assert_tensor_eq!(batch, tensor!([[8.0], [24.0]]));
    let (batch, _) = iter.next().unwrap();
    assert_tensor_eq!(batch, tensor!([[4.0], [12.0], [20.0]]));
}

#[test]
fn dataset_shuffle_test() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = TensorDataset::from_tensors(inputs, labels);

    let mut iter = dataset
        .into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(3)
                .shuffle(true)
                .build()
                .unwrap(),
        )
        .cycle();
    let (batch, _) = iter.next().unwrap();
    batch.print();
    let (batch, _) = iter.next().unwrap();
    batch.print();
    let (batch, _) = iter.next().unwrap();
    batch.print();
    let (batch, _) = iter.next().unwrap();
    batch.print();
    let (batch, _) = iter.next().unwrap();
    batch.print();
}

#[test]
fn dataset_augment_test() {
    let inputs = tensor_vec![[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]];
    let labels = tensor_vec![[4.0], [10.0], [16.], [13.0], [25.], [31.], [7.], [19.0]];
    let dataset = TensorDataset::from_tensors(inputs, labels);

    let mut iter = dataset
        .augment::<TensorDataset, _, _>(|(x, y): (Arc<Tensor>, Arc<Tensor>)| {
            vec![
                (Arc::new(x.copy() * 2.0), Arc::new(y.copy() * 2.0)),
                (Arc::new(x.copy() * 2.0), Arc::new(y.copy() * 2.0)),
            ]
        })
        .into_loader(
            DataLoaderConfigBuilder::default()
                .batch_size(7)
                .build()
                .unwrap(),
        )
        .cycle();
    let (batch, _) = iter.next().unwrap();
    assert_tensor_eq!(batch, tensor!([[2.0], [2.0], [6.0], [6.0], [10.0], [10.0], [8.0]]));
}
