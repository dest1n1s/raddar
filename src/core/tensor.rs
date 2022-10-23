use std::sync::Arc;

use parking_lot::Mutex;
use tch::{kind::Element, Tensor};

pub type TensorCell = Arc<Mutex<Tensor>>;

pub trait Cellable {
    fn cell(self) -> TensorCell;
}

pub trait TensorIntoIter {
    fn into_iter(self) -> TensorIter;
}

pub struct TensorIter {
    tensor: Tensor,
    index: i64,
    size: i64,
}

impl TensorIntoIter for Tensor {
    fn into_iter(self) -> TensorIter {
        let size = self.size()[0];
        TensorIter {
            tensor: self,
            index: 0,
            size,
        }
    }
}

impl Iterator for TensorIter {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.size {
            let index = self.index;
            self.index += 1;
            Some(self.tensor.get(index))
        } else {
            None
        }
    }
}

impl Cellable for Tensor {
    fn cell(self) -> TensorCell {
        Arc::new(Mutex::new(self))
    }
}

/// Create a [`Vec<Arc<Tensor>>`] from an array of 1d arrays.
#[macro_export]
macro_rules! tensor_vec {
    ($($x:expr),* $(,)?) => {
        {
            use raddar::tensor;
            vec![$(std::sync::Arc::new(tensor!($x)),)*]
        }
    };
}

/// Decide if two tensors are equal.
///
/// Defaultly, the tensors are considered equal if they have the same shape and values whose MLE is less than 1e-6.
///
/// You can also explicitly specify the tolerance by passing a third argument.
#[macro_export]
macro_rules! tensor_eq {
    ($a:expr, $b:expr) => {{
        $a.size() == $b.size() && f64::from(($a - $b).square().sum(tch::Kind::Double)) < 1e-6
    }};
    ($a:expr, $b:expr, $c:expr) => {{
        $a.size() == $b.size() && f64::from(($a - $b).square().sum(tch::Kind::Double)) < $c
    }};
}

/// Assert if two tensors are equal.
///
/// Defaultly, the tensors are considered equal if they have the same shape and values whose MLE is less than 1e-6.
///
/// You can also explicitly specify the tolerance by passing a third argument.
#[macro_export]
macro_rules! assert_tensor_eq {
    ($a:expr, $b:expr) => {
        assert!(raddar::tensor_eq!($a, $b));
    };
    ($a:expr, $b:expr, $c:expr) => {
        assert!(raddar::tensor_eq!($a, $b, $c));
    };
}

pub trait ElementNestedArray<T: Element> {
    const DIMENSION: usize;
    fn shape(&self) -> [i64; Self::DIMENSION];
    fn flat(&self) -> &[T];
}

impl<T: Element, const N1: usize> ElementNestedArray<T> for [T; N1] {
    const DIMENSION: usize = 1;

    fn shape(&self) -> [i64; 1] {
        [self.len().try_into().unwrap()]
    }

    fn flat(&self) -> &[T] {
        self
    }
}

impl<T: Element, const N1: usize, const N2: usize> ElementNestedArray<T> for [[T; N2]; N1] {
    const DIMENSION: usize = 2;

    fn shape(&self) -> [i64; 2] {
        [
            self.len().try_into().unwrap(),
            self[0].len().try_into().unwrap(),
        ]
    }

    fn flat(&self) -> &[T] {
        self.flatten()
    }
}

impl<T: Element, const N1: usize, const N2: usize, const N3: usize> ElementNestedArray<T>
    for [[[T; N3]; N2]; N1]
{
    const DIMENSION: usize = 3;

    fn shape(&self) -> [i64; 3] {
        [
            self.len().try_into().unwrap(),
            self[0].len().try_into().unwrap(),
            self[0][0].len().try_into().unwrap(),
        ]
    }

    fn flat(&self) -> &[T] {
        self.flatten().flatten()
    }
}

impl<T: Element, const N1: usize, const N2: usize, const N3: usize, const N4: usize>
    ElementNestedArray<T> for [[[[T; N4]; N3]; N2]; N1]
{
    const DIMENSION: usize = 4;

    fn shape(&self) -> [i64; 4] {
        [
            self.len().try_into().unwrap(),
            self[0].len().try_into().unwrap(),
            self[0][0].len().try_into().unwrap(),
            self[0][0][0].len().try_into().unwrap(),
        ]
    }

    fn flat(&self) -> &[T] {
        self.flatten().flatten().flatten()
    }
}

impl<
        T: Element,
        const N1: usize,
        const N2: usize,
        const N3: usize,
        const N4: usize,
        const N5: usize,
    > ElementNestedArray<T> for [[[[[T; N5]; N4]; N3]; N2]; N1]
{
    const DIMENSION: usize = 5;

    fn shape(&self) -> [i64; 5] {
        [
            self.len().try_into().unwrap(),
            self[0].len().try_into().unwrap(),
            self[0][0].len().try_into().unwrap(),
            self[0][0][0].len().try_into().unwrap(),
            self[0][0][0][0].len().try_into().unwrap(),
        ]
    }

    fn flat(&self) -> &[T] {
        self.flatten().flatten().flatten().flatten()
    }
}

impl<
        T: Element,
        const N1: usize,
        const N2: usize,
        const N3: usize,
        const N4: usize,
        const N5: usize,
        const N6: usize,
    > ElementNestedArray<T> for [[[[[[T; N6]; N5]; N4]; N3]; N2]; N1]
{
    const DIMENSION: usize = 6;

    fn shape(&self) -> [i64; 6] {
        [
            self.len().try_into().unwrap(),
            self[0].len().try_into().unwrap(),
            self[0][0].len().try_into().unwrap(),
            self[0][0][0].len().try_into().unwrap(),
            self[0][0][0][0].len().try_into().unwrap(),
            self[0][0][0][0][0].len().try_into().unwrap(),
        ]
    }

    fn flat(&self) -> &[T] {
        self.flatten().flatten().flatten().flatten().flatten()
    }
}

/// Convert a multi-dimensional array to tensor
#[macro_export]
macro_rules! tensor {
    ($a:expr) => {{
        use raddar::core::ElementNestedArray;
        let data = $a;
        let shape: &[i64] = &data.shape();
        let flattened_array = data.flat();
        tch::Tensor::of_slice(flattened_array).view(shape)
    }};
}
