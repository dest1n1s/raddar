use std::sync::Arc;

use crate::{
    binary_op, go_backward,
    ndarr::{
        array_ops::BroadcastOp, kinded_batched_zip, ops::add_grad, BorrowView, KindedArrayD,
        KindedArrayViewD, NdArrayTensor, NdArrayTensorInternal, ViewMethods, ViewMutMethods,
    },
    tensor::{TensorKind, TensorMethods},
    AnyNum,
};
use more_asserts::{assert_ge, assert_lt};
use ndarray::{ArrayD, ArrayViewD, Ix1, Ix2, LinalgScalar, SliceInfoElem};
use state_compose::StateCompose;

/// Multiply two bivectors.
fn bivector_mul<T: LinalgScalar + AnyNum>(
    a: ArrayViewD<'_, T>,
    b: ArrayViewD<'_, T>,
    kind: TensorKind,
) -> KindedArrayD {
    assert_eq!(a.ndim(), 1);
    assert_eq!(b.ndim(), 1);

    let a = a.into_dimensionality::<Ix1>().unwrap();
    let b = b.into_dimensionality::<Ix1>().unwrap();

    let result = a.dot(&b);

    let mut res = KindedArrayD::zeros(&[1], kind);
    res += result;

    res
}

/// Multiply a bivector with a matrix.
/// 
/// If `bivec_first` is true, the bivector is multiplied with the matrix from the left.
/// Otherwise, the bivector is multiplied with the matrix from the right.
fn bivector_mul_matrix<T: LinalgScalar + AnyNum>(
    bivec: ArrayViewD<'_, T>,
    mat: ArrayViewD<'_, T>,
    bivec_first: bool,
) -> KindedArrayD
where
    KindedArrayD: From<ArrayD<T>>,
{
    assert_eq!(bivec.ndim(), 1);
    assert_eq!(mat.ndim(), 2);

    let bivec = bivec.into_dimensionality::<Ix1>().unwrap();
    let mat = mat.into_dimensionality::<Ix2>().unwrap();

    let result = if bivec_first {
        bivec.dot(&mat)
    } else {
        mat.dot(&bivec)
    };

    KindedArrayD::from(result.into_dyn())
}

/// Multiply two 2d matrices.
fn matrix_mul<T: LinalgScalar + AnyNum>(a: ArrayViewD<'_, T>, b: ArrayViewD<'_, T>) -> KindedArrayD
where
    KindedArrayD: From<ArrayD<T>>,
{
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);

    let a = a.into_dimensionality::<Ix2>().unwrap();
    let b = b.into_dimensionality::<Ix2>().unwrap();

    let result = a.dot(&b);

    KindedArrayD::from(result.into_dyn())
}

/// Apply function `f` to each slice of `a` and `b` and return the result.
///
/// The slice's length will be thought to be the same as `element_shape`'s.
///
/// For example, if `a` and `b` are 3d tensors with shape `[2, 3, 4]` and `element_shape` is `[2, 2]`,
/// `f` will be applied to each 2d slice of `a` and `b` with shape `[3, 4]` and expected to return a
/// 2d tensor with shape `[2, 2]`;
///
/// if `element_shape` is `[1]`, `f` will be applied to each 1d slice of `a` and `b` with shape `[4]`
/// and expected to return a 1d tensor with shape `[1]`.
///
/// The result of `f` will be concatenated along the batch dimensions and returned.
pub(crate) fn batched_zip<T: LinalgScalar + AnyNum, F>(
    a: ArrayViewD<'_, T>,
    b: ArrayViewD<'_, T>,
    kind: TensorKind,
    element_shape: &[usize],
    f: F,
) -> KindedArrayD
where
    KindedArrayD: From<ArrayD<T>>,
    F: Fn(ArrayViewD<'_, T>, ArrayViewD<'_, T>, Vec<SliceInfoElem>) -> KindedArrayD,
{
    assert_eq!(a.ndim(), b.ndim());
    let ndim = a.ndim();
    let element_ndim = element_shape.len();
    assert_lt!(element_ndim, ndim);
    let batch_ndim = ndim - element_ndim;
    let batch_shape = &a.shape()[..batch_ndim];
    assert_eq!(batch_shape, &b.shape()[..batch_ndim]);

    let result_shape = batch_shape
        .iter()
        .chain(element_shape.iter())
        .copied()
        .collect::<Vec<_>>();
    let mut result = KindedArrayD::empty(&result_shape, kind);

    let states = StateCompose::new(batch_shape);

    for state in states.iter() {
        let slice: Vec<SliceInfoElem> = (0..ndim)
            .into_iter()
            .map(|i| {
                if i < batch_ndim {
                    SliceInfoElem::Index(states.decode(state, i) as isize)
                } else {
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();

        let a_slice = a.slice(slice.as_slice());
        let b_slice = b.slice(slice.as_slice());

        let result_slice = f(a_slice, b_slice, slice.clone());

        result
            .view_mut()
            .slice_mut(slice.into())
            .assign(&result_slice.view());
    }

    result
}

/// Multiply two tensors with ndim > 2.
///
/// The last two dimensions of `a` and `b` are treated as matrices and multiplied.
///
/// The batch dimensions of `a` and `b` must be the same. The result will have the same batch
/// dimensions as `a` and `b`.
fn broadcasted_matmul<T: LinalgScalar + AnyNum>(
    a: ArrayViewD<'_, T>,
    b: ArrayViewD<'_, T>,
    kind: TensorKind,
) -> KindedArrayD
where
    KindedArrayD: From<ArrayD<T>>,
{
    assert_ge!(a.ndim(), 2);
    assert_ge!(b.ndim(), 2);
    assert!(
        a.ndim() > 2 || b.ndim() > 2,
        "At least one of the arguments must have ndim > 2. Otherwise, use matrix_mul instead."
    );

    let batched_a_shape = &a.shape()[..a.ndim() - 2];
    let batched_b_shape = &b.shape()[..b.ndim() - 2];
    let first_dim = a.shape()[a.ndim() - 2];
    let second_dim = b.shape()[b.ndim() - 1];
    let element_shape = [first_dim, second_dim];
    let broadcasted_batched_shape =
        BroadcastOp::cobroadcast_shape(batched_a_shape, batched_b_shape);

    let broadcasted_a_shape = broadcasted_batched_shape
        .iter()
        .chain(&a.shape()[a.ndim() - 2..])
        .copied()
        .collect::<Vec<_>>();

    let broadcasted_b_shape = broadcasted_batched_shape
        .iter()
        .chain(&b.shape()[b.ndim() - 2..])
        .copied()
        .collect::<Vec<_>>();

    let a = a.broadcast(broadcasted_a_shape).unwrap();
    let b = b.broadcast(broadcasted_b_shape).unwrap();

    batched_zip(a, b, kind, &element_shape, |a, b, _| matrix_mul(a, b))
}

/// Multiply two tensors.
///
/// if `a` and `b` are both 1d tensors, the result is a scalar;
/// if `a` is a 1d tensor and `b` is a 2d tensor, the result is a 1d tensor;
/// if `a` is a 2d tensor and `b` is a 1d tensor, the result is a 1d tensor, too;
/// if `a` and `b` are both 2d tensors, the result is a 2d tensor.
/// if one of `a` and `b`'s ndim > 2, they are treated as batched matrices.
pub(crate) fn matmul<T: LinalgScalar + AnyNum>(
    a: ArrayViewD<'_, T>,
    b: ArrayViewD<'_, T>,
    kind: TensorKind,
) -> KindedArrayD
where
    KindedArrayD: From<ArrayD<T>>,
{
    match (a.ndim(), b.ndim()) {
        (1, 1) => bivector_mul(a, b, kind),
        (1, 2) => bivector_mul_matrix(a, b, true),
        (2, 1) => bivector_mul_matrix(b, a, false),
        (2, 2) => matrix_mul(a, b),
        _ => broadcasted_matmul(a, b, kind),
    }
}

/// The backward pass for `matmul`.
fn backward(
    grad: &KindedArrayViewD<'_>,
    a: &KindedArrayViewD<'_>,
    b: &KindedArrayViewD<'_>,
    for_a: bool,
) -> NdArrayTensor {
    fn backward_to_array(
        grad: &KindedArrayViewD<'_>,
        a: &KindedArrayViewD<'_>,
        b: &KindedArrayViewD<'_>,
        for_a: bool,
    ) -> KindedArrayD {
        match (a.size().len(), b.size().len()) {
            (1, 1) => {
                if for_a {
                    grad * b
                } else {
                    grad * a
                }
            }
            (1, 2) => {
                // A(M), B(M,N) -> grad(N)
                if for_a {
                    // (M,N) * (N) -> (M)
                    b.matmul(grad)
                } else {
                    // (M) -> (M,1)
                    let expand_a = a.clone().into_unsqueeze(1);
                    // (N) -> (1,N)
                    let expand_grad = grad.clone().into_unsqueeze(0);
                    // (M,1) * (1,N) -> (M,N)
                    expand_a.matmul(expand_grad)
                }
            }
            (2, 1) => {
                // A(M,N), B(N) -> grad(M)
                if for_a {
                    // (N) -> (1,N)
                    let expand_b = b.clone().into_unsqueeze(0);
                    // (M) -> (M,1)
                    let expand_grad = grad.clone().into_unsqueeze(1);
                    // (M,1) * (1,N) -> (M,N)
                    expand_grad.matmul(expand_b)
                } else {
                    // (M) * (M,N) -> (N)
                    grad.matmul(a)
                }
            }
            (2, 2) => {
                // A(M,N), B(N,Q) -> grad(M,Q)
                if for_a {
                    // (M,Q) * (Q,N) -> (M,N)
                    grad.matmul(b.t())
                } else {
                    // (N,M) * (M,Q) -> (N,Q)
                    a.t().matmul(grad)
                }
            }
            (_, _) => {
                let shape = if for_a { a.size() } else { b.size() };
                let element_shape = &shape[shape.len() - 2..];

                kinded_batched_zip(a, b, element_shape, |a, b, slice| {
                    let grad_slice = grad.slice(slice.into());
                    backward_to_array(&grad_slice, &a, &b, for_a)
                })
            }
        }
    }

    backward_to_array(grad, a, b, for_a).into()
}

binary_op!(
    MatmulOp,
    inputs,
    grad,
    inputs.0.as_view().matmul(&*inputs.1.as_view()),
    backward(
        &*grad.i().as_view(),
        &*inputs.0.as_view(),
        &*inputs.1.as_view(),
        true
    ),
    backward(
        &*grad.i().as_view(),
        &*inputs.0.as_view(),
        &*inputs.1.as_view(),
        false
    )
);

mod state_compose {
    /// A helper struct to compose and decompose states. A state is a number that represents a
    /// combination of multiple values.
    ///
    /// For example, if `shape` is \[2, 3, 4\], then
    /// ```text
    /// State 0 -> [0, 0, 0]
    /// State 1 -> [0, 0, 1]
    /// State 2 -> [0, 0, 2]
    /// State 3 -> [0, 0, 3]
    /// ...
    /// State 23 -> [1, 2, 3]
    /// ```
    /// You can use `decode` to decompose a state into its components. With the same `shape`,
    /// `decode(23, 0) == 1`, `decode(23, 1) == 2`, `decode(23, 2) == 3`.
    pub(crate) struct StateCompose {
        shape: Vec<usize>,
        precomputed: Vec<usize>,
    }

    impl StateCompose {
        pub(crate) fn new(shape: &[usize]) -> Self {
            let mut precomputed = Vec::with_capacity(shape.len());

            let mut cur = 1;
            precomputed.push(cur);
            for i in shape.iter().rev().take(shape.len() - 1) {
                cur *= i;
                precomputed.push(cur);
            }

            precomputed.reverse();

            Self {
                shape: shape.to_vec(),
                precomputed,
            }
        }

        /// `O(1)` time to decompose a state into its components.
        pub(crate) fn decode(&self, state: usize, pos: usize) -> usize {
            state / self.precomputed[pos] % self.shape[pos]
        }

        /// Return an iterator over all possible states.
        pub(crate) fn iter(&self) -> impl Iterator<Item = usize> {
            (0..self.shape.iter().product()).into_iter()
        }
    }
}
