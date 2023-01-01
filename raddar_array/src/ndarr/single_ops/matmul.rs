use std::sync::{Arc, Mutex};

use crate::{
    binary_op, go_backward,
    ndarr::{
        array_ops::BroadcastOp, ops::add_grad, Element, NdArrayTensor, NdArrayTensorInternal,
        ViewMethods,
    },
    tensor::index::IndexInfo,
};
use more_asserts::assert_ge;
use ndarray::{ArrayD, ArrayViewD, Ix1, Ix2, SliceInfoElem};
use state_compose::StateCompose;

fn bivector_mul<E: Element>(a: ArrayViewD<'_, E>, b: ArrayViewD<'_, E>) -> ArrayD<E> {
    assert_eq!(a.ndim(), 1);
    assert_eq!(b.ndim(), 1);

    let a = a.into_dimensionality::<Ix1>().unwrap();
    let b = b.into_dimensionality::<Ix1>().unwrap();

    let result = a.dot(&b);

    let mut res = ArrayD::<E>::zeros(vec![1]);
    res += result;

    res
}

fn bivector_mul_matrix<E: Element>(
    bivec: ArrayViewD<'_, E>,
    mat: ArrayViewD<'_, E>,
    bivec_first: bool,
) -> ArrayD<E> {
    assert_eq!(bivec.ndim(), 1);
    assert_eq!(mat.ndim(), 2);

    let bivec = bivec.into_dimensionality::<Ix1>().unwrap();
    let mat = mat.into_dimensionality::<Ix2>().unwrap();

    let result = if bivec_first {
        bivec.dot(&mat)
    } else {
        mat.dot(&bivec)
    };

    ArrayD::<E>::from(result.into_dyn())
}

fn matrix_mul<E: Element>(a: ArrayViewD<'_, E>, b: ArrayViewD<'_, E>) -> ArrayD<E> {
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);

    let a = a.into_dimensionality::<Ix2>().unwrap();
    let b = b.into_dimensionality::<Ix2>().unwrap();

    let result = a.dot(&b);

    ArrayD::from(result.into_dyn())
}

fn broadcasted_matmul<E: Element>(a: ArrayViewD<'_, E>, b: ArrayViewD<'_, E>) -> ArrayD<E> {
    assert_ge!(a.ndim(), 2);
    assert_ge!(b.ndim(), 2);
    assert!(
        a.ndim() > 2 || b.ndim() > 2,
        "At least one of the arguments must have ndim > 2. Otherwise, use matrix_mul instead."
    );

    let broadcasted_shape = BroadcastOp::<E>::cobroadcast_shape(a.shape(), b.shape());

    let a = a.broadcast(broadcasted_shape.clone()).unwrap();
    let b = b.broadcast(broadcasted_shape.clone()).unwrap();

    let mut result = ArrayD::zeros(&*broadcasted_shape);

    // a naive implementation would be to iterate over the last two dimensions of a and b and
    // multiply them. However, this is not efficient, because it does not take advantage of
    // accelerated matrix multiplication routines. Reshape the arrays to 2D and then multiply
    // them if possible in the future.

    let broadcast_ndim = broadcasted_shape.len() - 2;
    let states = StateCompose::new(&broadcasted_shape[..broadcast_ndim]);

    for state in states.iter() {
        let slice: Vec<SliceInfoElem> = (0..broadcasted_shape.len())
            .into_iter()
            .map(|i| {
                if i < broadcast_ndim {
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

        let result_slice = matrix_mul(a_slice, b_slice);

        result
            .view_mut()
            .slice_mut(slice.as_slice())
            .assign(&result_slice.view());
    }

    result
}

pub(crate) fn matmul<E: Element>(a: ArrayViewD<'_, E>, b: ArrayViewD<'_, E>) -> ArrayD<E>
where
    ArrayD<E>: From<ArrayD<E>>,
{
    match (a.ndim(), b.ndim()) {
        (1, 1) => bivector_mul(a, b),
        (1, 2) => bivector_mul_matrix(a, b, true),
        (2, 1) => bivector_mul_matrix(b, a, false),
        (2, 2) => matrix_mul(a, b),
        _ => broadcasted_matmul(a, b),
    }
}

fn backward<E: Element>(
    grad: &ArrayViewD<'_, E>,
    a: &ArrayViewD<'_, E>,
    b: &ArrayViewD<'_, E>,
    for_a: bool,
) -> NdArrayTensor<E> {
    let res = match (a.size().len(), b.size().len()) {
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
        (_, _) => unimplemented!("matmul backward for ndim > 2"),
    };

    res.into()
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

        pub(crate) fn decode(&self, state: usize, pos: usize) -> usize {
            state / self.precomputed[pos] % self.shape[pos]
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = usize> {
            (0..self.shape.iter().product()).into_iter()
        }
    }
}
