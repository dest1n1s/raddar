use std::sync::{Arc, Mutex};

use crate::tensor::{ops::Operation, TensorMethods};

use super::{KindedArrayD, NdArrayTensor, NdArrayTensorInternal, ViewMethods};

/// A helper method to add `grad` to the `tensor`'s grad.
///
/// If the `tensor`'s grad is `None`, we will create a new grad with the same shape and dtype as the `tensor`.
pub(crate) fn add_grad(tensor: Arc<Mutex<NdArrayTensorInternal>>, grad: NdArrayTensor) {
    let mut tensor = tensor.lock().unwrap();

    let shape = tensor.as_view().size();
    let dtype = tensor.as_view().kind();

    if tensor.grad.is_none() {
        tensor.grad = Some(KindedArrayD::zeros(&shape, dtype));
    }
    *tensor.grad.as_mut().unwrap() += &*grad.i().data.read().unwrap();
}

#[macro_export]
macro_rules! go_backward {
    ($tensor_data:expr, $backward_grad:expr) => {
        let tmp_tensor_in_go_backward = $tensor_data.lock().unwrap();
        if let Some(op) = $crate::ndarr::IntoOp::op(tmp_tensor_in_go_backward) {
            op.backward($backward_grad);
        }
    };
}

/// A helper method to borrow two tensor's internals.
/// 
/// `$a` and `$b` are the two tensors to borrow, type `Arc<Mutex<NdArrayTensorInternal>>`.
/// 
/// `$tuple_name` is the name of the tuple to store the two borrowed internals, type `(&MutexGuard<NdArrayTensorInternal>, &MutexGuard<NdArrayTensorInternal>)`.
/// 
/// `$execution` is the code block to execute.
macro_rules! borrow_two_tensor_internals {
    ($a:expr, $b:expr, $tuple_name: ident, $execution: block) => {{
        let first = $a.lock().unwrap();

        let result = if std::sync::Arc::ptr_eq(&$a, &$b) {
            let $tuple_name = (&first, &first);
            $execution
        } else {
            let second = $b.lock().unwrap();
            let $tuple_name = (&first, &second);
            let result = $execution;
            drop(second);
            result
        };
        drop(first);

        result
    }};
}

macro_rules! binary_op {
    ($op_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to_a:expr, $backward_to_b:expr) => {
        pub(crate) struct $op_name {
            a: Arc<Mutex<NdArrayTensorInternal>>,
            b: Arc<Mutex<NdArrayTensorInternal>>,
            output: Arc<Mutex<NdArrayTensorInternal>>,
        }

        impl $op_name {
            pub fn forward($input_name: (&NdArrayTensor, &NdArrayTensor)) -> NdArrayTensor {
                // CAUTION: decide whether the tuple is the same or not first,
                // if the tuple is the same, we should prevent locking the tensor twice.
                let (a, b) = ($input_name.0.i_copy(), $input_name.1.i_copy());

                let mut tensor = borrow_two_tensor_internals!(a, b, $input_name, {
                    NdArrayTensor::from($forward_calculation)
                });

                tensor.i().is_leaf = false;
                use crate::tensor::AutoGradTensorMethods;
                if $input_name.0.requires_grad() || $input_name.1.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a,
                        b,
                        output: tensor.i_copy(),
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl Operation for $op_name {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.clone());

                let (a, b) = borrow_two_tensor_internals!(self.a, self.b, $input_name, {
                    let res_a = $backward_to_a;
                    let res_b = $backward_to_b;
                    (res_a, res_b)
                });

                go_backward!(self.a, a);
                go_backward!(self.b, b);
            }
        }
    };
}

macro_rules! unary_op {
    ($op_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name {
            a: Arc<Mutex<NdArrayTensorInternal>>,
            output: Arc<Mutex<NdArrayTensorInternal>>,
        }

        impl $op_name {
            pub fn forward($input_name: &NdArrayTensor) -> NdArrayTensor {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::from($forward_calculation);
                    ret
                };
                tensor.i().is_leaf = false;
                use crate::tensor::AutoGradTensorMethods;
                if $input_name.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a: $input_name.i_copy(),
                        output: tensor.i_copy(),
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl Operation for $op_name {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.clone());

                go_backward!(self.a, $backward_to);
            }
        }
    };
}

macro_rules! unary_op_with_scalar {
    ($op_name: ident, $param_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name<T: num::NumCast + Copy + 'static> {
            a: Arc<Mutex<NdArrayTensorInternal>>,
            output: Arc<Mutex<NdArrayTensorInternal>>,
            $param_name: T,
        }

        impl<T: num::NumCast + Copy + 'static> $op_name<T> {
            pub fn forward($input_name: &NdArrayTensor, $param_name: T) -> NdArrayTensor {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::from($forward_calculation);
                    ret
                };
                tensor.i().is_leaf = false;
                use crate::tensor::AutoGradTensorMethods;
                if $input_name.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a: $input_name.i_copy(),
                        output: tensor.i_copy(),
                        $param_name,
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl<T: num::NumCast + Copy + 'static> Operation for $op_name<T> {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.clone());

                let $param_name = self.$param_name;
                go_backward!(self.a, $backward_to);
            }
        }
    };
}

unary_op!(NegOp, input, grad, -&*input.as_view(), -&grad);


binary_op!(
    AddOp,
    inputs,
    grad,
    &*inputs.0.as_view() + &*inputs.1.as_view(),
    grad.clone(),
    grad
);

binary_op!(
    SubOp,
    inputs,
    grad,
    &*inputs.0.as_view() - &*inputs.1.as_view(),
    grad.clone(),
    -&grad
);

binary_op!(
    MulOp,
    inputs,
    grad,
    &*inputs.0.as_view() * &*inputs.1.as_view(),
    NdArrayTensor::from(&*grad.i().as_view() * &*inputs.1.as_view()),
    NdArrayTensor::from(&*grad.i().as_view() * &*inputs.0.as_view())
);

binary_op!(
    DivOp,
    inputs,
    grad,
    &*inputs.0.as_view() / &*inputs.1.as_view(),
    NdArrayTensor::from(&*grad.i().as_view() / &*inputs.1.as_view()),
    NdArrayTensor::from(
        -&(&(&*grad.i().as_view() * &*inputs.0.as_view())
            / &(&*inputs.1.as_view() * &*inputs.1.as_view()))
    )
);

unary_op_with_scalar!(
    AddScalarOp,
    scalar,
    input,
    grad,
    &*input.as_view() + scalar,
    grad
);

unary_op_with_scalar!(
    SubScalarOp,
    scalar,
    input,
    grad,
    &*input.as_view() - scalar,
    grad
);

unary_op_with_scalar!(
    MulScalarOp,
    scalar,
    input,
    grad,
    &*input.as_view() * scalar,
    &grad * scalar
);

unary_op_with_scalar!(
    DivScalarOp,
    scalar,
    input,
    grad,
    &*input.as_view() / scalar,
    &grad / scalar
);

pub(crate) struct GradAccumulateOp {
    tensor: Arc<Mutex<NdArrayTensorInternal>>,
}

impl GradAccumulateOp {
    pub(crate) fn new(tensor: Arc<Mutex<NdArrayTensorInternal>>) -> Self {
        Self { tensor }
    }
}

impl Operation for GradAccumulateOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.tensor.clone(), grad);
    }
}
