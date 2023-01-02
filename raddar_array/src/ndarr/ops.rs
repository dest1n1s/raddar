use std::sync::{Arc, Mutex, Weak};

use num::cast;

use crate::{
    ndarr::BorrowView,
    tensor::{ops::Operation, ArrayMethods, TensorMethods},
};

use super::{KindedArrayD, NdArrayTensor, NdArrayTensorInternal, ViewMethods};

/// A helper method to add `grad` to the `tensor`'s grad.
///
/// If the `tensor`'s grad is `None`, we will create a new grad with the same shape and dtype as the `tensor`.
pub(crate) fn add_grad(tensor: Weak<Mutex<NdArrayTensorInternal>>, grad: NdArrayTensor) {
    let tensor = tensor.upgrade();
    if tensor.is_none() {
        return;
    }
    let tensor = tensor.unwrap();
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
#[macro_export]
macro_rules! borrow_two_tensor_internals {
    ($a:expr, $b:expr, $tuple_name: ident, $execution: block) => {{
        let tmp_a_in_borrow_two_tensor_internals = &$a;
        let tmp_b_in_borrow_two_tensor_internals = &$b;
        let first = tmp_a_in_borrow_two_tensor_internals.lock().unwrap();

        let result = if std::sync::Arc::ptr_eq(
            &tmp_a_in_borrow_two_tensor_internals,
            &tmp_b_in_borrow_two_tensor_internals,
        ) {
            let $tuple_name = (&first, &first);
            $execution
        } else {
            let second = tmp_b_in_borrow_two_tensor_internals.lock().unwrap();
            let $tuple_name = (&first, &second);
            let result = $execution;
            drop(second);
            result
        };
        drop(first);

        result
    }};
}

#[macro_export]
macro_rules! binary_op {
    ($op_name: ident, $input_name:ident, $grad_name:ident,  $forward_calculation:expr, $backward_to_a:expr, $backward_to_b:expr) => {
        $crate::binary_op!(
            $op_name,
            $input_name,
            $grad_name,
            output,
            $forward_calculation,
            $backward_to_a,
            $backward_to_b
        );
    };
    ($op_name: ident, $input_name:ident, $grad_name:ident, $output_name:ident, $forward_calculation:expr, $backward_to_a:expr, $backward_to_b:expr) => {
        pub(crate) struct $op_name {
            a: std::sync::Arc<Mutex<NdArrayTensorInternal>>,
            b: std::sync::Arc<Mutex<NdArrayTensorInternal>>,
            $output_name: std::sync::Weak<Mutex<NdArrayTensorInternal>>,
        }

        impl $op_name {
            pub fn forward($input_name: (&NdArrayTensor, &NdArrayTensor)) -> NdArrayTensor {
                // CAUTION: decide whether the tuple is the same or not first,
                // if the tuple is the same, we should prevent locking the tensor twice.
                let (a, b) = ($input_name.0.i_copy(), $input_name.1.i_copy());

                let mut tensor = $crate::borrow_two_tensor_internals!(a, b, $input_name, {
                    NdArrayTensor::from($forward_calculation)
                });

                tensor.i().is_leaf = false;
                use crate::tensor::AutoGradTensorMethods;
                if $input_name.0.requires_grad() || $input_name.1.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a,
                        b,
                        $output_name: tensor.i_ref(),
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl $crate::tensor::ops::Operation for $op_name {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                let (a, b) = $crate::borrow_two_tensor_internals!(self.a, self.b, $input_name, {
                    let $output_name = &self.$output_name.upgrade().expect("output is dropped");
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
            output: Weak<Mutex<NdArrayTensorInternal>>,
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
                        output: tensor.i_ref(),
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl Operation for $op_name {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                go_backward!(self.a, $backward_to);
            }
        }
    };
}

macro_rules! unary_op_with_scalar {
    ($op_name: ident, $param_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        unary_op_with_scalar!(
            $op_name,
            $param_name,
            $input_name,
            $grad_name,
            output,
            $forward_calculation,
            $backward_to
        );
    };
    ($op_name: ident, $param_name: ident, $input_name:ident, $grad_name:ident, $output_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name<T: $crate::AnyNum> {
            a: std::sync::Arc<Mutex<NdArrayTensorInternal>>,
            $output_name: std::sync::Weak<Mutex<NdArrayTensorInternal>>,
            $param_name: T,
        }

        impl<T: $crate::AnyNum> $op_name<T> {
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
                        output: tensor.i_ref(),
                        $param_name,
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl<T: $crate::AnyNum> Operation for $op_name<T> {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                let $param_name = self.$param_name;
                let $output_name = &self.$output_name.upgrade().expect("output is dropped");
                let $input_name = &self.a;
                go_backward!(self.a, $backward_to);
            }
        }
    };
}

macro_rules! unary_op_with_non_generic_param {
    ($op_name: ident, $param_name: ident, $param_type: ty, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name {
            a: Arc<Mutex<NdArrayTensorInternal>>,
            output: Weak<Mutex<NdArrayTensorInternal>>,
            $param_name: $param_type,
        }

        impl $op_name {
            pub fn forward($input_name: &NdArrayTensor, $param_name: $param_type) -> NdArrayTensor {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::from($forward_calculation);
                    ret
                };
                tensor.i().is_leaf = false;
                use $crate::tensor::AutoGradTensorMethods;
                if $input_name.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a: $input_name.i_copy(),
                        output: tensor.i_ref(),
                        $param_name,
                    }));
                    tensor.set_requires_grad(true);
                }
                tensor
            }
        }

        impl Operation for $op_name {
            fn backward(&self, $grad_name: NdArrayTensor) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                let $param_name = &self.$param_name;
                let $input_name = &self.a;
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

binary_op!(
    PowOp,
    inputs,
    grad,
    output,
    inputs.0.as_view().pow(&*inputs.1.as_view()),
    {
        // todo: is this implementation correct?
        let a = &*inputs.0.as_view();
        let b = &*inputs.1.as_view();
        let a_pow_b_minus_1 = a.pow((b - 1.0).view());
        let b_times_a_pow_b_minus_1 = b * &a_pow_b_minus_1.view();
        NdArrayTensor::from(&*grad.i().as_view() * &b_times_a_pow_b_minus_1.view())
    },
    {
        let a = &*inputs.0.as_view();
        let a_log = a.ln();
        let a_pow_b = output.lock().unwrap();
        let a_pow_b_times_log_a = &*a_pow_b.as_view() * &a_log.view();
        drop(a_pow_b);

        NdArrayTensor::from(&*grad.i().as_view() * &a_pow_b_times_log_a.view())
    }
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

unary_op_with_scalar!(
    PowScalarOp,
    scalar,
    input,
    grad,
    input.as_view().pow_scalar(scalar),
    {
        let a = input.lock().unwrap();
        let a_pow_b_minus_1 = a.as_view().pow_scalar(scalar - T::one());
        drop(a);

        let b_times_a_pow_b_minus_1 = &a_pow_b_minus_1.view() * scalar;

        NdArrayTensor::from(&*grad.i().as_view() * &b_times_a_pow_b_minus_1.view())
    }
);

unary_op_with_scalar!(
    ExpScalarOp,
    scalar,
    input,
    grad,
    output,
    input.as_view().exp_scalar(scalar),
    {
        let scalar: f64 = cast(scalar).unwrap();

        let output = output.lock().unwrap();
        let output_times_ln_scalar = output.as_view().mul_scalar(scalar.ln());
        drop(output);

        NdArrayTensor::from(&*grad.i().as_view() * &output_times_ln_scalar.view())
    }
);

unary_op_with_scalar!(
    LogScalarOp,
    scalar,
    input,
    grad,
    input.as_view().log_scalar(scalar),
    {
        let scalar: f64 = cast(scalar).unwrap();
        let log_scalar = scalar.ln();
        let a = input.lock().unwrap();
        let a_times_log_scalar = a.as_view().mul_scalar(log_scalar);
        drop(a);

        let inv = a_times_log_scalar.view().pow_scalar(-1);

        NdArrayTensor::from(&*grad.i().as_view() * &inv.view())
    }
);

unary_op_with_non_generic_param!(
    SumOp,
    axes_and_keep_dim,
    (Vec<usize>, bool),
    input,
    grad,
    input
        .as_view()
        .sum_dim(&axes_and_keep_dim.0, axes_and_keep_dim.1),
    {
        // get the original shape of the input
        let input = input.lock().unwrap();
        let shape = input.as_view().size();
        // unlock the input as early as possible to avoid deadlock
        drop(input);

        let mut grad = grad;
        // if keep_dim is false, we need to unsqueeze the grad to the original shape
        if !axes_and_keep_dim.1 {
            for &axis in &axes_and_keep_dim.0 {
                grad.unsqueeze_(axis);
            }
        }

        // broadcast the grad to the original shape
        grad.broadcast(&shape)
    }
);

unary_op_with_non_generic_param!(
    SqueezeOp,
    dim,
    usize,
    input,
    grad,
    input.as_view().clone().into_squeeze(dim).upgrade(),
    grad.unsqueeze(*dim)
);

unary_op_with_non_generic_param!(
    UnsqueezeOp,
    dim,
    usize,
    input,
    grad,
    input.as_view().clone().into_unsqueeze(dim).upgrade(),
    grad.squeeze(*dim)
);
pub(crate) struct GradAccumulateOp {
    tensor: Weak<Mutex<NdArrayTensorInternal>>,
}

impl GradAccumulateOp {
    pub(crate) fn new(tensor: Weak<Mutex<NdArrayTensorInternal>>) -> Self {
        Self { tensor }
    }
}

impl Operation for GradAccumulateOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.tensor.clone(), grad);
    }
}
