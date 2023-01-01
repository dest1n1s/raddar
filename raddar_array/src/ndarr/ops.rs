use std::sync::{Arc, Mutex};

use ndarray::ArrayD;

use crate::{tensor::{ops::Operation, TensorMethods, ArrayMethods}};

use super::{NdArrayTensor, NdArrayTensorInternal, ViewMethods, Element};

/// A helper method to add `grad` to the `tensor`'s grad.
///
/// If the `tensor`'s grad is `None`, we will create a new grad with the same shape and dtype as the `tensor`.
pub(crate) fn add_grad<E: Element>(tensor: Arc<Mutex<NdArrayTensorInternal<E>>>, grad: NdArrayTensor<E>) {
    let mut tensor = tensor.lock().unwrap();

    let shape = tensor.as_view().size();

    if tensor.grad.is_none() {
        tensor.grad = Some(ArrayD::<E>::zeros(shape));
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
/// `$a` and `$b` are the two tensors to borrow, type `Arc<Mutex<NdArrayTensorInternal<E>>>`.
/// 
/// `$tuple_name` is the name of the tuple to store the two borrowed internals, type `(&MutexGuard<NdArrayTensorInternal<E>>, &MutexGuard<NdArrayTensorInternal<E>>)`.
/// 
/// `$execution` is the code block to execute.
#[macro_export]
macro_rules! borrow_two_tensor_internals {
    ($a:expr, $b:expr, $tuple_name: ident, $execution: block) => {{
        let tmp_a_in_borrow_two_tensor_internals = &$a;
        let tmp_b_in_borrow_two_tensor_internals = &$b;
        let first = tmp_a_in_borrow_two_tensor_internals.lock().unwrap();

        let result = if std::sync::Arc::ptr_eq(&tmp_a_in_borrow_two_tensor_internals, &tmp_b_in_borrow_two_tensor_internals) {
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
    ($op_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to_a:expr, $backward_to_b:expr) => {
        pub(crate) struct $op_name<E: $crate::Element> {
            a: Arc<Mutex<NdArrayTensorInternal<E>>>,
            b: Arc<Mutex<NdArrayTensorInternal<E>>>,
            output: Arc<Mutex<NdArrayTensorInternal<E>>>,
        }

        impl<E: $crate::Element> $op_name<E> {
            pub fn forward($input_name: (&NdArrayTensor<E>, &NdArrayTensor<E>)) -> NdArrayTensor<E> {
                // CAUTION: decide whether the tuple is the same or not first,
                // if the tuple is the same, we should prevent locking the tensor twice.
                let (a, b) = ($input_name.0.i_copy(), $input_name.1.i_copy());

                let mut tensor = $crate::borrow_two_tensor_internals!(a, b, $input_name, {
                    NdArrayTensor::<E>::from($forward_calculation)
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

        impl<E: $crate::Element> $crate::tensor::ops::Operation<E> for $op_name <E> {
            fn backward(&self, $grad_name: NdArrayTensor<E>) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                let (a, b) = $crate::borrow_two_tensor_internals!(self.a, self.b, $input_name, {
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
        pub(crate) struct $op_name<E: $crate::Element> {
            a: Arc<Mutex<NdArrayTensorInternal<E>>>,
            output: Arc<Mutex<NdArrayTensorInternal<E>>>,
        }

        impl<E: $crate::Element> $op_name<E> {
            pub fn forward($input_name: &NdArrayTensor<E>) -> NdArrayTensor<E> {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::<E>::from($forward_calculation);
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

        impl<E: $crate::Element> Operation<E> for $op_name<E> {
            fn backward(&self, $grad_name: NdArrayTensor<E>) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                go_backward!(self.a, $backward_to);
            }
        }
    };
}

macro_rules! unary_op_with_scalar {
    ($op_name: ident, $param_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name<E: Element> {
            a: Arc<Mutex<NdArrayTensorInternal<E>>>,
            output: Arc<Mutex<NdArrayTensorInternal<E>>>,
            $param_name: E,
        }

        impl<E: Element> $op_name<E> {
            pub fn forward($input_name: &NdArrayTensor<E>, $param_name: E) -> NdArrayTensor<E> {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::<E>::from($forward_calculation);
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

        impl<E: Element> Operation<E> for $op_name<E> {
            fn backward(&self, $grad_name: NdArrayTensor<E>) {
                add_grad(self.output.clone(), $grad_name.name_clone());

                let $param_name = self.$param_name;
                go_backward!(self.a, $backward_to);
            }
        }
    };
}

macro_rules! unary_op_with_non_generic_param {
    ($op_name: ident, $param_name: ident, $param_type: ty, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to:expr) => {
        pub(crate) struct $op_name <E: Element> {
            a: Arc<Mutex<NdArrayTensorInternal<E>>>,
            output: Arc<Mutex<NdArrayTensorInternal<E>>>,
            $param_name: $param_type,
        }

        impl<E: Element> $op_name<E> {
            pub fn forward($input_name: &NdArrayTensor<E>, $param_name: $param_type) -> NdArrayTensor<E> {
                let mut tensor = {
                    let $input_name = &$input_name.i();
                    let ret = NdArrayTensor::<E>::from($forward_calculation);
                    ret
                };
                tensor.i().is_leaf = false;
                use $crate::tensor::AutoGradTensorMethods;
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

        impl<E: Element> Operation<E> for $op_name <E> {
            fn backward(&self, $grad_name: NdArrayTensor<E>) {
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
    NdArrayTensor::<E>::from(&*grad.i().as_view() * &*inputs.1.as_view()),
    NdArrayTensor::<E>::from(&*grad.i().as_view() * &*inputs.0.as_view())
);

binary_op!(
    DivOp,
    inputs,
    grad,
    &*inputs.0.as_view() / &*inputs.1.as_view(),
    NdArrayTensor::<E>::from(&*grad.i().as_view() / &*inputs.1.as_view()),
    NdArrayTensor::<E>::from(
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

unary_op_with_non_generic_param!(
    SumOp,
    axes_and_keep_dim,
    (Vec<usize>, bool),
    input,
    grad,
    input.as_view().sum_dim(&axes_and_keep_dim.0, axes_and_keep_dim.1),
    {
        // get the original shape of the input
        let input = input.lock().unwrap();
        let shape = input.as_view().size();
        // unlock the input as early as possible to avoid deadlock
        drop(input);

        let mut grad = grad;
        // if keep_dim is false, we need to unsqueeze the grad to the original shape
        if !axes_and_keep_dim.1 {
            for &axis in &axes_and_keep_dim.0{
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
pub(crate) struct GradAccumulateOp<E: Element> {
    tensor: Arc<Mutex<NdArrayTensorInternal<E>>>,
}

impl<E: Element> GradAccumulateOp<E> {
    pub(crate) fn new(tensor: Arc<Mutex<NdArrayTensorInternal<E>>>) -> Self {
        Self { tensor }
    }
}

impl<E: Element> Operation<E> for GradAccumulateOp<E> {
    fn backward(&self, grad: NdArrayTensor<E>) {
        add_grad(self.tensor.clone(), grad);
    }
}
