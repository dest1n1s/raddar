use std::sync::{Arc, Mutex};

use crate::tensor::{ops::Operation, TensorMethods};

use super::{KindedArrayD, NdArrayTensor, NdArrayTensorInternal, ViewMethods};

pub(crate) fn add_grad(tensor: Arc<Mutex<NdArrayTensorInternal>>, grad: NdArrayTensor) {
    let mut tensor = tensor.lock().unwrap();

    let shape = tensor.as_view().size();
    let dtype = tensor.as_view().kind();

    if tensor.grad.is_none() {
        tensor.grad = Some(KindedArrayD::zeros(&shape, dtype));
    }
    tensor
        .grad
        .as_mut()
        .unwrap()
        .add_(&*grad.i().data.read().unwrap());
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

macro_rules! binary_op {
    ($op_name: ident, $input_name:ident, $grad_name:ident, $forward_calculation:expr, $backward_to_a:expr, $backward_to_b:expr) => {
        pub(crate) struct $op_name {
            a: Arc<Mutex<NdArrayTensorInternal>>,
            b: Arc<Mutex<NdArrayTensorInternal>>,
            output: Arc<Mutex<NdArrayTensorInternal>>,
        }

        impl $op_name {
            pub fn forward($input_name: (&NdArrayTensor, &NdArrayTensor)) -> NdArrayTensor {
                let mut tensor = NdArrayTensor::from($forward_calculation);
                tensor.i().is_leaf = false;
                use crate::tensor::AutoGradTensorMethods;
                if $input_name.0.requires_grad() || $input_name.1.requires_grad() {
                    tensor.i().op = Some(Arc::new($op_name {
                        a: $input_name.0.i_copy(),
                        b: $input_name.1.i_copy(),
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

                let $input_name = (&self.a.lock().unwrap(), &self.b.lock().unwrap());

                go_backward!(self.a, $backward_to_a);
                go_backward!(self.b, $backward_to_b);
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
                let mut tensor = NdArrayTensor::from($forward_calculation);
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
                let mut tensor = NdArrayTensor::from($forward_calculation);
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

unary_op!(NegOp, input, grad, -&*input.i().as_view(), -&grad);

binary_op!(
    AddOp,
    inputs,
    grad,
    &*inputs.0.i().as_view() + &*inputs.1.i().as_view(),
    grad.clone(),
    grad
);

binary_op!(
    SubOp,
    inputs,
    grad,
    &*inputs.0.i().as_view() - &*inputs.1.i().as_view(),
    grad.clone(),
    -&grad
);

binary_op!(
    MulOp,
    inputs,
    grad,
    &*inputs.0.i().as_view() * &*inputs.1.i().as_view(),
    NdArrayTensor::from(&*grad.i().as_view() * &*inputs.1.as_view()),
    NdArrayTensor::from(&*grad.i().as_view() * &*inputs.0.as_view())
);

binary_op!(
    DivOp,
    inputs,
    grad,
    &*inputs.0.i().as_view() / &*inputs.1.i().as_view(),
    NdArrayTensor::from(&*grad.i().as_view() / &*inputs.1.as_view()),
    NdArrayTensor::from(
        &(&*grad.i().as_view() * &*inputs.0.as_view())
            / &(&*inputs.1.as_view() * &*inputs.1.as_view())
    )
);

unary_op_with_scalar!(
    AddScalarOp,
    scalar,
    input,
    grad,
    &*input.i().as_view() + scalar,
    grad
);

unary_op_with_scalar!(
    SubScalarOp,
    scalar,
    input,
    grad,
    &*input.i().as_view() - scalar,
    grad
);

unary_op_with_scalar!(
    MulScalarOp,
    scalar,
    input,
    grad,
    &*input.i().as_view() * scalar,
    &grad * scalar
);

unary_op_with_scalar!(
    DivScalarOp,
    scalar,
    input,
    grad,
    &*input.i().as_view() / scalar,
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
