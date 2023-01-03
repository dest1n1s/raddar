use std::sync::{Arc, Mutex, Weak};

use crate::{
    ndarr::{
        AsView, KindedArrayViewD, KindedArrayViewMutD, NdArrayTensor, NdArrayTensorInternal,
        ViewMethods, ops::add_grad, ViewMutMethods,
    },
    tensor::{AutoGradTensorMethods, TensorMethods, ops::Operation}, go_backward,
};

pub(crate) struct ReshapeOp {
    shape: Vec<usize>,
    output: Weak<Mutex<NdArrayTensorInternal>>,
    input: Arc<Mutex<NdArrayTensorInternal>>,
}

struct ReshapeView {
    shape: Vec<usize>,
}

impl ReshapeView {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl AsView for ReshapeView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor.into_reshape(self.shape.as_slice()).unwrap()
    }

    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        tensor.into_reshape_mut(self.shape.as_slice()).unwrap()
    }
}

impl ReshapeOp {
    pub fn forward(input: &NdArrayTensor, shape: &[usize]) -> NdArrayTensor {
        let input_i = input.i();
        let try_reshape = input_i.as_view().clone().into_reshape(shape).is_some();
        drop(input_i);
        let mut output = if try_reshape {
            let cloned = input.clone();
            cloned
                .i()
                .view
                .push(Arc::new(ReshapeView::new(shape.to_vec())));
            
            cloned
        } else {
            // clone the input tensor as standard layout and reshape it
            let input_i = input.i();
            let input = input_i.as_view().standard_layout();
            drop(input_i);
            let reshaped = input.into_reshape(shape).unwrap();
            reshaped.into()
        };
        if input.requires_grad() {
            output.i().op = Some(Arc::new(ReshapeOp {
                shape: shape.to_vec(),
                output: output.i_ref(),
                input: input.i_copy(),
            }));
            output.set_requires_grad(true);
        }
        output
    }
}

impl Operation for ReshapeOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.name_clone());

        let input = self.input.lock().unwrap();
        let shape = input.as_view().size();
        drop(input);
        let subgrad = grad.reshape(shape.as_slice());
        go_backward!(self.input, subgrad);
    }
}