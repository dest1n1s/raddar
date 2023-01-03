use std::sync::{Arc, Mutex, Weak};

use crate::go_backward;
use crate::ndarr::ops::add_grad;
use crate::ndarr::{
    BorrowView, KindedArrayD, NdArrayTensor, NdArrayTensorInternal, ViewMethods, ViewMutMethods,
};
use crate::tensor::ops::Operation;
use crate::tensor::{AutoGradTensorMethods, ScatterReduction, TensorMethods};
pub(crate) struct ExtOp {
    dim: usize,
    keep_dim: bool,
    is_max: bool,
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output_ext: Weak<Mutex<NdArrayTensorInternal>>,
    output_argext: Arc<Mutex<NdArrayTensorInternal>>,
}

impl ExtOp {
    pub fn forward(
        input: &NdArrayTensor,
        params: (usize, bool, bool),
    ) -> (NdArrayTensor, NdArrayTensor) {
        let (dim, keep_dim, is_max) = params;
        let input_i = input.i();
        let (ext, argext) = input_i.as_view().ext_dim(dim, keep_dim, is_max);
        drop(input_i);
        let (ext, argext): (NdArrayTensor, NdArrayTensor) = (ext.into(), argext.into());

        if input.requires_grad() {
            ext.i().op = Some(Arc::new(ExtOp {
                dim,
                keep_dim,
                is_max,
                input: input.i_copy(),
                output_ext: ext.i_ref(),
                output_argext: argext.i_copy(),
            }))
        }

        (ext, argext)
    }
}

impl Operation for ExtOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output_ext.clone(), grad.name_clone());

        let input = self.input.lock().unwrap();
        let shape = input.as_view().size();
        let dtype = input.as_view().kind();
        drop(input);

        let grad = grad.i();
        let argext = self.output_argext.lock().unwrap();
        let (argext_view, grad_view) = if self.keep_dim {
            (argext.as_view().clone(), grad.as_view().clone())
        } else {
            (
                argext.as_view().clone().into_unsqueeze(self.dim),
                grad.as_view().clone().into_unsqueeze(self.dim),
            )
        };
        let mut factor = KindedArrayD::zeros(&shape, dtype);
        factor
            .view_mut()
            .scatter_dim_(self.dim, &argext_view, &grad_view, ScatterReduction::Add);
        drop(argext);
        drop(grad);

        go_backward!(self.input, factor.into());
    }
}
