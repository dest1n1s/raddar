use std::sync::{Arc, Mutex, Weak};

use higher_order_closure::hrtb;
use more_asserts::assert_ge;

use crate::{
    go_backward,
    ndarr::{
        lens::{CompositeLen, LookThrough, ViewLens},
        ops::add_grad,
        KindedArrayD, KindedArrayViewD, NdArrayTensor, NdArrayTensorInternal, ViewMethods,
        ViewsImmut,
    },
    tensor::{
        index::{IndexInfo, IndexInfoItem},
        ops::Operation,
        ArrayMethods, AutoGradTensorMethods, TensorMethods,
    },
};
pub(crate) struct CatOp {
    dim: usize,
    output: Weak<Mutex<NdArrayTensorInternal>>,
    inputs: Vec<Arc<Mutex<NdArrayTensorInternal>>>,
}

impl CatOp {
    pub fn forward(inputs: &[&NdArrayTensor], dim: usize) -> NdArrayTensor {
        assert_ge!(inputs.len(), 1, "CatOp requires at least one input");

        let mut len = ViewLens::with(inputs[0].i_copy());
        for input in inputs.iter().skip(1) {
            len = len.and(input.i_copy());
        }
        let mut result: NdArrayTensor = len
            .look_through(hrtb!(|inputs: ViewsImmut<'_, '_>| -> KindedArrayD {
                KindedArrayViewD::cat(
                    &inputs.iter().map(|handle| &**handle).collect::<Vec<_>>(),
                    dim,
                )
            }))
            .into();

        if inputs.iter().any(|x| x.requires_grad()) {
            result.i().op = Some(Arc::new(CatOp {
                dim,
                output: result.i_ref(),
                inputs: inputs.iter().map(|x| x.i_copy()).collect(),
            }));
            result.set_requires_grad(true);
        }
        result
    }
}

impl Operation for CatOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.name_clone());

        let mut offset = 0;
        let shape = grad.size();
        let mut slice = IndexInfo::from(Vec::<IndexInfoItem>::new()).rest_full_for(&shape);

        for input in self.inputs.iter() {
            let input_locked = input.lock().unwrap();
            let size = input_locked.as_view().size()[self.dim] as isize;
            drop(input_locked);
            slice.infos[self.dim] = IndexInfoItem::Range(offset, Some(offset + size), 1);

            let sub_grad = grad.slice(slice.clone());
            go_backward!(input, sub_grad);

            offset += size;
        }
    }
}
