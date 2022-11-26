use std::sync::{Arc, Mutex};

use ndarray::SliceInfoElem;

use crate::{
    go_backward,
    tensor::{
        index::{IndexInfo, IndexInfoItem},
        ops::Operation,
        ArrayMethods, TensorMethods,
    },
};

use super::{
    ops::add_grad, AsView, KindedArrayViewD, KindedArrayViewMutD, NdArrayTensor,
    NdArrayTensorInternal, ViewMethods, ViewMutMethods,
};

impl From<IndexInfoItem> for SliceInfoElem {
    fn from(item: IndexInfoItem) -> Self {
        match item {
            IndexInfoItem::Single(i) => SliceInfoElem::Index(i),
            IndexInfoItem::Range(start, end, step) => SliceInfoElem::Slice {
                start: start,
                end: Some(end),
                step: step,
            },
            IndexInfoItem::NewAxis => SliceInfoElem::NewAxis,
        }
    }
}

impl From<IndexInfo> for Vec<SliceInfoElem> {
    fn from(info: IndexInfo) -> Self {
        info.infos.into_iter().map(|item| item.into()).collect()
    }
}

pub(crate) struct SliceView {
    slice: IndexInfo,
}

impl SliceView {
    pub fn new(slice: IndexInfo) -> Self {
        Self { slice }
    }
}

impl AsView for SliceView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor.into_slice(self.slice.clone())
    }

    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        tensor.into_slice_mut(self.slice.clone())
    }

    fn op(&self, input: &NdArrayTensor, output: &NdArrayTensor) -> Arc<dyn Operation> {
        Arc::new(SliceOp {
            input: input.i_copy(),
            output: output.i_copy(),
            slice: self.slice.clone(),
        })
    }
}

pub(crate) struct SliceOp {
    slice: IndexInfo,
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output: Arc<Mutex<NdArrayTensorInternal>>,
}

impl SliceOp {
    pub(crate) fn forward(input: &NdArrayTensor, index: IndexInfo) -> NdArrayTensor {
        let cloned = input.clone();
        cloned.i().view = input.i().view.clone();
        cloned
            .i()
            .view
            .0
            .push(Arc::new(SliceView::new(index.clone())));
        cloned.i().is_leaf = input.i().is_leaf;
        cloned.i().requires_grad = input.i().requires_grad;
        cloned.i().grad = None;
        if cloned.i().requires_grad {
            cloned.i().op = Some(Arc::new(SliceOp {
                input: input.i_copy(),
                output: cloned.i_copy(),
                slice: index,
            }));
        }
        cloned
    }
}
impl Operation for SliceOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.clone());

        let tensor = self.input.lock().unwrap();
        let kind = tensor.as_view().kind();
        let size = tensor.as_view().size();
        drop(tensor);

        let backward_grad = NdArrayTensor::zeros(&size, kind);
        let mut sub_grad = backward_grad.slice(self.slice.clone());
        sub_grad += grad;

        go_backward!(self.input, backward_grad);
    }
}
