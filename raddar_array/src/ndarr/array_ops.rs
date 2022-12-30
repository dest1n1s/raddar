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
}

pub(crate) struct SliceOp {
    slice: IndexInfo,
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output: Arc<Mutex<NdArrayTensorInternal>>,
}

impl SliceOp {
    pub(crate) fn forward(input: &NdArrayTensor, index: IndexInfo) -> NdArrayTensor {
        let cloned = input.clone();
        cloned
            .i()
            .view
            .0
            .push(Arc::new(SliceView::new(index.clone())));
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

pub(crate) struct PermuteView {
    permute: Vec<usize>,
}

impl PermuteView {
    pub fn new(permute: Vec<usize>) -> Self {
        Self { permute }
    }
}

impl AsView for PermuteView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor.into_permute(&self.permute)
    }

    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        tensor.into_permute_mut(&self.permute)
    }
}

pub(crate) struct PermuteOp {
    permute: Vec<usize>,
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output: Arc<Mutex<NdArrayTensorInternal>>,
}

impl PermuteOp {
    pub(crate) fn forward(input: &NdArrayTensor, permute: &[usize]) -> NdArrayTensor {
        let cloned = input.clone();
        cloned
            .i()
            .view
            .0
            .push(Arc::new(PermuteView::new(permute.to_vec())));
        if cloned.i().requires_grad {
            cloned.i().op = Some(Arc::new(PermuteOp {
                input: input.i_copy(),
                output: cloned.i_copy(),
                permute: permute.to_vec(),
            }));
        }
        cloned
    }
}

impl Operation for PermuteOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.clone());

        let tensor = self.input.lock().unwrap();
        let kind = tensor.as_view().kind();
        let size = tensor.as_view().size();
        drop(tensor);

        let backward_grad = NdArrayTensor::zeros(&size, kind);
        let mut sub_grad = backward_grad.permute(&self.permute);
        sub_grad += grad;

        go_backward!(self.input, backward_grad);
    }
}

pub(crate) struct TransposeOp;

impl TransposeOp {
    pub(crate) fn forward(input: &NdArrayTensor, dim0: isize, dim1: isize) -> NdArrayTensor {
        let ndim = input.size().len();

        let dim0 = if dim0 < 0 {
            (ndim as isize + dim0) as usize
        } else {
            dim0 as usize
        };
        let dim1 = if dim1 < 0 {
            (ndim as isize + dim1) as usize
        } else {
            dim1 as usize
        };
        let permuted: Vec<usize> = (0..ndim)
            .map(|i| {
                if i == dim0 {
                    dim1
                } else if i == dim1 {
                    dim0
                } else {
                    i
                }
            })
            .collect();

        PermuteOp::forward(input, &permuted)
    }
}

pub(crate) struct BroadcastView {
    broadcast: Vec<usize>,
}

impl BroadcastView {
    pub fn new(broadcast: Vec<usize>) -> Self {
        Self { broadcast }
    }
}

impl AsView for BroadcastView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor.into_broadcast(&self.broadcast)
    }

    fn view_mut<'a>(&self, _: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        panic!("Cannot broadcast into a mutable view")
    }
}

pub(crate) struct BroadcastOp {
    broadcast: Vec<usize>,
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output: Arc<Mutex<NdArrayTensorInternal>>,
}

impl BroadcastOp {
    pub(crate) fn forward(input: &NdArrayTensor, broadcast: &[usize]) -> NdArrayTensor {
        let cloned = input.clone();
        cloned
            .i()
            .view
            .0
            .push(Arc::new(BroadcastView::new(broadcast.to_vec())));
        if cloned.i().requires_grad {
            cloned.i().op = Some(Arc::new(BroadcastOp {
                input: input.i_copy(),
                output: cloned.i_copy(),
                broadcast: broadcast.to_vec(),
            }));
        }
        cloned
    }

    /// Broadcasts two tensors to the same size if possible.
    ///
    /// # Panics
    ///
    /// Panics if the two tensors cannot be broadcasted to the same size.
    pub(crate) fn cobroadcast(
        input: &NdArrayTensor,
        other: &NdArrayTensor,
    ) -> (NdArrayTensor, NdArrayTensor) {
        let mut this_size = input.size();
        let mut other_size = other.size();

        this_size.reverse();
        other_size.reverse();

        let mut broadcast = Vec::new();

        let longest_len = std::cmp::max(this_size.len(), other_size.len());
        for i in 0..longest_len {
            let this_dim = this_size.get(i).cloned().unwrap_or(1);
            let other_dim = other_size.get(i).cloned().unwrap_or(1);
            if this_dim == 1 {
                broadcast.push(other_dim);
            } else if other_dim == 1 {
                broadcast.push(this_dim);
            } else if this_dim == other_dim {
                broadcast.push(this_dim);
            } else {
                panic!("Cannot broadcast tensors of size {:?} and {:?}", input.size(), other.size());
            }
        }

        broadcast.reverse();

        let this = if broadcast == input.size() {
            input.name_clone()
        } else {
            Self::forward(input, &broadcast)
        };

        let other = if broadcast == other.size() {
            other.name_clone()
        } else {
            Self::forward(other, &broadcast)
        };

        (this, other)
    }
}

impl Operation for BroadcastOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.clone());

        let tensor = self.input.lock().unwrap();
        let kind = tensor.as_view().kind();
        let size = tensor.as_view().size();
        drop(tensor);

        let backward_grad = NdArrayTensor::zeros(&size, kind);
        // todo: you cannot use broadcasted tensor as a mutable view!
        let mut sub_grad = backward_grad.broadcast(&self.broadcast);
        sub_grad += grad;

        go_backward!(self.input, backward_grad);
    }
}

pub(crate) struct IdentityView;

impl AsView for IdentityView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor
    }

    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        tensor
    }
}

pub(crate) struct IdentityOp {
    input: Arc<Mutex<NdArrayTensorInternal>>,
    output: Arc<Mutex<NdArrayTensorInternal>>,
}

impl IdentityOp {
    pub(crate) fn forward(input: &NdArrayTensor) -> NdArrayTensor {
        let cloned = input.clone();
        cloned
            .i()
            .view
            .0
            .push(Arc::new(IdentityView));
        if cloned.i().requires_grad {
            cloned.i().op = Some(Arc::new(IdentityOp {
                input: input.i_copy(),
                output: cloned.i_copy(),
            }));
        }
        cloned
    }
}

impl Operation for IdentityOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.clone());
        go_backward!(self.input, grad);
    }
}