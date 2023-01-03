use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

use crate::{
    go_backward,
    ndarr::{ops::add_grad, KindedArrayViewD, NdArrayTensor, NdArrayTensorInternal, ViewMethods},
    tensor::{
        index::{IndexInfo, IndexInfoItem},
        ops::Operation,
        AutoGradTensorMethods, TensorMethods, ArrayMethods,
    },
};
pub(crate) struct CatOp {
    dim: usize,
    output: Weak<Mutex<NdArrayTensorInternal>>,
    inputs: Vec<Arc<Mutex<NdArrayTensorInternal>>>,
}

/// Borrow multiple mutex guards from a slice of objects, allowing the same object to appear multiple times.
///
/// This function is used to borrow tensor internals from a slice of tensors.
/// If the same tensor appears multiple times in the slice,
/// the same lock guard is used for all of them to avoid deadlocks.
///
/// todo: replace the `borrow_three_tensor_internals` macro with this function and remove those macros.
fn borrow_arc_mutexes<T, U>(inputs: &[&Arc<Mutex<U>>], f: impl Fn(&[&MutexGuard<U>]) -> T) -> T {
    let mut guards = HashMap::with_capacity(inputs.len());
    let mut guard_refs = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let duplicated = inputs
            .iter()
            .enumerate()
            .take(i)
            .find(|(_, x)| Arc::ptr_eq(x, input));
        if let Some((j, _)) = duplicated {
            guard_refs.push(j);
        } else {
            guards.insert(i, input.lock().expect("The mutex is poisoned"));
            guard_refs.push(i);
        }
    }
    let guard_refs = guard_refs
        .into_iter()
        .map(|i| &guards[&i])
        .collect::<Vec<_>>();
    f(guard_refs.as_slice())
}

impl CatOp {
    pub fn forward(inputs: &[&NdArrayTensor], dim: usize) -> NdArrayTensor {
        let inputs_i = inputs
            .iter()
            .map(|x| x.internal.as_ref().unwrap())
            .collect::<Vec<_>>();
        let mut result: NdArrayTensor = borrow_arc_mutexes(inputs_i.as_slice(), |inputs| {
            let inputs = inputs
                .into_iter()
                .map(|x| x.as_view().clone())
                .collect::<Vec<_>>();
            KindedArrayViewD::cat(&inputs, dim).into()
        });
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
