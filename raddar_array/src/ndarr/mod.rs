#![allow(dead_code)]
#![allow(unused_macros)]
use std::{
    borrow::Borrow,
    ops::{Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard},
};

use self::{
    array_ops::{BroadcastOp, PermuteOp, SliceOp, SqueezeView, TransposeOp, UnsqueezeView},
    ops::{
        AddOp, AddScalarOp, DivOp, DivScalarOp, GradAccumulateOp, MulOp, MulScalarOp, NegOp,
        SqueezeOp, SubOp, SubScalarOp, SumOp, UnsqueezeOp,
    },
    single_ops::matmul::MatmulOp,
};
use crate::{
    arith_impl, borrow_two_tensor_internals,
    tensor::{
        index::IndexInfo, ops::Operation, ArrayMethods, AutoGradTensorMethods, TensorMethods,
    },
};
use more_asserts::assert_gt;
use ndarray::{
    ArrayD, ArrayViewD, ArrayViewMutD, Axis, IxDyn, LinalgScalar, ScalarOperand,
    SliceInfoElem,
};
use num::{One, Zero};
use owning_ref::OwningHandle;

pub mod array_ops;
pub mod ops;
mod single_ops;

/// A tensor exported to users. It holds a reference to the actual data.
///
/// ```text
///      ┌───────────────┐
/// ┌───►│ NdArrayTensor │
/// │    │               │       ┌───────────────────────┐
/// │    │  internal─────┼──────►│ NdArrayTensorInternal │
/// │    │               │       │                       │         ┌─────────────────────────────────┐
/// │    └───────────────┘       │  data─────────────────┼────────►│ ArrayD<E>                    │
/// │                            │                       │         │                                 │
/// └────────────────────────────┼──grad (could be None) │         │  (Holding the real tensor data) │
///                              │                       │         └─────────────────────────────────┘
///                              └───────────────────────┘
/// ```
///
/// ## What happens if I call `&t1 + &t2`, where `t1` and `t2` are two tensors?
///
/// 1. `t1.add(&t2)` is called, where `add` is a method of `TensorMethods` on `NdArrayTensor`;
/// 2. `NdArrayTensor::add` calls `BroadcastOp::cobroadcast`, which returns two new `NdArrayTensor`s in the same shape (let us skip the details of broadcasting for now);
/// 3. `NdArrayTensor::add` calls `AddOp::forward` then;
/// 4. `AddOp::forward` calls `&*t1.i().as_view() + &*t2.i().as_view()`. Here `t1.i()` and `t2.i()` are `NdArrayTensorInternal`s, and `as_view()` returns a view of the tensor. The view is a `KindedArrayViewD`, which is a wrapper of `ArrayViewD`.
///    We need such an abstraction because we need to support different views, such as `TransposeView`, `SliceView`, etc. You should note that **t1.i().data does not contain the information of how we should view the tensor,** the transformation is
///    done by `as_view()`, which applies the transformations in `t1.i().view` to the tensor and obtain a correct view;
/// 5. `t1_view.add(&t2_view)` is called, where `add` is a method of `ViewMethods` on `KindedArrayViewD`;
/// 6. `KindedArrayViewD::add` extracts the underlying `ArrayViewD` from `t1_view` and `t2_view`, and calls `add` on `ArrayViewD` (**the actual addition is done here**) to get the result of the addition, an `ArrayD`. It then wraps the result into a `ArrayD<E>` and returns;
/// 7. `AddOp::forward` now has the result of the addition in a `ArrayD<E>`, which is a wrapper of `ArrayD`. It then wraps the result and an AddOp into a `NdArrayTensorInternal`, and returns a `NdArrayTensor` with this internal;
/// 8. The `NdArrayTensor` is returned to the user from `NdArrayTensor::add`.
pub struct NdArrayTensor<E: Element> {
    internal: Option<Arc<Mutex<NdArrayTensorInternal<E>>>>,
}

/// ViewType is used to indicate whether the tensor is a view of some tensor.
#[derive(Clone)]
pub(crate) struct ViewType<E: Element>(Vec<Arc<dyn AsView<E>>>);

impl<E: Element> Deref for ViewType<E> {
    type Target = Vec<Arc<dyn AsView<E>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: Element> DerefMut for ViewType<E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// AsView is used to get a view of a tensor.
pub(crate) trait AsView<E: Element> {
    fn view<'a>(&self, tensor: ArrayViewD<'a, E>) -> ArrayViewD<'a, E>;
    fn view_mut<'a>(&self, tensor: ArrayViewMutD<'a, E>) -> ArrayViewMutD<'a, E>;
}

/// A type that impls `BorrowView` can be borrowed as a view of some tensor.
pub(crate) trait BorrowView<E: Element> {
    fn view(&self) -> ArrayViewD<'_, E>;
    fn view_mut(&mut self) -> ArrayViewMutD<'_, E>;
}

pub(crate) struct NdArrayTensorInternal<E: Element> {
    /// The view type of this tensor.
    view: ViewType<E>,
    /// The reference to the actual data.
    data: Arc<RwLock<ArrayD<E>>>,
    is_leaf: bool,
    requires_grad: bool,
    grad: Option<ArrayD<E>>,
    /// The operation that generates this tensor.
    op: Option<Arc<dyn Operation<E>>>,
}

impl<E: Element> Clone for NdArrayTensorInternal<E> {
    /// Clone the internal tensor.
    ///
    /// Note: it is a cheap clone, since it only clones the reference to the data.
    /// To prevent any expensive operation, the `grad` field is not cloned.
    fn clone(&self) -> Self {
        Self {
            view: self.view.clone(),
            data: self.data.clone(),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            grad: None,
            op: self.op.clone(),
        }
    }
}

pub(crate) trait IntoOp<E: Element> {
    fn op(self) -> Option<Arc<dyn Operation<E>>>;
}

impl<'a, E: Element> IntoOp<E> for MutexGuard<'a, NdArrayTensorInternal<E>> {
    /// Unlock the mutex guard and return the operation.
    fn op(self) -> Option<Arc<dyn Operation<E>>> {
        self.op.clone()
    }
}

// Since we have protected the data with a mutex, we can safely implement Send and Sync.
unsafe impl<E: Element> Send for NdArrayTensorInternal<E> {}
unsafe impl<E: Element> Sync for NdArrayTensorInternal<E> {}

pub trait ElementAdd =
    Sized + Add + Add<ArrayD<Self>> + for<'a> Add<&'a ArrayD<Self>> where for<'a> &'a Self: Add;
pub trait ElementSub =
    Sized + Sub<Output = Self> + Sub<ArrayD<Self>> + for<'a> Sub<&'a ArrayD<Self>>
    where for<'a> &'a Self: Sub;
pub trait ElementMul =
    Sized + Mul + Mul<ArrayD<Self>> + for<'a> Mul<&'a ArrayD<Self>> where for<'a> &'a Self: Mul;
pub trait ElementDiv =
    Sized + Div<Output = Self> + Div<ArrayD<Self>> + for<'a> Div<&'a ArrayD<Self>>
    where for<'a> &'a Self: Div;
pub trait ElementNeg = Sized + Neg<Output = Self> where for<'a> &'a Self: Neg<Output = Self>;

pub trait Element:
    'static
    + Clone
    + Copy
    + std::fmt::Debug
    + ElementAdd
    + ElementSub
    + ElementMul
    + ElementDiv
    + ElementNeg
    + Zero
    + One
    + Send
    + Sync
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + ScalarOperand
    + LinalgScalar
{
}

impl Element for f32 {}
impl Element for f64 {}
impl Element for i16 {}
impl Element for i32 {}
impl Element for i64 {}

impl<E: Element> NdArrayTensor<E> {
    fn none() -> Self {
        Self { internal: None }
    }

    pub(crate) fn debug_print(&self) {
        match self.internal {
            Some(ref kinded_array) => {
                let lock = kinded_array.lock().unwrap();
                println!("{:?}", *lock.as_view());
            }
            None => println!("None"),
        }
    }

    /// Get the internal data of the tensor.
    fn i(&self) -> MutexGuard<'_, NdArrayTensorInternal<E>> {
        self.internal.as_ref().unwrap().lock().unwrap()
    }

    /// Get a copy of the reference of the internal data of the tensor.
    ///
    /// Note: this is a shallow copy, so the data is not copied.
    fn i_copy(&self) -> Arc<Mutex<NdArrayTensorInternal<E>>> {
        self.internal.as_ref().unwrap().clone()
    }

    /// Do a very shallow copy of the tensor.
    ///
    /// Only the reference of the internal data is copied, not the data itself.
    ///
    /// Compared to `clone`, this function does not really clone anything, so it is much faster.
    ///
    /// It is usually useful to get a tensor from its reference. You can view the returned tensor as the exact same tensor as `self`.
    ///
    /// Note: they share the same internal data and lock, do not lock the same tensor twice!
    pub(crate) fn name_clone(&self) -> Self {
        Self {
            internal: self.internal.clone(),
        }
    }
}

impl<E: Element> Clone for NdArrayTensor<E> {
    /// Shallow copy the tensor as a new tensor.
    ///
    /// Note: this is a shallow copy, so only the reference of the data is copied, not the data itself. The gradient is not copied, either.
    fn clone(&self) -> Self {
        match self.internal {
            Some(ref tensor_internal) => {
                let tensor_internal = tensor_internal.lock().unwrap();
                Self {
                    internal: Some(Arc::new(Mutex::new(tensor_internal.clone()))),
                }
            }
            None => Self::none(),
        }
    }
}

impl<E: Element> NdArrayTensorInternal<E> {
    pub(crate) fn as_view<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, ArrayD<E>>, Box<ArrayViewD<'a, E>>> {
        let data = self.data.read().unwrap();
        OwningHandle::new_with_fn(data, |data| {
            let tensor = unsafe { &*data };
            let mut view = tensor.view();
            for viewer in self.view.iter() {
                view = viewer.view(view);
            }
            Box::new(view)
        })
    }

    pub(crate) fn as_view_mut<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, ArrayD<E>>, Box<ArrayViewMutD<'a, E>>> {
        let data = self.data.read().unwrap();
        OwningHandle::new_with_fn(data, |data| {
            let tensor = unsafe { &mut *(data as *mut ArrayD<E>) };
            let mut view = tensor.view_mut();
            for viewer in self.view.iter() {
                view = viewer.view_mut(view);
            }
            Box::new(view)
        })
    }
}

// =================================================================================================
// Implementations for `NdArrayTensor` and `ArrayD<E>`.
// =================================================================================================

impl<E: Element> TensorMethods<E> for NdArrayTensor<E> {
    fn zeros(shape: &[usize]) -> Self {
        NdArrayTensor::from(ArrayD::<E>::zeros(shape))
    }

    fn ones(shape: &[usize]) -> Self {
        NdArrayTensor::from(ArrayD::<E>::ones(shape))
    }

    fn size(&self) -> Vec<usize> {
        self.i().as_view().shape().to_vec()
    }

    fn item(&self) -> E {
        self.i().as_view().item()
    }

    fn t(&self) -> Self {
        TransposeOp::forward(self, -1, -2)
    }

    fn neg(&self) -> Self {
        NegOp::forward(self)
    }

    fn add(&self, other: &Self) -> Self {
        let (self_, other_) = BroadcastOp::cobroadcast(self, other);
        AddOp::forward((&self_, &other_))
    }

    fn sub(&self, other: &Self) -> Self {
        let (self_, other_) = BroadcastOp::cobroadcast(self, other);
        SubOp::forward((&self_, &other_))
    }

    fn mul(&self, other: &Self) -> Self {
        let (self_, other_) = BroadcastOp::cobroadcast(self, other);
        MulOp::forward((&self_, &other_))
    }

    fn div(&self, other: &Self) -> Self {
        let (self_, other_) = BroadcastOp::cobroadcast(self, other);
        DivOp::forward((&self_, &other_))
    }

    fn add_(&mut self, other: &Self) {
        borrow_two_tensor_internals!(
            self.internal.as_mut().unwrap(),
            other.internal.as_ref().unwrap(),
            inputs,
            {
                *inputs.0.as_view_mut() += &*inputs.1.as_view();
            }
        )
    }

    fn sub_(&mut self, other: &Self) {
        borrow_two_tensor_internals!(
            self.internal.as_mut().unwrap(),
            other.internal.as_ref().unwrap(),
            inputs,
            {
                *inputs.0.as_view_mut() -= &*inputs.1.as_view();
            }
        )
    }

    fn mul_(&mut self, other: &Self) {
        borrow_two_tensor_internals!(
            self.internal.as_mut().unwrap(),
            other.internal.as_ref().unwrap(),
            inputs,
            {
                *inputs.0.as_view_mut() *= &*inputs.1.as_view();
            }
        )
    }

    fn div_(&mut self, other: &Self) {
        borrow_two_tensor_internals!(
            self.internal.as_mut().unwrap(),
            other.internal.as_ref().unwrap(),
            inputs,
            {
                *inputs.0.as_view_mut() /= &*inputs.1.as_view();
            }
        )
    }

    fn add_scalar(&self, other: E) -> Self {
        AddScalarOp::forward(self, other)
    }

    fn sub_scalar(&self, other: E) -> Self {
        SubScalarOp::forward(self, other)
    }

    fn mul_scalar(&self, other: E) -> Self {
        MulScalarOp::forward(self, other)
    }

    fn div_scalar(&self, other: E) -> Self {
        DivScalarOp::forward(self, other)
    }

    fn add_scalar_(&mut self, other: E) {
        *self.i().as_view_mut() += other;
    }

    fn sub_scalar_(&mut self, other: E) {
        *self.i().as_view_mut() -= other;
    }

    fn mul_scalar_(&mut self, other: E) {
        *self.i().as_view_mut() *= other;
    }

    fn div_scalar_(&mut self, other: E) {
        *self.i().as_view_mut() /= other;
    }

    fn matmul(&self, other: &Self) -> Self {
        MatmulOp::forward((self, other))
    }

    fn assign(&mut self, other: &Self) {
        borrow_two_tensor_internals!(
            self.internal.as_mut().unwrap(),
            other.internal.as_ref().unwrap(),
            inputs,
            {
                inputs.0.as_view_mut().assign(&*inputs.1.as_view());
            }
        )
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self {
        SumOp::forward(self, (dim.to_vec(), keep_dim))
    }

    fn unsqueeze(&self, dim: usize) -> Self {
        UnsqueezeOp::forward(self, dim)
    }

    fn unsqueeze_(&mut self, dim: usize) {
        self.i().view.push(Arc::new(UnsqueezeView::new(dim)));
    }

    fn squeeze(&self, dim: usize) -> Self {
        SqueezeOp::forward(self, dim)
    }

    fn squeeze_(&mut self, dim: usize) {
        self.i().view.push(Arc::new(SqueezeView::new(dim)));
    }
}

impl<E: Element> ArrayMethods<E> for NdArrayTensor<E> {
    fn slice(&self, index: IndexInfo) -> Self {
        SliceOp::forward(self, index)
    }

    fn permute(&self, permute: &[usize]) -> Self {
        PermuteOp::forward(self, permute)
    }

    fn broadcast(&self, shape: &[usize]) -> Self {
        BroadcastOp::forward(self, shape)
    }
}

impl<E: Element> BorrowView<E> for ArrayD<E> {
    fn view(&self) -> ArrayViewD<'_, E> {
        ArrayD::view(self)
    }

    fn view_mut(&mut self) -> ArrayViewMutD<'_, E> {
        ArrayD::view_mut(self)
    }
}

impl<E: Element> TensorMethods<E> for ArrayD<E> {
    fn zeros(shape: &[usize]) -> Self {
        ArrayD::zeros(IxDyn(&shape))
    }

    fn ones(shape: &[usize]) -> Self {
        ArrayD::ones(IxDyn(&shape))
    }

    fn size(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn item(&self) -> E {
        self.view().item()
    }

    fn t(&self) -> Self {
        ArrayD::<E>::from(self.t().into_owned())
    }

    fn neg(&self) -> Self {
        -&self.view()
    }

    fn add(&self, other: &Self) -> Self {
        &self.view() + &other.view()
    }

    fn sub(&self, other: &Self) -> Self {
        &self.view() - &other.view()
    }

    fn mul(&self, other: &Self) -> Self {
        &self.view() * &other.view()
    }

    fn div(&self, other: &Self) -> Self {
        &self.view() / &other.view()
    }

    fn add_(&mut self, other: &Self) {
        let mut self_view = self.view_mut();
        self_view += &other.view();
    }

    fn sub_(&mut self, other: &Self) {
        let mut self_view = self.view_mut();
        self_view -= &other.view();
    }

    fn mul_(&mut self, other: &Self) {
        let mut self_view = self.view_mut();
        self_view *= &other.view();
    }

    fn div_(&mut self, other: &Self) {
        let mut self_view = self.view_mut();
        self_view /= &other.view();
    }

    fn add_scalar(&self, other: E) -> Self {
        &self.view() + other
    }

    fn sub_scalar(&self, other: E) -> Self {
        &self.view() - other
    }

    fn mul_scalar(&self, other: E) -> Self {
        &self.view() * other
    }

    fn div_scalar(&self, other: E) -> Self {
        &self.view() / other
    }

    fn add_scalar_(&mut self, other: E) {
        let mut self_view = self.view_mut();
        self_view += other;
    }

    fn sub_scalar_(&mut self, other: E) {
        let mut self_view = self.view_mut();
        self_view -= other;
    }

    fn mul_scalar_(&mut self, other: E) {
        let mut self_view = self.view_mut();
        self_view *= other;
    }

    fn div_scalar_(&mut self, other: E) {
        let mut self_view = self.view_mut();
        self_view /= other;
    }

    fn matmul(&self, other: &Self) -> Self {
        self.view().matmul(other.view())
    }

    fn assign(&mut self, other: &Self) {
        self.view_mut().assign(&other.view());
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self {
        self.view().sum_dim(dim, keep_dim)
    }

    fn unsqueeze(&self, dim: usize) -> Self {
        self.view().into_unsqueeze(dim).upgrade()
    }

    fn squeeze(&self, dim: usize) -> Self {
        self.view().into_squeeze(dim).upgrade()
    }

    fn unsqueeze_(&mut self, _: usize) {
        unimplemented!()
    }

    fn squeeze_(&mut self, _: usize) {
        unimplemented!()
    }
}

// =================================================================================================
// Autograd methods
// =================================================================================================
impl<E: Element> AutoGradTensorMethods<E> for NdArrayTensor<E> {
    fn backward(&mut self) {
        if let Some(op) = self.i().op() {
            let size = self.size();
            op.backward(NdArrayTensor::ones(&size));
        }
    }

    fn grad(&self) -> Self {
        // perf: avoid clone
        NdArrayTensor::from(self.i().grad.as_ref().unwrap().clone())
    }

    fn zero_grad(&mut self) {
        self.i().grad = None;
    }

    fn requires_grad(&self) -> bool {
        self.i().requires_grad
    }

    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.i().requires_grad = requires_grad;
        if requires_grad && self.i().op.is_none() {
            self.i().op = Some(Arc::new(GradAccumulateOp::new(self.i_copy())));
        } else if !requires_grad {
            self.i().op = None;
        }
    }
}

arith_impl!(NdArrayTensor<E>, E);

// =================================================================================================
// Implementations of quick ways to create `NdArrayTensor`, `ArrayD<E>`, `KindedArrayViewD` and `KindedArrayViewMutD`.
// =================================================================================================
impl<E: Element> From<ArrayD<E>> for NdArrayTensor<E> {
    fn from(array: ArrayD<E>) -> Self {
        Self::from(Arc::new(RwLock::new(array)))
    }
}

impl<E: Element> From<Arc<RwLock<ArrayD<E>>>> for NdArrayTensor<E> {
    fn from(array: Arc<RwLock<ArrayD<E>>>) -> Self {
        NdArrayTensor {
            internal: Some(Arc::new(Mutex::new(NdArrayTensorInternal {
                data: array,
                op: None,
                grad: None,
                requires_grad: false,
                is_leaf: true,
                view: ViewType(vec![]),
            }))),
        }
    }
}

// =================================================================================================
// View and ViewMut's Methods and Implementations
//
// Below are the bottom implementations of the tensor arithmetic methods.
// Upper structs' arithmetic methods should invoke these as needed.
// =================================================================================================
pub(crate) trait SuperViewMethods<E: Element> {
    type OwnedType;
    type ViewType<'a>;
    type ViewMutType<'a>;
}

pub(crate) trait ViewMutMethods<'this, E: Element>: SuperViewMethods<E> + 'this {
    fn upgrade(&self) -> Self::OwnedType;

    fn slice_mut<'a>(&'a mut self, info: IndexInfo) -> Self::ViewMutType<'a>;
    fn into_slice_mut(self, info: IndexInfo) -> Self::ViewMutType<'this>;
    fn permute_mut<'a>(&'a mut self, axes: &[usize]) -> Self::ViewMutType<'a>;
    fn into_permute_mut(self, axes: &[usize]) -> Self::ViewMutType<'this>;

    fn add_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn sub_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn mul_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn div_(&mut self, other: impl Borrow<Self::ViewType<'_>>);

    fn add_scalar_(&mut self, other: E);
    fn sub_scalar_(&mut self, other: E);
    fn mul_scalar_(&mut self, other: E);
    fn div_scalar_(&mut self, other: E);

    fn assign(&mut self, other: impl Borrow<Self::ViewType<'_>>);

    fn into_unsqueeze_mut(self, axis: usize) -> Self::ViewMutType<'this>;
    fn into_squeeze_mut(self, axis: usize) -> Self::ViewMutType<'this>;

    fn downgrade<'a>(&'a self) -> Self::ViewType<'a>;
}

pub(crate) trait ViewMethods<'this, E: Element>: SuperViewMethods<E> + 'this {
    fn size(&self) -> Vec<usize>;
    fn item(&self) -> E;
    fn upgrade(&self) -> Self::OwnedType;

    fn slice<'a>(&'a self, info: IndexInfo) -> Self::ViewType<'a>;
    fn into_slice(self, info: IndexInfo) -> Self::ViewType<'this>;
    fn permute<'a>(&'a self, order: &[usize]) -> Self::ViewType<'a>;
    fn into_permute(self, order: &[usize]) -> Self::ViewType<'this>;
    fn broadcast<'a>(&'a self, shape: &[usize]) -> Self::ViewType<'a>;
    fn into_broadcast(self, shape: &[usize]) -> Self::ViewType<'this>;
    fn t<'a>(&'a self) -> Self::ViewType<'a>;

    fn neg(&self) -> Self::OwnedType;
    fn matmul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn add(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn sub(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn mul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn div(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;

    fn add_scalar(&self, other: E) -> Self::OwnedType;
    fn sub_scalar(&self, other: E) -> Self::OwnedType;
    fn mul_scalar(&self, other: E) -> Self::OwnedType;
    fn div_scalar(&self, other: E) -> Self::OwnedType;

    fn sum_one_dim(&self, dim: usize, keep_dim: bool) -> ArrayD<E>;
    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self::OwnedType;
    fn into_unsqueeze(self, dim: usize) -> Self::ViewType<'this>;
    fn into_squeeze(self, dim: usize) -> Self::ViewType<'this>;
}

impl<E: Element> SuperViewMethods<E> for ArrayViewMutD<'_, E> {
    type OwnedType = ArrayD<E>;

    type ViewType<'a> = ArrayViewD<'a, E>;

    type ViewMutType<'a> = ArrayViewMutD<'a, E>;
}

impl<'this, E: Element> ViewMutMethods<'this, E> for ArrayViewMutD<'this, E> {
    fn upgrade(&self) -> Self::OwnedType {
        ArrayD::from(self.to_owned())
    }

    fn slice_mut<'a>(&'a mut self, info: IndexInfo) -> Self::ViewMutType<'a> {
        let info: Vec<SliceInfoElem> = info.into();
        ArrayViewMutD::from(self.slice_mut(info.as_slice()))
    }

    fn into_slice_mut(self, info: IndexInfo) -> Self::ViewMutType<'this> {
        let info: Vec<SliceInfoElem> = info.into();
        self.slice_move(info.as_slice())
    }

    fn permute_mut<'a>(&'a mut self, axes: &[usize]) -> Self::ViewMutType<'a> {
        todo!()
    }

    fn into_permute_mut(self, axes: &[usize]) -> Self::ViewMutType<'this> {
        self.permuted_axes(axes)
    }

    fn add_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        *self += other.borrow();
    }

    fn sub_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        *self -= other.borrow();
    }

    fn mul_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        *self *= other.borrow();
    }

    fn div_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        *self /= other.borrow();
    }

    fn add_scalar_(&mut self, other: E) {
        *self += other;
    }

    fn sub_scalar_(&mut self, other: E) {
        *self -= other;
    }

    fn mul_scalar_(&mut self, other: E) {
        *self *= other;
    }

    fn div_scalar_(&mut self, other: E) {
        *self /= other;
    }

    fn assign(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        self.assign(other.borrow());
    }

    fn into_unsqueeze_mut(self, axis: usize) -> Self::ViewMutType<'this> {
        self.insert_axis(Axis(axis))
    }

    fn into_squeeze_mut(self, axis: usize) -> Self::ViewMutType<'this> {
        self.remove_axis(Axis(axis))
    }

    fn downgrade(&self) -> Self::ViewType<'_> {
        self.view()
    }
}

impl<E: Element> SuperViewMethods<E> for ArrayViewD<'_, E> {
    type OwnedType = ArrayD<E>;

    type ViewType<'a> = ArrayViewD<'a, E>;

    type ViewMutType<'a> = ArrayViewMutD<'a, E>;
}

impl<'this, E: Element> ViewMethods<'this, E> for ArrayViewD<'this, E> {
    fn upgrade(&self) -> Self::OwnedType {
        ArrayD::from(self.to_owned())
    }

    fn neg(&self) -> Self::OwnedType {
        ArrayD::from(self.mapv(|x| -x))
    }

    fn item(&self) -> E {
        assert_eq!(self.len(), 1);
        *self.first().unwrap()
    }

    fn slice<'a>(&'a self, info: IndexInfo) -> Self::ViewType<'a> {
        let info: Vec<SliceInfoElem> = info.into();
        ArrayViewD::from(self.slice(info.as_slice()))
    }

    fn into_slice(self, info: IndexInfo) -> Self::ViewType<'this> {
        let info: Vec<SliceInfoElem> = info.into();
        self.slice_move(info.as_slice())
    }

    fn permute<'a>(&'a self, order: &[usize]) -> Self::ViewType<'a> {
        todo!()
    }

    fn into_permute(self, order: &[usize]) -> Self::ViewType<'this> {
        self.permuted_axes(order)
    }

    fn broadcast<'a>(&'a self, shape: &[usize]) -> Self::ViewType<'a> {
        self.broadcast(shape).unwrap()
    }

    fn into_broadcast(self, shape: &[usize]) -> Self::ViewType<'this> {
        // TODO: This is a hack to get around the borrow checker,
        // fooling it that we are not borrowing self.
        let array = &self as *const ArrayViewD<'_, E>;
        let array = unsafe { &*array };
        array.broadcast(shape).unwrap()
    }

    fn t<'a>(&'a self) -> Self::ViewType<'a> {
        self.t()
    }

    fn add(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        ArrayD::from(self + other.borrow())
    }

    fn sub(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        ArrayD::from(self - other.borrow())
    }

    fn mul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        ArrayD::from(self * other.borrow())
    }

    fn div(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        ArrayD::from(self / other.borrow())
    }

    fn matmul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        let other: &Self::ViewType<'_> = other.borrow();
        single_ops::matmul::matmul(self.view(), other.view())
    }

    fn add_scalar(&self, other: E) -> Self::OwnedType {
        ArrayD::from(self.mapv(|x| x + other))
    }

    fn sub_scalar(&self, other: E) -> Self::OwnedType {
        ArrayD::from(self.mapv(|x| x - other))
    }

    fn mul_scalar(&self, other: E) -> Self::OwnedType {
        ArrayD::from(self.mapv(|x| x * other))
    }

    fn div_scalar(&self, other: E) -> Self::OwnedType {
        ArrayD::from(self.mapv(|x| x / other))
    }

    fn size(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn sum_one_dim(&self, dim: usize, keep_dim: bool) -> ArrayD<E> {
        let mut array = self.sum_axis(Axis(dim));
        if keep_dim {
            array.insert_axis_inplace(Axis(dim));
        }
        ArrayD::from(array)
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self::OwnedType {
        assert_gt!(dim.len(), 0, "dim must not be empty");
        // sum from the first dimension to the last dimension

        // order the dimensions in ascending order
        let mut dim = dim.to_vec();
        dim.sort_unstable();
        let mut array = self.sum_one_dim(dim[0], keep_dim);
        let mut removed_ndim = 1;
        for i in 1..dim.len() {
            array = array.view().sum_one_dim(
                if keep_dim {
                    dim[i]
                } else {
                    dim[i] - removed_ndim
                },
                keep_dim,
            );
            removed_ndim += 1;
        }
        array
    }

    fn into_unsqueeze(self, dim: usize) -> Self::ViewType<'this> {
        self.insert_axis(Axis(dim))
    }

    fn into_squeeze(self, dim: usize) -> Self::ViewType<'this> {
        self.remove_axis(Axis(dim))
    }
}
