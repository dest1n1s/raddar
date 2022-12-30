use std::{
    borrow::Borrow,
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard},
};

use self::{
    array_ops::{BroadcastOp, PermuteOp, SliceOp, TransposeOp},
    ops::{
        AddOp, AddScalarOp, DivOp, DivScalarOp, GradAccumulateOp, MulOp, MulScalarOp, NegOp, SubOp,
        SubScalarOp, SumOp,
    },
};
use crate::{
    arith_impl,
    tensor::{
        index::IndexInfo, ops::Operation, ArrayMethods, AutoGradTensorMethods, TensorKind,
        TensorMethods,
    },
};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Axis, IxDyn, SliceInfoElem};
use num::{cast, NumCast};
use owning_ref::OwningHandle;

pub mod array_ops;
pub mod ops;

/// A tensor exported to users. It holds a reference to the actual data.
pub struct NdArrayTensor {
    internal: Option<Arc<Mutex<NdArrayTensorInternal>>>,
}

/// ViewType is used to indicate whether the tensor is a view of some tensor.
#[derive(Clone)]
pub(crate) struct ViewType(Vec<Arc<dyn AsView>>);

/// AsView is used to get a view of a tensor.
pub(crate) trait AsView {
    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a>;
    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a>;
}

/// A type that impls `BorrowView` can be borrowed as a view of some tensor.
pub(crate) trait BorrowView {
    fn view(&self) -> KindedArrayViewD<'_>;
    fn view_mut(&mut self) -> KindedArrayViewMutD<'_>;
}

pub(crate) struct NdArrayTensorInternal {
    /// The view type of this tensor.
    view: ViewType,
    /// The reference to the actual data.
    data: Arc<RwLock<KindedArrayD>>,
    is_leaf: bool,
    requires_grad: bool,
    grad: Option<KindedArrayD>,
    /// The operation that generates this tensor.
    op: Option<Arc<dyn Operation>>,
}

impl Clone for NdArrayTensorInternal {
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

pub(crate) trait IntoOp {
    fn op(self) -> Option<Arc<dyn Operation>>;
}

impl<'a> IntoOp for MutexGuard<'a, NdArrayTensorInternal> {
    /// Unlock the mutex guard and return the operation.
    fn op(self) -> Option<Arc<dyn Operation>> {
        self.op.clone()
    }
}

// Since we have protected the data with a mutex, we can safely implement Send and Sync.
unsafe impl Send for NdArrayTensorInternal {}
unsafe impl Sync for NdArrayTensorInternal {}

/// Declare some common methods for KindedArrayD, KindedArrayViewD and KindedArrayViewMutD.
macro_rules! declare_kinded_array_variant {
    ($array_type: ident, $enum: ident, $kind_fn: ident, $new_kinded_array: ident,
        $obtain_kind_array: ident, $obtain_2_kind_arrays: ident) => {

            /// Get the kind of the array.
            fn $kind_fn(kinded_array: &$enum) -> TensorKind {
                match kinded_array {
                    $enum::F32(_) => TensorKind::F32,
                    $enum::F64(_) => TensorKind::F64,
                    $enum::I16(_) => TensorKind::I16,
                    $enum::I32(_) => TensorKind::I32,
                    $enum::I64(_) => TensorKind::I64,
                }
            }

            /// Create a new kinded array.
            macro_rules! $new_kinded_array {
                ($data:expr, $kind:expr) => {
                    match $kind {
                        TensorKind::F32 => $enum::F32($data),
                        TensorKind::F64 => $enum::F64($data),
                        TensorKind::I16 => $enum::I16($data),
                        TensorKind::I32 => $enum::I32($data),
                        TensorKind::I64 => $enum::I64($data),
                        _ => unimplemented!(),
                    }
                };
            }

            /// Get the real array `$array_name` from our array type `$kind_array`, and
            /// set the type `$kind_type_name` to the type of the array.
            /// Then run the code in `$execution`.
            macro_rules! $obtain_kind_array {
                ($kind_array:expr, $array_name:ident, $kind_type_name:ident, $execution:block) => {
                    match $kind_array {
                        $enum::F32($array_name) => {
                            type $kind_type_name = f32;
                            $execution
                        }
                        $enum::F64($array_name) => {
                            type $kind_type_name = f64;
                            $execution
                        }
                        $enum::I16($array_name) => {
                            type $kind_type_name = i16;
                            $execution
                        }
                        $enum::I32($array_name) => {
                            type $kind_type_name = i32;
                            $execution
                        }
                        $enum::I64($array_name) => {
                            type $kind_type_name = i64;
                            $execution
                        }
                    }
                };
                ($kind_array:expr, $array_name:ident, $execution:block) => {
                    $obtain_kind_array!($kind_array, $array_name, KindType, $execution)
                };
            }

            /// Get the real array `$array_name1` and `$array_name2` from two elements of our array type,
            /// and set the type `OriginalKindType` to the type of the `$array_name1`,
            /// and then cast the type of the `$array_name2` to `OriginalKindType` if necessary.
            /// Then run the code in `$execution`.
            /// Note: the `$array_name2` will be either of the same type as `$array_name1`, or an `ArrayD` if it has been casted.
            macro_rules! $obtain_2_kind_arrays {
                ($kind_array1:expr, $array_name1:ident, $kind_array2:expr, $array_name2:ident, $execution:block) => {
                    match ($kind_array1, $kind_array2) {
                        ($enum::F32($array_name1), $enum::F32($array_name2)) => {
                            type OriginalKindType = f32;
                            $execution
                        }
                        ($enum::F64($array_name1), $enum::F64($array_name2)) => {
                            type OriginalKindType = f64;
                            $execution
                        }
                        ($enum::I16($array_name1), $enum::I16($array_name2)) => {
                            type OriginalKindType = i16;
                            $execution
                        }
                        ($enum::I32($array_name1), $enum::I32($array_name2)) => {
                            type OriginalKindType = i32;
                            $execution
                        }
                        ($enum::I64($array_name1), $enum::I64($array_name2)) => {
                            type OriginalKindType = i64;
                            $execution
                        }
                        (_tmp_array_1, _tmp_array_2) => {
                            $obtain_kind_array!(_tmp_array_1, $array_name1, OriginalKindType, {
                                $obtain_kind_array!(_tmp_array_2, $array_name2, OtherKindType, {
                                    let $array_name2 = &$array_name2
                                        .mapv(|x| num::cast::<OtherKindType, OriginalKindType>(x).unwrap());
                                    $execution
                                })
                            })
                        }
                    }
                };
            }
    };
}

#[derive(Clone)]
#[non_exhaustive]
pub enum KindedArrayD {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum KindedArrayViewD<'a> {
    F32(ArrayViewD<'a, f32>),
    F64(ArrayViewD<'a, f64>),
    I16(ArrayViewD<'a, i16>),
    I32(ArrayViewD<'a, i32>),
    I64(ArrayViewD<'a, i64>),
}

#[derive(Debug)]
#[non_exhaustive]
pub enum KindedArrayViewMutD<'a> {
    F32(ArrayViewMutD<'a, f32>),
    F64(ArrayViewMutD<'a, f64>),
    I16(ArrayViewMutD<'a, i16>),
    I32(ArrayViewMutD<'a, i32>),
    I64(ArrayViewMutD<'a, i64>),
}

declare_kinded_array_variant!(
    ArrayD,
    KindedArrayD,
    kind,
    new_kinded_array,
    obtain_kind_array,
    obtain_2_kind_arrays
);

declare_kinded_array_variant!(
    ArrayViewD,
    KindedArrayViewD,
    view_kind,
    new_kinded_array_view,
    obtain_kind_array_view,
    obtain_2_kind_array_views
);

declare_kinded_array_variant!(
    ArrayViewMutD,
    KindedArrayViewMutD,
    view_mut_kind,
    new_kinded_array_view_mut,
    obtain_kind_array_view_mut,
    obtain_2_kind_array_view_muts
);

/// Get the real array `$array_name1` and `$array_name2` from two elements of our array type, `KindedArrayViewMutD` & `KindedArrayViewD`,
/// and set the type `OriginalKindType` to the type of the `$array_name1`,
/// and then cast the type of the `$array_name2` to `OriginalKindType` if necessary.
/// Then run the code in `$execution`.
/// Note: the `$array_name2` will be either of the same type as `$array_name1`, or an `ArrayD` if it has been casted.
macro_rules! obtain_2_kind_array_view_mut_with_immut {
    ($kind_array1:expr, $array_name1:ident, $kind_array2:expr, $array_name2:ident, $execution:block) => {
        match ($kind_array1, $kind_array2) {
            (KindedArrayViewMutD::F32($array_name1), KindedArrayViewD::F32($array_name2)) => {
                type OriginalKindType = f32;
                $execution
            }
            (KindedArrayViewMutD::F64($array_name1), KindedArrayViewD::F64($array_name2)) => {
                type OriginalKindType = f64;
                $execution
            }
            (KindedArrayViewMutD::I16($array_name1), KindedArrayViewD::I16($array_name2)) => {
                type OriginalKindType = i16;
                $execution
            }
            (KindedArrayViewMutD::I32($array_name1), KindedArrayViewD::I32($array_name2)) => {
                type OriginalKindType = i32;
                $execution
            }
            (KindedArrayViewMutD::I64($array_name1), KindedArrayViewD::I64($array_name2)) => {
                type OriginalKindType = i64;
                $execution
            }
            (_tmp_array_1, _tmp_array_2) => {
                obtain_kind_array_view_mut!(_tmp_array_1, $array_name1, OriginalKindType, {
                    obtain_kind_array_view!(_tmp_array_2, $array_name2, OtherKindType, {
                        let $array_name2 = &$array_name2
                            .mapv(|x| num::cast::<OtherKindType, OriginalKindType>(x).unwrap());
                        $execution
                    })
                })
            }
        }
    };
}

impl NdArrayTensor {
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
    fn i(&self) -> MutexGuard<'_, NdArrayTensorInternal> {
        self.internal.as_ref().unwrap().lock().unwrap()
    }

    /// Get a copy of the reference of the internal data of the tensor.
    ///
    /// Note: this is a shallow copy, so the data is not copied.
    fn i_copy(&self) -> Arc<Mutex<NdArrayTensorInternal>> {
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
    fn name_clone(&self) -> Self {
        Self { internal: self.internal.clone() }
    }
}

impl Clone for NdArrayTensor {
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

impl NdArrayTensorInternal {
    pub(crate) fn as_view<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, KindedArrayD>, Box<KindedArrayViewD<'a>>> {
        let data = self.data.read().unwrap();
        OwningHandle::new_with_fn(data, |data| {
            let tensor = unsafe { &*data };
            let mut view = tensor.view();
            for viewer in self.view.0.iter() {
                view = viewer.view(view);
            }
            Box::new(view)
        })
    }

    pub(crate) fn as_view_mut<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, KindedArrayD>, Box<KindedArrayViewMutD<'a>>> {
        let data = self.data.read().unwrap();
        OwningHandle::new_with_fn(data, |data| {
            let tensor = unsafe { &mut *(data as *mut KindedArrayD) };
            let mut view = tensor.view_mut();
            for viewer in self.view.0.iter() {
                view = viewer.view_mut(view);
            }
            Box::new(view)
        })
    }
}

// =================================================================================================
// Implementations for `NdArrayTensor` and `KindedArrayD`.
// =================================================================================================

impl TensorMethods for NdArrayTensor {
    fn zeros(shape: &[usize], dtype: TensorKind) -> Self {
        NdArrayTensor::from(KindedArrayD::zeros(shape, dtype))
    }

    fn ones(shape: &[usize], dtype: TensorKind) -> Self {
        NdArrayTensor::from(KindedArrayD::ones(shape, dtype))
    }

    fn size(&self) -> Vec<usize> {
        self.i().data.read().unwrap().size()
    }

    fn kind(&self) -> TensorKind {
        self.i().data.read().unwrap().kind()
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
        *self.i().as_view_mut() += &*other.i().as_view();
    }

    fn sub_(&mut self, other: &Self) {
        *self.i().as_view_mut() -= &*other.i().as_view();
    }

    fn mul_(&mut self, other: &Self) {
        *self.i().as_view_mut() *= &*other.i().as_view();
    }

    fn div_(&mut self, other: &Self) {
        *self.i().as_view_mut() /= &*other.i().as_view();
    }

    fn add_scalar<T: num::cast::NumCast + Copy + 'static>(&self, other: T) -> Self {
        AddScalarOp::forward(self, other)
    }

    fn sub_scalar<T: num::cast::NumCast + Copy + 'static>(&self, other: T) -> Self {
        SubScalarOp::forward(self, other)
    }

    fn mul_scalar<T: num::cast::NumCast + Copy + 'static>(&self, other: T) -> Self {
        MulScalarOp::forward(self, other)
    }

    fn div_scalar<T: num::cast::NumCast + Copy + 'static>(&self, other: T) -> Self {
        DivScalarOp::forward(self, other)
    }

    fn add_scalar_<T: num::cast::NumCast + Copy + 'static>(&mut self, other: T) {
        *self.i().as_view_mut() += other;
    }

    fn sub_scalar_<T: num::cast::NumCast + Copy + 'static>(&mut self, other: T) {
        *self.i().as_view_mut() -= other;
    }

    fn mul_scalar_<T: num::cast::NumCast + Copy + 'static>(&mut self, other: T) {
        *self.i().as_view_mut() *= other;
    }

    fn div_scalar_<T: num::cast::NumCast + Copy + 'static>(&mut self, other: T) {
        *self.i().as_view_mut() /= other;
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self {
        SumOp::forward(self, (dim.to_vec(), keep_dim))
    }
}

impl ArrayMethods for NdArrayTensor {
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

impl BorrowView for KindedArrayD {
    fn view(&self) -> KindedArrayViewD<'_> {
        obtain_kind_array!(self, array, { KindedArrayViewD::from(array.view()) })
    }

    fn view_mut(&mut self) -> KindedArrayViewMutD<'_> {
        obtain_kind_array!(self, array, { KindedArrayViewMutD::from(array.view_mut()) })
    }
}

impl TensorMethods for KindedArrayD {
    fn zeros(shape: &[usize], dtype: TensorKind) -> Self {
        new_kinded_array!(ArrayD::zeros(IxDyn(&shape)), dtype)
    }

    fn ones(shape: &[usize], dtype: TensorKind) -> Self {
        new_kinded_array!(ArrayD::ones(IxDyn(&shape)), dtype)
    }

    fn size(&self) -> Vec<usize> {
        obtain_kind_array!(self, array, { array.shape().to_vec() })
    }

    fn kind(&self) -> TensorKind {
        kind(self)
    }

    fn t(&self) -> Self {
        obtain_kind_array!(self, array, { KindedArrayD::from(array.t().into_owned()) })
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

    fn add_scalar<T: num::NumCast + Copy + 'static>(&self, other: T) -> Self {
        &self.view() + other
    }

    fn sub_scalar<T: num::NumCast + Copy + 'static>(&self, other: T) -> Self {
        &self.view() - other
    }

    fn mul_scalar<T: num::NumCast + Copy + 'static>(&self, other: T) -> Self {
        &self.view() * other
    }

    fn div_scalar<T: num::NumCast + Copy + 'static>(&self, other: T) -> Self {
        &self.view() / other
    }

    fn add_scalar_<T: num::NumCast + Copy + 'static>(&mut self, other: T) {
        let mut self_view = self.view_mut();
        self_view += other;
    }

    fn sub_scalar_<T: num::NumCast + Copy + 'static>(&mut self, other: T) {
        let mut self_view = self.view_mut();
        self_view -= other;
    }

    fn mul_scalar_<T: num::NumCast + Copy + 'static>(&mut self, other: T) {
        let mut self_view = self.view_mut();
        self_view *= other;
    }

    fn div_scalar_<T: num::NumCast + Copy + 'static>(&mut self, other: T) {
        let mut self_view = self.view_mut();
        self_view /= other;
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self {
        self.view().sum_dim(dim, keep_dim)
    }
}

arith_impl!(KindedArrayD);

// =================================================================================================
// Autograd methods
// =================================================================================================
impl AutoGradTensorMethods for NdArrayTensor {
    fn backward(&mut self) {
        if let Some(op) = self.i().op() {
            let size = self.size();
            let kind = self.kind();
            op.backward(NdArrayTensor::ones(&size, kind));
        }
    }

    fn grad(&self) -> Self {
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

arith_impl!(NdArrayTensor);

// =================================================================================================
// Implementations of quick ways to create `NdArrayTensor`, `KindedArrayD`, `KindedArrayViewD` and `KindedArrayViewMutD`.
// =================================================================================================
impl From<KindedArrayD> for NdArrayTensor {
    fn from(array: KindedArrayD) -> Self {
        Self::from(Arc::new(RwLock::new(array)))
    }
}

impl From<Arc<RwLock<KindedArrayD>>> for NdArrayTensor {
    fn from(array: Arc<RwLock<KindedArrayD>>) -> Self {
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
macro_rules! impl_from_arrayd_into_tensor {
    ($typ:ty, $from:ident, $data:expr) => {
        impl From<ArrayD<$typ>> for NdArrayTensor {
            fn from($from: ArrayD<$typ>) -> Self {
                NdArrayTensor::from($data)
            }
        }

        impl From<ArrayD<$typ>> for KindedArrayD {
            fn from($from: ArrayD<$typ>) -> Self {
                $data
            }
        }
    };
}

impl_from_arrayd_into_tensor!(f32, array, KindedArrayD::F32(array));
impl_from_arrayd_into_tensor!(f64, array, KindedArrayD::F64(array));
impl_from_arrayd_into_tensor!(i16, array, KindedArrayD::I16(array));
impl_from_arrayd_into_tensor!(i32, array, KindedArrayD::I32(array));
impl_from_arrayd_into_tensor!(i64, array, KindedArrayD::I64(array));

macro_rules! impl_from_arrayd_view_into_tensor {
    ($enum:ident, $array_type:ident, $typ:ty, $from:ident, $data:expr) => {
        impl<'a> From<$array_type<'a, $typ>> for $enum<'a> {
            fn from($from: $array_type<'a, $typ>) -> Self {
                $data
            }
        }
    };
}

impl_from_arrayd_view_into_tensor!(
    KindedArrayViewD,
    ArrayViewD,
    f32,
    array,
    KindedArrayViewD::F32(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewD,
    ArrayViewD,
    f64,
    array,
    KindedArrayViewD::F64(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewD,
    ArrayViewD,
    i16,
    array,
    KindedArrayViewD::I16(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewD,
    ArrayViewD,
    i32,
    array,
    KindedArrayViewD::I32(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewD,
    ArrayViewD,
    i64,
    array,
    KindedArrayViewD::I64(array)
);

impl_from_arrayd_view_into_tensor!(
    KindedArrayViewMutD,
    ArrayViewMutD,
    f32,
    array,
    KindedArrayViewMutD::F32(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewMutD,
    ArrayViewMutD,
    f64,
    array,
    KindedArrayViewMutD::F64(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewMutD,
    ArrayViewMutD,
    i16,
    array,
    KindedArrayViewMutD::I16(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewMutD,
    ArrayViewMutD,
    i32,
    array,
    KindedArrayViewMutD::I32(array)
);
impl_from_arrayd_view_into_tensor!(
    KindedArrayViewMutD,
    ArrayViewMutD,
    i64,
    array,
    KindedArrayViewMutD::I64(array)
);

// =================================================================================================
// View and ViewMut's Methods and Implementations
//
// Below are the bottom implementations of the tensor arithmetic methods.
// Upper structs' arithmetic methods should invoke these as needed.
// =================================================================================================
pub(crate) trait SuperViewMethods {
    type OwnedType;
    type ViewType<'a>;
    type ViewMutType<'a>;
}

pub(crate) trait ViewMutMethods<'this>: SuperViewMethods + 'this {
    fn upgrade(&self) -> Self::OwnedType;

    fn slice_mut<'a>(&'a mut self, info: IndexInfo) -> Self::ViewMutType<'a>;
    fn into_slice_mut(self, info: IndexInfo) -> Self::ViewMutType<'this>;
    fn permute_mut<'a>(&'a mut self, axes: &[usize]) -> Self::ViewMutType<'a>;
    fn into_permute_mut(self, axes: &[usize]) -> Self::ViewMutType<'this>;

    fn add_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn sub_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn mul_(&mut self, other: impl Borrow<Self::ViewType<'_>>);
    fn div_(&mut self, other: impl Borrow<Self::ViewType<'_>>);

    fn add_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn sub_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn mul_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn div_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);

    fn downgrade<'a>(&'a self) -> Self::ViewType<'a>;
}

pub(crate) trait ViewMethods<'this>: SuperViewMethods + 'this {
    fn kind(&self) -> TensorKind;
    fn size(&self) -> Vec<usize>;
    fn upgrade(&self) -> Self::OwnedType;

    fn slice<'a>(&'a self, info: IndexInfo) -> Self::ViewType<'a>;
    fn into_slice(self, info: IndexInfo) -> Self::ViewType<'this>;
    fn permute<'a>(&'a self, order: &[usize]) -> Self::ViewType<'a>;
    fn into_permute(self, order: &[usize]) -> Self::ViewType<'this>;
    fn broadcast<'a>(&'a self, shape: &[usize]) -> Self::ViewType<'a>;
    fn into_broadcast(self, shape: &[usize]) -> Self::ViewType<'this>;

    fn neg(&self) -> Self::OwnedType;
    fn matmul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn add(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn sub(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn mul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;
    fn div(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType;

    fn add_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType;
    fn sub_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType;
    fn mul_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType;
    fn div_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType;

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self::OwnedType;
}

macro_rules! impl_arith_for_all {
    ($own_type: ident, $view_type: ident, $view_mut_type: ident) => {
        impl<'a> std::ops::AddAssign<&$view_type<'_>> for $view_mut_type<'a> {
            fn add_assign(&mut self, rhs: &$view_type<'_>) {
                ViewMutMethods::add_(self, rhs);
            }
        }

        impl<'a> std::ops::SubAssign<&$view_type<'_>> for $view_mut_type<'a> {
            fn sub_assign(&mut self, rhs: &$view_type<'_>) {
                ViewMutMethods::sub_(self, rhs);
            }
        }

        impl<'a> std::ops::MulAssign<&$view_type<'_>> for $view_mut_type<'a> {
            fn mul_assign(&mut self, rhs: &$view_type<'_>) {
                ViewMutMethods::mul_(self, rhs);
            }
        }

        impl<'a> std::ops::DivAssign<&$view_type<'_>> for $view_mut_type<'a> {
            fn div_assign(&mut self, rhs: &$view_type<'_>) {
                ViewMutMethods::div_(self, rhs);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::AddAssign<T> for $view_mut_type<'_> {
            fn add_assign(&mut self, other: T) {
                ViewMutMethods::add_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::SubAssign<T> for $view_mut_type<'_> {
            fn sub_assign(&mut self, other: T) {
                ViewMutMethods::sub_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::MulAssign<T> for $view_mut_type<'_> {
            fn mul_assign(&mut self, other: T) {
                ViewMutMethods::mul_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::DivAssign<T> for $view_mut_type<'_> {
            fn div_assign(&mut self, other: T) {
                ViewMutMethods::div_scalar_(self, other);
            }
        }

        // View methods
        impl std::ops::Neg for &$view_type<'_> {
            type Output = $own_type;

            fn neg(self) -> Self::Output {
                ViewMethods::neg(self)
            }
        }

        impl std::ops::Add<&$view_type<'_>> for &$view_type<'_> {
            type Output = $own_type;

            fn add(self, rhs: &$view_type<'_>) -> Self::Output {
                ViewMethods::add(self, rhs)
            }
        }

        impl std::ops::Sub<&$view_type<'_>> for &$view_type<'_> {
            type Output = $own_type;

            fn sub(self, rhs: &$view_type<'_>) -> Self::Output {
                ViewMethods::sub(self, rhs)
            }
        }

        impl std::ops::Mul<&$view_type<'_>> for &$view_type<'_> {
            type Output = $own_type;

            fn mul(self, rhs: &$view_type<'_>) -> Self::Output {
                ViewMethods::mul(self, rhs)
            }
        }

        impl std::ops::Div<&$view_type<'_>> for &$view_type<'_> {
            type Output = $own_type;

            fn div(self, rhs: &$view_type<'_>) -> Self::Output {
                ViewMethods::div(self, rhs)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Add<T> for &$view_type<'_> {
            type Output = $own_type;

            fn add(self, other: T) -> Self::Output {
                ViewMethods::add_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Sub<T> for &$view_type<'_> {
            type Output = $own_type;

            fn sub(self, other: T) -> Self::Output {
                ViewMethods::sub_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Mul<T> for &$view_type<'_> {
            type Output = $own_type;

            fn mul(self, other: T) -> Self::Output {
                ViewMethods::mul_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Div<T> for &$view_type<'_> {
            type Output = $own_type;

            fn div(self, other: T) -> Self::Output {
                ViewMethods::div_scalar(self, other)
            }
        }
    };
}

impl_arith_for_all!(KindedArrayD, KindedArrayViewD, KindedArrayViewMutD);

impl SuperViewMethods for KindedArrayViewMutD<'_> {
    type OwnedType = KindedArrayD;

    type ViewType<'a> = KindedArrayViewD<'a>;

    type ViewMutType<'a> = KindedArrayViewMutD<'a>;
}

impl<'this> ViewMutMethods<'this> for KindedArrayViewMutD<'this> {
    fn upgrade(&self) -> Self::OwnedType {
        obtain_kind_array_view_mut!(self, array, { KindedArrayD::from(array.to_owned()) })
    }

    fn slice_mut<'a>(&'a mut self, info: IndexInfo) -> Self::ViewMutType<'a> {
        obtain_kind_array_view_mut!(self, array, {
            let info: Vec<SliceInfoElem> = info.into();
            KindedArrayViewMutD::from(array.slice_mut(info.as_slice()))
        })
    }

    fn into_slice_mut(self, info: IndexInfo) -> Self::ViewMutType<'this> {
        obtain_kind_array_view_mut!(self, array, {
            let info: Vec<SliceInfoElem> = info.into();
            KindedArrayViewMutD::from(array.slice_move(info.as_slice()))
        })
    }

    fn permute_mut<'a>(&'a mut self, axes: &[usize]) -> Self::ViewMutType<'a> {
        todo!()
    }

    fn into_permute_mut(self, axes: &[usize]) -> Self::ViewMutType<'this> {
        obtain_kind_array_view_mut!(self, array, {
            KindedArrayViewMutD::from(array.permuted_axes(axes))
        })
    }

    fn add_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other.borrow(), array2, {
            *array1 += array2;
        })
    }

    fn sub_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other.borrow(), array2, {
            *array1 -= array2;
        })
    }

    fn mul_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other.borrow(), array2, {
            *array1 *= array2;
        })
    }

    fn div_(&mut self, other: impl Borrow<Self::ViewType<'_>>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other.borrow(), array2, {
            *array1 /= array2;
        })
    }

    fn add_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array += cast::<T, KindType>(other).unwrap();
        })
    }

    fn sub_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array -= cast::<T, KindType>(other).unwrap();
        })
    }

    fn mul_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array *= cast::<T, KindType>(other).unwrap();
        })
    }

    fn div_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array /= cast::<T, KindType>(other).unwrap();
        })
    }

    fn downgrade(&self) -> Self::ViewType<'_> {
        obtain_kind_array_view_mut!(self, array, { KindedArrayViewD::from(array.view()) })
    }
}

impl SuperViewMethods for KindedArrayViewD<'_> {
    type OwnedType = KindedArrayD;

    type ViewType<'a> = KindedArrayViewD<'a>;

    type ViewMutType<'a> = KindedArrayViewMutD<'a>;
}

impl<'this> ViewMethods<'this> for KindedArrayViewD<'this> {
    fn upgrade(&self) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, { KindedArrayD::from(array.to_owned()) })
    }

    fn neg(&self) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, { KindedArrayD::from(array.mapv(|x| -x)) })
    }

    fn slice<'a>(&'a self, info: IndexInfo) -> Self::ViewType<'a> {
        obtain_kind_array_view!(self, array, {
            let info: Vec<SliceInfoElem> = info.into();
            KindedArrayViewD::from(array.slice(info.as_slice()))
        })
    }

    fn into_slice(self, info: IndexInfo) -> Self::ViewType<'this> {
        obtain_kind_array_view!(self, array, {
            let info: Vec<SliceInfoElem> = info.into();
            KindedArrayViewD::from(array.slice_move(info.as_slice()))
        })
    }

    fn permute<'a>(&'a self, order: &[usize]) -> Self::ViewType<'a> {
        todo!()
    }

    fn into_permute(self, order: &[usize]) -> Self::ViewType<'this> {
        obtain_kind_array_view!(self, array, {
            KindedArrayViewD::from(array.permuted_axes(order))
        })
    }

    fn broadcast<'a>(&'a self, shape: &[usize]) -> Self::ViewType<'a>{
        obtain_kind_array_view!(self, array, {
            KindedArrayViewD::from(array.broadcast(shape).unwrap())
        })
    }

    fn into_broadcast(self, shape: &[usize]) -> Self::ViewType<'this>{
        obtain_kind_array_view!(self, array, {
            // TODO: This is a hack to get around the borrow checker,
            // fooling it that we are not borrowing self.
            let array = &array as *const ArrayViewD<'_, KindType>;
            let array = unsafe { &*array };
            KindedArrayViewD::from(array.broadcast(shape).unwrap())
        })
    }

    fn add(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other.borrow(), array2, {
            KindedArrayD::from(array1 + array2)
        })
    }

    fn sub(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other.borrow(), array2, {
            KindedArrayD::from(array1 - array2)
        })
    }

    fn mul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other.borrow(), array2, {
            KindedArrayD::from(array1 * array2)
        })
    }

    fn div(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other.borrow(), array2, {
            KindedArrayD::from(array1 / array2)
        })
    }

    fn matmul(&self, other: impl Borrow<Self::ViewType<'_>>) -> Self::OwnedType {
        todo!()
    }

    fn add_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x + cast::<T, KindType>(other).unwrap()))
        })
    }

    fn sub_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x - cast::<T, KindType>(other).unwrap()))
        })
    }

    fn mul_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x * cast::<T, KindType>(other).unwrap()))
        })
    }

    fn div_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x / cast::<T, KindType>(other).unwrap()))
        })
    }

    fn kind(&self) -> TensorKind {
        view_kind(self)
    }

    fn size(&self) -> Vec<usize> {
        obtain_kind_array_view!(self, array, { array.shape().to_vec() })
    }

    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self::OwnedType {
        assert!(dim.len() > 0, "dim must not be empty");
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
}

impl<'a> KindedArrayViewD<'a> {
    fn sum_one_dim(&self, dim: usize, keep_dim: bool) -> KindedArrayD {
        obtain_kind_array_view!(self, array, {
            let mut array = array.sum_axis(Axis(dim));
            if keep_dim {
                array.insert_axis_inplace(Axis(dim));
            }
            KindedArrayD::from(array)
        })
    }

    fn unsqueeze(self, dim: usize) -> KindedArrayViewD<'a> {
        obtain_kind_array_view!(self, array, {
            KindedArrayViewD::from(array.insert_axis(Axis(dim)))
        })
    }
}
