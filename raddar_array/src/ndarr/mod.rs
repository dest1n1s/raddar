use std::{
    ops::{AddAssign, DivAssign, MulAssign, SubAssign, Add},
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard},
};

use self::ops::{GradAccumulateOp, AddOp, SubOp, NegOp};
use crate::{
    arith_impl,
    tensor::{ops::Operation, AutoGradTensorMethods, TensorKind, TensorMethods},
};
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
use num::{cast, NumCast};
use owning_ref::OwningHandle;

pub mod ops;

pub struct NdArrayTensor {
    internal: Option<Arc<Mutex<NdArrayTensorInternal>>>,
}
#[derive(Clone)]
pub(crate) enum ViewType {
    All,
    Other(Arc<dyn AsView>),
}

pub(crate) trait AsView {
    fn view<'a>(&self, tensor: &KindedArrayD) -> KindedArrayViewD<'a>;
    fn view_mut<'a>(&self, tensor: &mut KindedArrayD) -> KindedArrayViewMutD<'a>;
    fn op(&self) -> Arc<dyn Operation>;
}

pub(crate) struct NdArrayTensorInternal {
    view: ViewType,
    data: Arc<RwLock<KindedArrayD>>,
    is_leaf: bool,
    requires_grad: bool,
    grad: Option<KindedArrayD>,
    op: Option<Arc<dyn Operation>>,
}

pub(crate) trait IntoOp {
    fn op(self) -> Option<Arc<dyn Operation>>;
}

impl<'a> IntoOp for MutexGuard<'a, NdArrayTensorInternal> {
    fn op(self) -> Option<Arc<dyn Operation>> {
        self.op.clone()
    }
}
// Since we have protected the data with a mutex, we can safely implement Send and Sync
unsafe impl Send for NdArrayTensorInternal {}
unsafe impl Sync for NdArrayTensorInternal {}

macro_rules! declare_kinded_array_variant {
    ($array_type: ident, $enum: ident, $kind_fn: ident, $new_kinded_array: ident,
        $obtain_kind_array: ident, $obtain_2_kind_arrays: ident) => {
            fn $kind_fn(kinded_array: &$enum) -> TensorKind {
                match kinded_array {
                    $enum::F32(_) => TensorKind::F32,
                    $enum::F64(_) => TensorKind::F64,
                    $enum::I16(_) => TensorKind::I16,
                    $enum::I32(_) => TensorKind::I32,
                    $enum::I64(_) => TensorKind::I64,
                }
            }

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

            /// Get the real ArrayD `$array_name` from a KindedArrayD `$kind_array`, and
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

            /// Get the real ArrayD `$array_name1` and `$array_name2` from two KindedArrayD,
            /// and set the type `OriginalKindType` to the type of the `$array_name1`,
            /// also cast the type of the `$array_name2` to `OriginalKindType` if necessary.
            /// Then run the code in `$execution`.
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

#[derive(Debug)]
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

    pub(crate) fn plus_one(&self) {
        match self.internal {
            Some(ref kinded_array) => {
                let lock = kinded_array.lock().unwrap();
                obtain_kind_array_view_mut!(&mut *lock.as_view_mut(), array, OriginalKindType, {
                    array.mapv_inplace(|x| x + cast::<i32,OriginalKindType>(1).unwrap());
                });
            }
            None => {}
        }
    }

    /// Get the internal data of the tensor.
    fn i(&self) -> MutexGuard<'_, NdArrayTensorInternal> {
        self.internal.as_ref().unwrap().lock().unwrap()
    }

    fn i_copy(&self) -> Arc<Mutex<NdArrayTensorInternal>> {
        self.internal.as_ref().unwrap().clone()
    }
}

impl Clone for NdArrayTensor {
    fn clone(&self) -> Self {
        match self.internal {
            Some(ref tensor_internal) => {
                let tensor_internal = tensor_internal.lock().unwrap();
                Self::from(tensor_internal.data.clone())
            }
            None => Self::none(),
        }
    }
}

impl NdArrayTensorInternal {
    pub(crate) fn as_view<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, KindedArrayD>, Box<KindedArrayViewD>> {
        let data = self.data.read().unwrap();
        match self.view {
            ViewType::All => OwningHandle::new_with_fn(data, |data| {
                obtain_kind_array!(unsafe { &*data }, array, {
                    Box::new(KindedArrayViewD::from(array.view()))
                })
            }),
            ViewType::Other(ref as_view) => {
                OwningHandle::new_with_fn(data, |data| Box::new(as_view.view(unsafe { &*data })))
            }
        }
    }

    pub(crate) fn as_view_mut<'a>(
        &'a self,
    ) -> OwningHandle<RwLockReadGuard<'a, KindedArrayD>, Box<KindedArrayViewMutD>> {
        let data = self.data.read().unwrap();
        match self.view {
            ViewType::All => OwningHandle::new_with_fn(data, |data| {
                obtain_kind_array!(unsafe { &mut *(data as *mut _) }, array, {
                    Box::new(KindedArrayViewMutD::from(array.view_mut()))
                })
            }),
            ViewType::Other(ref as_view) => OwningHandle::new_with_fn(data, |data| {
                Box::new(as_view.view_mut(unsafe { &mut *(data as *mut _) }))
            }),
        }
    }
}



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
        todo!()
    }

    fn neg(&self) -> Self {
        NegOp::forward(self)
    }

    fn add(&self, other: &Self) -> Self {
        AddOp::forward((self, other))
    }

    fn sub(&self, other: &Self) -> Self {
        SubOp::forward((self, other))
    }

    fn mul(&self, other: &Self) -> Self {
        todo!()
    }

    fn div(&self, other: &Self) -> Self {
        todo!()
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

    fn add_scalar<T: num::cast::NumCast + Copy>(&self, other: T) -> Self {
        todo!()
    }

    fn sub_scalar<T: num::cast::NumCast + Copy>(&self, other: T) -> Self {
        todo!()
    }

    fn mul_scalar<T: num::cast::NumCast + Copy>(&self, other: T) -> Self {
        todo!()
    }

    fn div_scalar<T: num::cast::NumCast + Copy>(&self, other: T) -> Self {
        todo!()
    }

    fn add_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        *self.i().as_view_mut() += other;
    }

    fn sub_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        *self.i().as_view_mut() -= other;
    }

    fn mul_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        *self.i().as_view_mut() *= other;
    }

    fn div_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        *self.i().as_view_mut() /= other;
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
        obtain_kind_array!(self, array, { KindedArrayD::from(-array) })
    }

    fn add(&self, other: &Self) -> Self {
        obtain_2_kind_arrays!(self, array1, other, array2, {
            KindedArrayD::from(array1 + array2)
        })
    }

    fn sub(&self, other: &Self) -> Self {
        obtain_2_kind_arrays!(self, array1, other, array2, {
            KindedArrayD::from(array1 - array2)
        })
    }

    fn mul(&self, other: &Self) -> Self {
        obtain_2_kind_arrays!(self, array1, other, array2, {
            KindedArrayD::from(array1 * array2)
        })
    }

    fn div(&self, other: &Self) -> Self {
        obtain_2_kind_arrays!(self, array1, other, array2, {
            KindedArrayD::from(array1 / array2)
        })
    }

    fn add_(&mut self, other: &Self) {
        obtain_2_kind_arrays!(self, array1, &other, array2, {
            *array1 += array2;
        });
    }

    fn sub_(&mut self, other: &Self) {
        obtain_2_kind_arrays!(self, array1, &other, array2, {
            *array1 -= array2;
        });
    }

    fn mul_(&mut self, other: &Self) {
        obtain_2_kind_arrays!(self, array1, &other, array2, {
            *array1 *= array2;
        });
    }

    fn div_(&mut self, other: &Self) {
        obtain_2_kind_arrays!(self, array1, &other, array2, {
            *array1 /= array2;
        });
    }

    fn add_scalar<T: num::NumCast + Copy>(&self, other: T) -> Self {
        obtain_kind_array!(self, array, {
            KindedArrayD::from(array + cast::<T, KindType>(other).unwrap())
        })
    }

    fn sub_scalar<T: num::NumCast + Copy>(&self, other: T) -> Self {
        obtain_kind_array!(self, array, {
            KindedArrayD::from(array - cast::<T, KindType>(other).unwrap())
        })
    }

    fn mul_scalar<T: num::NumCast + Copy>(&self, other: T) -> Self {
        obtain_kind_array!(self, array, {
            KindedArrayD::from(array * cast::<T, KindType>(other).unwrap())
        })
    }

    fn div_scalar<T: num::NumCast + Copy>(&self, other: T) -> Self {
        obtain_kind_array!(self, array, {
            KindedArrayD::from(array / cast::<T, KindType>(other).unwrap())
        })
    }

    fn add_scalar_<T: num::NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array!(self, array, {
            *array += cast::<T, KindType>(other).unwrap();
        });
    }

    fn sub_scalar_<T: num::NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array!(self, array, {
            *array -= cast::<T, KindType>(other).unwrap();
        });
    }

    fn mul_scalar_<T: num::NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array!(self, array, {
            *array *= cast::<T, KindType>(other).unwrap();
        });
    }

    fn div_scalar_<T: num::NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array!(self, array, {
            *array /= cast::<T, KindType>(other).unwrap();
        });
    }
}

arith_impl!(KindedArrayD);

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
                view: ViewType::All,
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

pub(crate) trait ViewMutMethods {
    type OwnedType;
    type ViewType<'a>;
    fn upgrade(&self) -> Self::OwnedType;

    fn add_(&mut self, other: &Self::ViewType<'_>);
    fn sub_(&mut self, other: &Self::ViewType<'_>);
    fn mul_(&mut self, other: &Self::ViewType<'_>);
    fn div_(&mut self, other: &Self::ViewType<'_>);

    fn add_scalar_<T: NumCast + Copy>(&mut self, other: T);
    fn sub_scalar_<T: NumCast + Copy>(&mut self, other: T);
    fn mul_scalar_<T: NumCast + Copy>(&mut self, other: T);
    fn div_scalar_<T: NumCast + Copy>(&mut self, other: T);

    fn downgrade(&self) -> Self::ViewType<'_>;
}

pub(crate) trait ViewMethods {
    type OwnedType;
    type ViewType<'a>;

    fn kind(&self) -> TensorKind;
    fn size(&self) -> Vec<usize>;
    fn upgrade(&self) -> Self::OwnedType;

    fn neg(&self) -> Self::OwnedType;
    fn add(&self, other: &Self::ViewType<'_>) -> Self::OwnedType;
    fn sub(&self, other: &Self::ViewType<'_>) -> Self::OwnedType;
    fn mul(&self, other: &Self::ViewType<'_>) -> Self::OwnedType;
    fn div(&self, other: &Self::ViewType<'_>) -> Self::OwnedType;

    fn add_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType;
    fn sub_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType;
    fn mul_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType;
    fn div_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType;
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

        impl<T: num::NumCast + Copy> std::ops::AddAssign<T> for $view_mut_type<'_> {
            fn add_assign(&mut self, other: T) {
                ViewMutMethods::add_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy> std::ops::SubAssign<T> for $view_mut_type<'_> {
            fn sub_assign(&mut self, other: T) {
                ViewMutMethods::sub_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy> std::ops::MulAssign<T> for $view_mut_type<'_> {
            fn mul_assign(&mut self, other: T) {
                ViewMutMethods::mul_scalar_(self, other);
            }
        }

        impl<T: num::NumCast + Copy> std::ops::DivAssign<T> for $view_mut_type<'_> {
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

        impl<T: num::NumCast + Copy> std::ops::Add<T> for &$view_type<'_> {
            type Output = $own_type;

            fn add(self, other: T) -> Self::Output {
                ViewMethods::add_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy> std::ops::Sub<T> for &$view_type<'_> {
            type Output = $own_type;

            fn sub(self, other: T) -> Self::Output {
                ViewMethods::sub_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy> std::ops::Mul<T> for &$view_type<'_> {
            type Output = $own_type;

            fn mul(self, other: T) -> Self::Output {
                ViewMethods::mul_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy> std::ops::Div<T> for &$view_type<'_> {
            type Output = $own_type;

            fn div(self, other: T) -> Self::Output {
                ViewMethods::div_scalar(self, other)
            }
        }
    };
}

impl_arith_for_all!(KindedArrayD, KindedArrayViewD, KindedArrayViewMutD);

impl ViewMutMethods for KindedArrayViewMutD<'_>{
    type OwnedType = KindedArrayD;

    type ViewType<'a> = KindedArrayViewD<'a>;

    fn upgrade(&self) -> Self::OwnedType {
        todo!()
    }

    fn add_(&mut self, other: &Self::ViewType<'_>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other, array2, {
            *array1 += array2;
        })
    }

    fn sub_(&mut self, other: &Self::ViewType<'_>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other, array2, {
            *array1 -= array2;
        })
    }

    fn mul_(&mut self, other: &Self::ViewType<'_>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other, array2, {
            *array1 *= array2;
        })
    }

    fn div_(&mut self, other: &Self::ViewType<'_>) {
        obtain_2_kind_array_view_mut_with_immut!(self, array1, other, array2, {
            *array1 /= array2;
        })
    }

    fn add_scalar_<T: NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array += cast::<T, KindType>(other).unwrap();
        })
    }

    fn sub_scalar_<T: NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array -= cast::<T, KindType>(other).unwrap();
        })
    }

    fn mul_scalar_<T: NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array *= cast::<T, KindType>(other).unwrap();
        })
    }

    fn div_scalar_<T: NumCast + Copy>(&mut self, other: T) {
        obtain_kind_array_view_mut!(self, array, {
            *array /= cast::<T, KindType>(other).unwrap();
        })
    }

    fn downgrade(&self) -> Self::ViewType<'_> {
        obtain_kind_array_view_mut!(self, array, { KindedArrayViewD::from(array.view()) })
    }
}

impl ViewMethods for KindedArrayViewD<'_> {
    type OwnedType = KindedArrayD;
    type ViewType<'a> = KindedArrayViewD<'a>;

    fn upgrade(&self) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, { KindedArrayD::from(array.to_owned()) })
    }

    fn neg(&self) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, { KindedArrayD::from(array.mapv(|x| -x)) })
    }

    fn add(&self, other: &KindedArrayViewD<'_>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other, array2, {
            KindedArrayD::from(array1 + array2)
        })
    }

    fn sub(&self, other: &KindedArrayViewD<'_>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other, array2, {
            KindedArrayD::from(array1 - array2)
        })
    }

    fn mul(&self, other: &KindedArrayViewD<'_>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other, array2, {
            KindedArrayD::from(array1 * array2)
        })
    }

    fn div(&self, other: &KindedArrayViewD<'_>) -> Self::OwnedType {
        obtain_2_kind_array_views!(self, array1, other, array2, {
            KindedArrayD::from(array1 / array2)
        })
    }

    fn add_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x + cast::<T, KindType>(other).unwrap()))
        })
    }

    fn sub_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x - cast::<T, KindType>(other).unwrap()))
        })
    }

    fn mul_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType {
        obtain_kind_array_view!(self, array, {
            KindedArrayD::from(array.mapv(|x| x * cast::<T, KindType>(other).unwrap()))
        })
    }

    fn div_scalar<T: NumCast + Copy>(&self, other: T) -> Self::OwnedType {
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

}