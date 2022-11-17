use std::sync::{Arc, Mutex, MutexGuard};

use self::ops::{AddOp, GradAccumulateOp, SubOp, NegOp, TransposeOp};
use crate::{
    arith_impl,
    tensor::{ops::Operation, AutoGradTensorMethods, TensorKind, TensorMethods},
};
use ndarray::{ArrayD, IxDyn};
use num::cast;

pub mod ops;

pub struct NdArrayTensor {
    internal: Option<Arc<Mutex<NdArrayTensorInternal>>>,
}

pub(crate) struct NdArrayTensorInternal {
    data: KindedArrayD,
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

#[derive(Clone)]
pub enum KindedArrayD {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
}

fn kind(kinded_array: &KindedArrayD) -> TensorKind {
    match kinded_array {
        KindedArrayD::F32(_) => TensorKind::F32,
        KindedArrayD::F64(_) => TensorKind::F64,
        KindedArrayD::I16(_) => TensorKind::I16,
        KindedArrayD::I32(_) => TensorKind::I32,
        KindedArrayD::I64(_) => TensorKind::I64,
    }
}

macro_rules! new_kinded_array {
    ($data:expr, $kind:expr) => {
        match $kind {
            TensorKind::F32 => KindedArrayD::F32($data),
            TensorKind::F64 => KindedArrayD::F64($data),
            TensorKind::I16 => KindedArrayD::I16($data),
            TensorKind::I32 => KindedArrayD::I32($data),
            TensorKind::I64 => KindedArrayD::I64($data),
            _ => unimplemented!(),
        }
    };
}

/// Get the real ArrayD `$array_name` from a KindedArrayD `$kind_array`, and
/// set the type `$kind_type_name` to the type of the array.
/// Then run the code in `$execution`.
macro_rules! obtain_kind_array {
    ($kind_array:expr, $array_name:ident, $kind_type_name:ident, $execution:block) => {
        match $kind_array {
            KindedArrayD::F32($array_name) => {
                type $kind_type_name = f32;
                $execution
            }
            KindedArrayD::F64($array_name) => {
                type $kind_type_name = f64;
                $execution
            }
            KindedArrayD::I16($array_name) => {
                type $kind_type_name = i16;
                $execution
            }
            KindedArrayD::I32($array_name) => {
                type $kind_type_name = i32;
                $execution
            }
            KindedArrayD::I64($array_name) => {
                type $kind_type_name = i64;
                $execution
            }
        }
    };
    ($kind_array:expr, $array_name:ident, $execution:block) => {
        obtain_kind_array!($kind_array, $array_name, KindType, $execution)
    };
}

/// Get the real ArrayD `$array_name1` and `$array_name2` from two KindedArrayD,
/// and set the type `OriginalKindType` to the type of the `$array_name1`,
/// also cast the type of the `$array_name2` to `OriginalKindType` if necessary.
/// Then run the code in `$execution`.
macro_rules! obtain_2_kind_arrays {
    ($kind_array1:expr, $array_name1:ident,$kind_array2:expr, $array_name2:ident, $execution:block) => {
        match ($kind_array1, $kind_array2) {
            (KindedArrayD::F32($array_name1), KindedArrayD::F32($array_name2)) => {
                type OriginalKindType = f32;
                $execution
            }
            (KindedArrayD::F64($array_name1), KindedArrayD::F64($array_name2)) => {
                type OriginalKindType = f64;
                $execution
            }
            (KindedArrayD::I16($array_name1), KindedArrayD::I16($array_name2)) => {
                type OriginalKindType = i16;
                $execution
            }
            (KindedArrayD::I32($array_name1), KindedArrayD::I32($array_name2)) => {
                type OriginalKindType = i32;
                $execution
            }
            (KindedArrayD::I64($array_name1), KindedArrayD::I64($array_name2)) => {
                type OriginalKindType = i64;
                $execution
            }
            (_tmp_array_1, _tmp_array_2) => {
                obtain_kind_array!(_tmp_array_1, $array_name1, OriginalKindType, {
                    obtain_kind_array!(_tmp_array_2, $array_name2, OtherKindType, {
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
                obtain_kind_array!(&kinded_array.lock().unwrap().data, array, {
                    println!("{:?}", array);
                });
            }
            None => println!("None"),
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
impl TensorMethods for NdArrayTensor {
    fn zeros(shape: &[usize], dtype: TensorKind) -> Self {
        NdArrayTensor::from(KindedArrayD::zeros(shape, dtype))
    }

    fn ones(shape: &[usize], dtype: TensorKind) -> Self {
        NdArrayTensor::from(KindedArrayD::ones(shape, dtype))
    }

    fn size(&self) -> Vec<usize> {
        self.i().data.size()
    }

    fn kind(&self) -> TensorKind {
        self.i().data.kind()
    }

    fn t(&self) -> Self {
        TransposeOp::forward(self)
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
        self.i().data += &other.i().data;
    }

    fn sub_(&mut self, other: &Self) {
        self.i().data -= &other.i().data;
    }

    fn mul_(&mut self, other: &Self) {
        self.i().data *= &other.i().data;
    }

    fn div_(&mut self, other: &Self) {
        self.i().data /= &other.i().data;
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
        self.i().data += other;
    }

    fn sub_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        self.i().data -= other;
    }

    fn mul_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        self.i().data *= other;
    }

    fn div_scalar_<T: num::cast::NumCast + Copy>(&mut self, other: T) {
        self.i().data /= other;
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
        NdArrayTensor {
            internal: Some(Arc::new(Mutex::new(NdArrayTensorInternal {
                data: array,
                op: None,
                grad: None,
                requires_grad: false,
                is_leaf: true,
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
