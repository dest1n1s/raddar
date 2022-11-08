use std::{ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Neg}, fmt::Debug};

use tch::Tensor;

pub trait AddSelf<T> = Add<T, Output = T> + for<'a> Add<&'a T, Output = T> where T: Sized;
pub trait AddScalar<T> = Add<f64, Output = T> + Add<i64, Output = T> where T: Sized;
pub trait AddSelfWith<T> = where T: AddSelf<Self>;
pub trait AddScalarWith<T> = where T: AddScalar<Self>;
pub trait AddTrait = AddSelfWith<Self>
    + for<'a> AddSelfWith<&'a Self>
    + AddScalarWith<Self>
    + for<'a> AddScalarWith<&'a Self>
    + AddSelfWith<f64>
    + AddSelfWith<i64>
    + AddSelfWith<f32>
    + AddSelfWith<i32>;

pub trait SubSelf<T> = Sub<T, Output = T> + for<'a> Sub<&'a T, Output = T> where T: Sized;
pub trait SubScalar<T> = Sub<f64, Output = T> + Sub<i64, Output = T> where T: Sized;
pub trait SubSelfWith<T> = where T: SubSelf<Self>;
pub trait SubScalarWith<T> = where T: SubScalar<Self>;
pub trait SubTrait = SubSelfWith<Self>
    + for<'a> SubSelfWith<&'a Self>
    + SubScalarWith<Self>
    + for<'a> SubScalarWith<&'a Self>
    + SubSelfWith<f64>
    + SubSelfWith<i64>
    + SubSelfWith<f32>
    + SubSelfWith<i32>;

pub trait MulSelf<T> = Mul<T, Output = T> + for<'a> Mul<&'a T, Output = T> where T: Sized;
pub trait MulScalar<T> = Mul<f64, Output = T> + Mul<i64, Output = T> where T: Sized;
pub trait MulSelfWith<T> = where T: MulSelf<Self>;
pub trait MulScalarWith<T> = where T: MulScalar<Self>;
pub trait MulTrait = MulSelfWith<Self>
    + for<'a> MulSelfWith<&'a Self>
    + MulScalarWith<Self>
    + for<'a> MulScalarWith<&'a Self>
    + MulSelfWith<f64>
    + MulSelfWith<i64>
    + MulSelfWith<f32>
    + MulSelfWith<i32>;

pub trait DivSelf<T> = Div<T, Output = T> + for<'a> Div<&'a T, Output = T> where T: Sized;
pub trait DivScalar<T> = Div<f64, Output = T> + Div<i64, Output = T> where T: Sized;
pub trait DivSelfWith<T> = where T: DivSelf<Self>;
pub trait DivScalarWith<T> = where T: DivScalar<Self>;
pub trait DivTrait = DivSelfWith<Self>
    + for<'a> DivSelfWith<&'a Self>
    + DivScalarWith<Self>
    + for<'a> DivScalarWith<&'a Self>
    + DivSelfWith<f64>
    + DivSelfWith<i64>
    + DivSelfWith<f32>
    + DivSelfWith<i32>;

pub trait AddAssignTrait = AddAssign<Self>
    + for<'a> AddAssign<&'a Self>
    + AddAssign<f64>
    + AddAssign<i64>
    + AddAssign<f32>
    + AddAssign<i32>
    + Sized;

pub trait SubAssignTrait = SubAssign<Self>
    + for<'a> SubAssign<&'a Self>
    + SubAssign<f64>
    + SubAssign<i64>
    + SubAssign<f32>
    + SubAssign<i32>
    + Sized;

pub trait MulAssignTrait = MulAssign<Self>
    + for<'a> MulAssign<&'a Self>
    + MulAssign<f64>
    + MulAssign<i64>
    + MulAssign<f32>
    + MulAssign<i32>
    + Sized;

pub trait DivAssignTrait = DivAssign<Self>
    + for<'a> DivAssign<&'a Self>
    + DivAssign<f64>
    + DivAssign<i64>
    + DivAssign<f32>
    + DivAssign<i32>
    + Sized;

pub trait NegTrait = Neg<Output = Self> + for<'a> Neg<Output = Self> + Sized;

pub trait TensorLike:
    AddTrait
    + SubTrait
    + MulTrait
    + DivTrait
    + AddAssignTrait
    + SubAssignTrait
    + MulAssignTrait
    + DivAssignTrait
    + NegTrait
    + PartialEq<Self>
    + AsRef<Self>
    + Debug
    + Default
{
}

impl TensorLike for Tensor {}
