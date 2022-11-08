use std::{
    borrow::Borrow,
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}, path::Path,
};

use tch::{Device, Kind, Tensor};

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

pub trait Element: Clone + tch::kind::Element {}

impl Element for u8 {}
impl Element for i8 {}
impl Element for i16 {}
impl Element for i32 {}
impl Element for i64 {}
impl Element for f32 {}
impl Element for f64 {}
impl Element for bool {}

/// A trait for a tensor-like object.
/// 
/// A tensor-like object should have the following properties:
/// - It should be able to perform element-wise operations, including addition, subtraction, multiplication, division, negation, and element-wise comparison, with other tensor-like objects.
/// - It should be associated with some env variable, which for example may be used to determine the device and data type of the tensor-like object, and a size indicating the length of each dimension.
/// - It should be able to be constructed with some factors, or other tensor-like objects.
/// - It should also be able to perform some other operations, such as reshaping, transposing, and slicing.
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
    /// The type of the env variable.
    type Env: Default;

    /// Get the env variable.
    fn env(&self) -> Self::Env;

    /// Get the size of the tensor-like object.
    fn size(&self) -> Vec<i64>;

    /// Create 
    fn of_slice<T: Element>(data: &[T]) -> Self;
    fn stack<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self;
    fn cat<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self;
    fn concat<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self;

    fn zeros(size: &[i64], env: Self::Env) -> Self;
    fn ones(size: &[i64], env: Self::Env) -> Self;
    fn empty(size: &[i64], env: Self::Env) -> Self;

    fn zeros_like(&self) -> Self {
        Self::zeros(&self.size(), self.env())
    }

    fn ones_like(&self) -> Self {
        Self::ones(&self.size(), self.env())
    }

    fn empty_like(&self) -> Self {
        Self::empty(&self.size(), self.env())
    }

    fn eye(size: i64, env: Self::Env) -> Self;

    fn read_npz<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error>;
    fn read_ot<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error>;
}

impl TensorLike for Tensor {
    type Env = (Kind, Device);

    fn env(&self) -> Self::Env {
        (self.kind(), self.device())
    }

    fn size(&self) -> Vec<i64> {
        self.size().iter().map(|x| *x as i64).collect()
    }

    fn of_slice<T: Element>(data: &[T]) -> Self {
        Tensor::of_slice(data)
    }

    fn stack<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self {
        Tensor::stack(tensors, dim)
    }

    fn cat<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self {
        Tensor::cat(tensors, dim)
    }

    fn concat<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self {
        Tensor::concat(tensors, dim)
    }

    fn zeros(size: &[i64], env: Self::Env) -> Self {
        Tensor::zeros(size, env)
    }

    fn ones(size: &[i64], env: Self::Env) -> Self {
        Tensor::ones(size, env)
    }

    fn empty(size: &[i64], env: Self::Env) -> Self {
        Tensor::empty(size, env)
    }

    fn eye(size: i64, env: Self::Env) -> Self {
        Tensor::eye(size, env)
    }

    fn read_npz<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error> {
        Ok(Tensor::read_npz(path)?)
    }

    fn read_ot<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error> {
        Ok(Tensor::load_multi(path)?)
    }
}
