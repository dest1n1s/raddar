use std::{
    borrow::Borrow,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use tch::Tensor;

use super::{TensorLike, Element};

// Trait bounds for tensor-like objects that can perform arithmetic operations.
// 
// These bounds include:
// - `Tensor` and `Tensor`, `Tensor` and `&Tensor`, `&Tensor` and `Tensor`, `&Tensor` and `&Tensor`, `Tensor` and `Scalar`, `Scalar` and `Tensor`, `&Tensor` and `Scalar`, `Scalar` and `&Tensor` can be added, subtracted, multiplied, and divided.
// - `Tensor` and `Tensor`, `Tensor` and `&Tensor`, `Tensor` and `Scalar` can be added, subtracted, multiplied, and divided in place.
// - `Tensor` can be negated.
// 
// Scalar types include `i32`, `i64`, `f32`, `f64`.

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


/// A trait for a tensor-like object that can perform basic algebraic operations, including arithmetic operations, matrix multiplication, trigonometric functions, and so on.
///
/// Some more advanced operations, such as gradient calculation, convolution, and so on, are not included in this trait.
pub trait TensorOps:
    TensorLike
    + AddTrait
    + SubTrait
    + MulTrait
    + DivTrait
    + AddAssignTrait
    + SubAssignTrait
    + MulAssignTrait
    + DivAssignTrait
    + NegTrait
{
    /// Calculate the sum of all elements in the tensor-like object.
    fn sum(&self) -> Self;

    /// Calculate the sum of each row of the tensor-like object in the given dimensions `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    fn sum_dim(&self, dim: &[i64], keep_dim: bool) -> Self;

    /// Calculate the mean of all elements in the tensor-like object.
    fn mean(&self) -> Self;

    /// Calculate the mean of each row of the tensor-like object in the given dimensions `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    fn mean_dim(&self, dim: &[i64], keep_dim: bool) -> Self;

    /// Calculate the maximum of all elements in the tensor-like object.
    fn max(&self) -> Self;

    /// Calculate the maximum of each row of the tensor-like object in the given dimension `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    /// 
    /// Returns a tuple of the maximum value and the index of the maximum value.
    fn max_dim(&self, dim: i64, keep_dim: bool) -> (Self, Self);

    /// Calculate the minimum of all elements in the tensor-like object.
    fn min(&self) -> Self;

    /// Calculate the minimum of each row of the tensor-like object in the given dimension `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    /// 
    /// Returns a tuple of the minimum value and the index of the minimum value.
    fn min_dim(&self, dim: i64, keep_dim: bool) -> (Self, Self);

    /// Calculate the argmax of all elements in the tensor-like object.
    fn argmax(&self) -> Self;

    /// Calculate the argmax of each row of the tensor-like object in the given dimension `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    fn argmax_dim(&self, dim: i64, keep_dim: bool) -> Self;

    /// Calculate the argmin of all elements in the tensor-like object.
    fn argmin(&self) -> Self;

    /// Calculate the argmin of each row of the tensor-like object in the given dimension `dim`.
    /// 
    /// If `keepdim` is true, the dimensions of the tensor-like object will be kept.
    fn argmin_dim(&self, dim: i64, keep_dim: bool) -> Self;

    /// Calculate the transpose of the tensor-like object.
    fn transpose(&self, dim0: i64, dim1: i64) -> Self;

    /// Calculate the matrix multiplication of the tensor-like object and the given tensor-like object.
    fn matmul<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Calculate the absolute value of the tensor-like object.
    fn abs(&self) -> Self;

    /// Calculate the element-wise square of the tensor-like object.
    fn square(&self) -> Self;

    /// Calculate the element-wise square root of the tensor-like object.
    fn sqrt(&self) -> Self;

    /// Calculate the element-wise exponential of the tensor-like object.
    fn exp(&self) -> Self;

    /// Calculate the element-wise natural logarithm of the tensor-like object.
    fn log(&self) -> Self;

    /// Calculate the element-wise sine of the tensor-like object.
    fn sin(&self) -> Self;

    /// Calculate the element-wise cosine of the tensor-like object.
    fn cos(&self) -> Self;

    /// Calculate the element-wise tangent of the tensor-like object.
    fn tan(&self) -> Self;

    /// Calculate the element-wise hyperbolic sine of the tensor-like object.
    fn sinh(&self) -> Self;

    /// Calculate the element-wise hyperbolic cosine of the tensor-like object.
    fn cosh(&self) -> Self;

    /// Calculate the element-wise hyperbolic tangent of the tensor-like object.
    fn tanh(&self) -> Self;

    /// Calculate the element-wise inverse sine of the tensor-like object.
    fn asin(&self) -> Self;

    /// Calculate the element-wise inverse cosine of the tensor-like object.
    fn acos(&self) -> Self;

    /// Calculate the element-wise inverse tangent of the tensor-like object.
    fn atan(&self) -> Self;

    /// Calculate the element-wise inverse hyperbolic sine of the tensor-like object.
    fn asinh(&self) -> Self;

    /// Calculate the element-wise inverse hyperbolic cosine of the tensor-like object.
    fn acosh(&self) -> Self;

    /// Calculate the element-wise inverse hyperbolic tangent of the tensor-like object.
    fn atanh(&self) -> Self;

    /// Compute the element-wise less-than comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn lt<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise less-than comparison of the tensor-like object and the given scalar.
    fn lt_scalar<S: Element>(&self, other: S) -> Self;

    /// Compute the element-wise less-than-or-equal comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn le<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise less-than-or-equal comparison of the tensor-like object and the given scalar.
    fn le_scalar<S: Element>(&self, other: S) -> Self;

    /// Compute the element-wise greater-than comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn gt<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise greater-than comparison of the tensor-like object and the given scalar.
    fn gt_scalar<S: Element>(&self, other: S) -> Self;

    /// Compute the element-wise greater-than-or-equal comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn ge<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise greater-than-or-equal comparison of the tensor-like object and the given scalar.
    fn ge_scalar<S: Element>(&self, other: S) -> Self;

    /// Compute the element-wise equality comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn eq<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise equality comparison of the tensor-like object and the given scalar.
    fn eq_scalar<S: Element>(&self, other: S) -> Self;

    /// Compute the element-wise inequality comparison of the tensor-like object and the given tensor-like object.
    /// 
    /// The given tensor-like object must have a shape that is broadcastable to the shape of the tensor-like object.
    fn ne<T: Borrow<Self>>(&self, other: T) -> Self;

    /// Compute the element-wise inequality comparison of the tensor-like object and the given scalar.
    fn ne_scalar<S: Element>(&self, other: S) -> Self;
}



impl TensorOps for Tensor {
    fn sum(&self) -> Self {
        Tensor::sum(self, self.kind())
    }

    fn sum_dim(&self, dim: &[i64], keep_dim: bool) -> Self {
        Tensor::sum_dim_intlist(self, dim, keep_dim, self.kind())
    }

    fn mean(&self) -> Self {
        Tensor::mean(self, self.kind())
    }

    fn mean_dim(&self, dim: &[i64], keep_dim: bool) -> Self {
        Tensor::mean_dim(self, dim, keep_dim, self.kind())
    }

    fn max(&self) -> Self {
        Tensor::max(self)
    }

    fn max_dim(&self, dim: i64, keep_dim: bool) -> (Self, Self) {
        Tensor::max_dim(self, dim, keep_dim)
    }

    fn min(&self) -> Self {
        Tensor::min(self)
    }

    fn min_dim(&self, dim: i64, keep_dim: bool) -> (Self, Self) {
        Tensor::min_dim(self, dim, keep_dim)
    }

    fn argmax(&self) -> Self {
        Tensor::argmax(self, None, false)
    }

    fn argmax_dim(&self, dim: i64, keep_dim: bool) -> Self {
        Tensor::argmax(self, Some(dim), keep_dim)
    }

    fn argmin(&self) -> Self {
        Tensor::argmin(self, None, false)
    }

    fn argmin_dim(&self, dim: i64, keep_dim: bool) -> Self {
        Tensor::argmin(self, Some(dim), keep_dim)
    }

    fn transpose(&self, dim0: i64, dim1: i64) -> Self {
        Tensor::transpose(self, dim0, dim1)
    }

    fn matmul<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::matmul(self, other.borrow())
    }
    
    fn abs(&self) -> Self {
        Tensor::abs(self)
    }

    fn square(&self) -> Self {
        Tensor::square(self)
    }

    fn sqrt(&self) -> Self {
        Tensor::sqrt(self)
    }

    fn exp(&self) -> Self {
        Tensor::exp(self)
    }

    fn log(&self) -> Self {
        Tensor::log(self)
    }

    fn sin(&self) -> Self {
        Tensor::sin(self)
    }

    fn cos(&self) -> Self {
        Tensor::cos(self)
    }

    fn tan(&self) -> Self {
        Tensor::tan(self)
    }

    fn asin(&self) -> Self {
        Tensor::asin(self)
    }

    fn acos(&self) -> Self {
        Tensor::acos(self)
    }

    fn atan(&self) -> Self {
        Tensor::atan(self)
    }

    fn sinh(&self) -> Self {
        Tensor::sinh(self)
    }

    fn cosh(&self) -> Self {
        Tensor::cosh(self)
    }

    fn tanh(&self) -> Self {
        Tensor::tanh(self)
    }

    fn asinh(&self) -> Self {
        Tensor::asinh(self)
    }

    fn acosh(&self) -> Self {
        Tensor::acosh(self)
    }

    fn atanh(&self) -> Self {
        Tensor::atanh(self)
    }

    fn lt<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::less_tensor(self, other.borrow())
    }

    fn lt_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::less(self, other)
    }

    fn gt<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::greater_tensor(self, other.borrow())
    }

    fn gt_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::greater(self, other)
    }

    fn le<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::le_tensor(self, other.borrow())
    }

    fn le_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::le(self, other)
    }

    fn ge<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::ge_tensor(self, other.borrow())
    }

    fn ge_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::ge(self, other)
    }

    fn eq<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::eq_tensor(self, other.borrow())
    }

    fn eq_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::eq(self, other)
    }

    fn ne<T: Borrow<Self>>(&self, other: T) -> Self {
        Tensor::ne_tensor(self, other.borrow())
    }

    fn ne_scalar<S: Element>(&self, other: S) -> Self {
        Tensor::ne(self, other)
    }
}