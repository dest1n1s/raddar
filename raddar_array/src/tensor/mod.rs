use std::f64::consts::E;

use crate::AnyNum;

use self::index::{IndexInfo, IndexInfoItem};

pub mod index;
pub mod ops;

#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum TensorKind {
    F32,
    F64,
    I16,
    I32,
    I64,
    OTHER,
}

#[derive(Clone, Copy)]
pub enum ScatterReduction {
    Add,
    Mul,
}

#[derive(Clone, Copy)]
pub enum CmpMode {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
}

pub trait TensorMethods: Sized {
    /// Constructors
    fn empty(shape: &[usize], dtype: TensorKind) -> Self;
    fn zeros(shape: &[usize], dtype: TensorKind) -> Self;
    fn ones(shape: &[usize], dtype: TensorKind) -> Self;
    /// Properties
    fn size(&self) -> Vec<usize>;
    fn kind(&self) -> TensorKind;
    fn item<T: AnyNum>(&self) -> T;
    /// Tensor operations
    fn t(&self) -> Self;
    fn cast(&self, dtype: TensorKind) -> Self;
    /// Arithmetic operations
    fn neg(&self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn pow(&self, other: &Self) -> Self;
    fn abs(&self) -> Self;
    fn sgn(&self) -> Self;
    fn cmp(&self, other: &Self, mode: CmpMode) -> Self;
    fn add_(&mut self, other: &Self);
    fn sub_(&mut self, other: &Self);
    fn mul_(&mut self, other: &Self);
    fn div_(&mut self, other: &Self);
    fn pow_(&mut self, other: &Self);
    fn abs_(&mut self);

    fn add_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn sub_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn mul_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn div_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn pow_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn exp_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn log_scalar<T: AnyNum>(&self, other: T) -> Self;
    fn cmp_scalar<T: AnyNum>(&self, other: T, mode: CmpMode) -> Self;
    fn ln(&self) -> Self {
        self.log_scalar(E)
    }
    fn exp(&self) -> Self {
        self.exp_scalar(E)
    }
    fn add_scalar_<T: AnyNum>(&mut self, other: T);
    fn sub_scalar_<T: AnyNum>(&mut self, other: T);
    fn mul_scalar_<T: AnyNum>(&mut self, other: T);
    fn div_scalar_<T: AnyNum>(&mut self, other: T);
    fn pow_scalar_<T: AnyNum>(&mut self, other: T);
    fn exp_scalar_<T: AnyNum>(&mut self, other: T);
    fn log_scalar_<T: AnyNum>(&mut self, other: T);
    fn ln_(&mut self) {
        self.log_scalar_(E)
    }
    fn exp_(&mut self) {
        self.exp_scalar_(E)
    }

    fn matmul(&self, other: &Self) -> Self;

    /// Assignment operations
    fn assign(&mut self, other: &Self);
    fn assign_scalar<T: AnyNum>(&mut self, other: T);

    /// Advanced arithmetic operations
    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self;
    fn sum(&self) -> Self {
        let dim = (0..self.size().len()).collect::<Vec<_>>();
        self.sum_dim(&dim, false)
    }
    fn mean_dim(&self, dim: &[usize], keep_dim: bool) -> Self;
    fn mean(&self) -> Self {
        let dim = (0..self.size().len()).collect::<Vec<_>>();
        self.mean_dim(&dim, false)
    }
    fn argext_dim(&self, dim: usize, keep_dim: bool, is_max: bool) -> Self;
    fn ext_dim(&self, dim: usize, keep_dim: bool, is_max: bool) -> (Self, Self);
    fn scatter_dim_(&mut self, dim: usize, index: &Self, src: &Self, reduction: ScatterReduction);
    fn unsqueeze(&self, dim: usize) -> Self;
    fn unsqueeze_(&mut self, dim: usize);
    fn squeeze(&self, dim: usize) -> Self;
    fn squeeze_(&mut self, dim: usize);
    fn r#where(&self, cond: &Self, other: &Self) -> Self;
    fn cat(tensors: &[&Self], dim: usize) -> Self;
    fn stack(tensors: &[&Self], dim: usize) -> Self {
        let unsqueezed = tensors.into_iter().map(|t| t.unsqueeze(dim)).collect::<Vec<_>>();
        Self::cat(unsqueezed.iter().collect::<Vec<_>>().as_slice(), dim)
    }
    fn reshape(&self, shape: &[usize]) -> Self;
}

pub trait ArrayMethods: TensorMethods + Sized {
    fn slice(&self, index: IndexInfo) -> Self;

    /// get the element at `index`.
    fn get(&self, index: isize) -> Self {
        self.slice(
            IndexInfo {
                infos: vec![IndexInfoItem::Single(index)],
            }
            .rest_full_for(&self.size()),
        )
    }

    fn permute(&self, permute: &[usize]) -> Self;
    fn broadcast(&self, shape: &[usize]) -> Self;
}

pub trait AutoGradTensorMethods: TensorMethods {
    fn backward(&mut self);
    fn grad(&self) -> Self;
    fn zero_grad(&mut self);
    fn requires_grad(&self) -> bool;
    fn set_requires_grad(&mut self, requires_grad: bool);
}

#[macro_export]
macro_rules! arith_impl {
    ($impl_type:ty) => {
        impl std::ops::Neg for &$impl_type {
            type Output = $impl_type;
            fn neg(self) -> Self::Output {
                TensorMethods::neg(self)
            }
        }

        impl std::ops::Add for &$impl_type {
            type Output = $impl_type;
            fn add(self, other: Self) -> Self::Output {
                TensorMethods::add(self, other)
            }
        }

        impl std::ops::Sub for &$impl_type {
            type Output = $impl_type;
            fn sub(self, other: Self) -> Self::Output {
                TensorMethods::sub(self, other)
            }
        }

        impl std::ops::Mul for &$impl_type {
            type Output = $impl_type;
            fn mul(self, other: Self) -> Self::Output {
                TensorMethods::mul(self, other)
            }
        }

        impl std::ops::Div for &$impl_type {
            type Output = $impl_type;
            fn div(self, other: Self) -> Self::Output {
                TensorMethods::div(self, other)
            }
        }

        impl<T: AnyNum> std::ops::Add<T> for &$impl_type {
            type Output = $impl_type;
            fn add(self, other: T) -> Self::Output {
                TensorMethods::add_scalar(self, other)
            }
        }

        impl<T: AnyNum> std::ops::Sub<T> for &$impl_type {
            type Output = $impl_type;
            fn sub(self, other: T) -> Self::Output {
                TensorMethods::sub_scalar(self, other)
            }
        }

        impl<T: AnyNum> std::ops::Mul<T> for &$impl_type {
            type Output = $impl_type;
            fn mul(self, other: T) -> Self::Output {
                TensorMethods::mul_scalar(self, other)
            }
        }

        impl<T: AnyNum> std::ops::Div<T> for &$impl_type {
            type Output = $impl_type;
            fn div(self, other: T) -> Self::Output {
                TensorMethods::div_scalar(self, other)
            }
        }

        impl std::ops::AddAssign for $impl_type {
            fn add_assign(&mut self, other: Self) {
                self.add_(&other)
            }
        }

        impl std::ops::SubAssign for $impl_type {
            fn sub_assign(&mut self, other: Self) {
                self.sub_(&other)
            }
        }

        impl std::ops::MulAssign for $impl_type {
            fn mul_assign(&mut self, other: Self) {
                self.mul_(&other)
            }
        }

        impl std::ops::DivAssign for $impl_type {
            fn div_assign(&mut self, other: Self) {
                self.div_(&other)
            }
        }

        impl std::ops::AddAssign<&$impl_type> for $impl_type {
            fn add_assign(&mut self, other: &$impl_type) {
                self.add_(other)
            }
        }

        impl std::ops::SubAssign<&$impl_type> for $impl_type {
            fn sub_assign(&mut self, other: &$impl_type) {
                self.sub_(other)
            }
        }

        impl std::ops::MulAssign<&$impl_type> for $impl_type {
            fn mul_assign(&mut self, other: &$impl_type) {
                self.mul_(other)
            }
        }

        impl std::ops::DivAssign<&$impl_type> for $impl_type {
            fn div_assign(&mut self, other: &$impl_type) {
                self.div_(other)
            }
        }

        impl<T: AnyNum> std::ops::AddAssign<T> for $impl_type {
            fn add_assign(&mut self, other: T) {
                self.add_scalar_(other);
            }
        }

        impl<T: AnyNum> std::ops::SubAssign<T> for $impl_type {
            fn sub_assign(&mut self, other: T) {
                self.sub_scalar_(other);
            }
        }

        impl<T: AnyNum> std::ops::MulAssign<T> for $impl_type {
            fn mul_assign(&mut self, other: T) {
                self.mul_scalar_(other);
            }
        }

        impl<T: AnyNum> std::ops::DivAssign<T> for $impl_type {
            fn div_assign(&mut self, other: T) {
                self.div_scalar_(other);
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{
        ndarr::NdArrayTensor,
        tensor::{TensorKind, TensorMethods},
    };

    #[test]
    fn it_works() {
        let mut ts = NdArrayTensor::zeros(&[1, 2, 3, 4, 5], TensorKind::F32);
        ts += 1i8;
        ts *= 2.0f64;
        // Should be all 2
        ts.debug_print();
    }
}
