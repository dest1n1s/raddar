use num::cast::NumCast;

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
pub trait TensorMethods: Sized {
    /// Constructors
    fn zeros(shape: &[usize], dtype: TensorKind) -> Self;
    fn ones(shape: &[usize], dtype: TensorKind) -> Self;
    /// Properties
    fn size(&self) -> Vec<usize>;
    fn kind(&self) -> TensorKind;
    fn item<T: NumCast + Copy + 'static>(&self) -> T;
    /// Tensor operations
    fn t(&self) -> Self;
    /// Arithmetic operations
    fn neg(&self) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn add_(&mut self, other: &Self);
    fn sub_(&mut self, other: &Self);
    fn mul_(&mut self, other: &Self);
    fn div_(&mut self, other: &Self);

    fn add_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self;
    fn sub_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self;
    fn mul_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self;
    fn div_scalar<T: NumCast + Copy + 'static>(&self, other: T) -> Self;
    fn add_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn sub_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn mul_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);
    fn div_scalar_<T: NumCast + Copy + 'static>(&mut self, other: T);

    fn matmul(&self, other: &Self) -> Self;

    /// Assignment operations
    fn assign(&mut self, other: &Self);

    /// Advanced arithmetic operations
    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self;
    fn sum(&self) -> Self {
        self.sum_dim(self.size().as_slice(), false)
    }
    fn unsqueeze(&self, dim: usize) -> Self;
    fn unsqueeze_(&mut self, dim: usize);
    fn squeeze(&self, dim: usize) -> Self;
    fn squeeze_(&mut self, dim: usize);
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

        impl<T: num::NumCast + Copy + 'static> std::ops::Add<T> for &$impl_type {
            type Output = $impl_type;
            fn add(self, other: T) -> Self::Output {
                TensorMethods::add_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Sub<T> for &$impl_type {
            type Output = $impl_type;
            fn sub(self, other: T) -> Self::Output {
                TensorMethods::sub_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Mul<T> for &$impl_type {
            type Output = $impl_type;
            fn mul(self, other: T) -> Self::Output {
                TensorMethods::mul_scalar(self, other)
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::Div<T> for &$impl_type {
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

        impl<T: num::NumCast + Copy + 'static> std::ops::AddAssign<T> for $impl_type {
            fn add_assign(&mut self, other: T) {
                self.add_scalar_(other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::SubAssign<T> for $impl_type {
            fn sub_assign(&mut self, other: T) {
                self.sub_scalar_(other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::MulAssign<T> for $impl_type {
            fn mul_assign(&mut self, other: T) {
                self.mul_scalar_(other);
            }
        }

        impl<T: num::NumCast + Copy + 'static> std::ops::DivAssign<T> for $impl_type {
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
