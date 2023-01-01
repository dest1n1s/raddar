use crate::ndarr::Element;

use self::index::{IndexInfo, IndexInfoItem};

pub mod index;
pub mod ops;

pub trait TensorMethods<E: Element>: Sized {
    /// Constructors
    fn zeros(shape: &[usize]) -> Self;
    fn ones(shape: &[usize]) -> Self;
    /// Properties
    fn size(&self) -> Vec<usize>;
    fn item(&self) -> E;
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

    fn add_scalar(&self, other: E) -> Self;
    fn sub_scalar(&self, other: E) -> Self;
    fn mul_scalar(&self, other: E) -> Self;
    fn div_scalar(&self, other: E) -> Self;
    fn add_scalar_(&mut self, other: E);
    fn sub_scalar_(&mut self, other: E);
    fn mul_scalar_(&mut self, other: E);
    fn div_scalar_(&mut self, other: E);

    fn matmul(&self, other: &Self) -> Self;

    /// Assignment operations
    fn assign(&mut self, other: &Self);

    /// Advanced arithmetic operations
    fn sum_dim(&self, dim: &[usize], keep_dim: bool) -> Self;
    fn unsqueeze(&self, dim: usize) -> Self;
    fn unsqueeze_(&mut self, dim: usize);
    fn squeeze(&self, dim: usize) -> Self;
    fn squeeze_(&mut self, dim: usize);
}

pub trait ArrayMethods<E: Element>: TensorMethods<E> + Sized {
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

pub trait AutoGradTensorMethods<E: Element>: TensorMethods<E> {
    fn backward(&mut self);
    fn grad(&self) -> Self;
    fn zero_grad(&mut self);
    fn requires_grad(&self) -> bool;
    fn set_requires_grad(&mut self, requires_grad: bool);
}

#[macro_export]
macro_rules! arith_impl {
    ($impl_type:ty, $generics:tt) => {
        impl<$generics: Element> std::ops::Neg for &$impl_type {
            type Output = $impl_type;
            fn neg(self) -> Self::Output {
                TensorMethods::neg(self)
            }
        }

        impl<$generics: Element> std::ops::Add for &$impl_type {
            type Output = $impl_type;
            fn add(self, other: Self) -> Self::Output {
                TensorMethods::add(self, other)
            }
        }

        impl<$generics: Element> std::ops::Sub for &$impl_type {
            type Output = $impl_type;
            fn sub(self, other: Self) -> Self::Output {
                TensorMethods::sub(self, other)
            }
        }

        impl<$generics: Element> std::ops::Mul for &$impl_type {
            type Output = $impl_type;
            fn mul(self, other: Self) -> Self::Output {
                TensorMethods::mul(self, other)
            }
        }

        impl<$generics: Element> std::ops::Div for &$impl_type {
            type Output = $impl_type;
            fn div(self, other: Self) -> Self::Output {
                TensorMethods::div(self, other)
            }
        }

        impl<$generics: Element> std::ops::Add<E> for &$impl_type {
            type Output = $impl_type;
            fn add(self, other: E) -> Self::Output {
                TensorMethods::add_scalar(self, other)
            }
        }

        impl<$generics: Element> std::ops::Sub<E> for &$impl_type {
            type Output = $impl_type;
            fn sub(self, other: E) -> Self::Output {
                TensorMethods::sub_scalar(self, other)
            }
        }

        impl<$generics: Element> std::ops::Mul<E> for &$impl_type {
            type Output = $impl_type;
            fn mul(self, other: E) -> Self::Output {
                TensorMethods::mul_scalar(self, other)
            }
        }

        impl<$generics: Element> std::ops::Div<E> for &$impl_type {
            type Output = $impl_type;
            fn div(self, other: E) -> Self::Output {
                TensorMethods::div_scalar(self, other)
            }
        }

        impl<$generics: Element> std::ops::AddAssign for $impl_type {
            fn add_assign(&mut self, other: Self) {
                self.add_(&other)
            }
        }

        impl<$generics: Element> std::ops::SubAssign for $impl_type {
            fn sub_assign(&mut self, other: Self) {
                self.sub_(&other)
            }
        }

        impl<$generics: Element> std::ops::MulAssign for $impl_type {
            fn mul_assign(&mut self, other: Self) {
                self.mul_(&other)
            }
        }

        impl<$generics: Element> std::ops::DivAssign for $impl_type {
            fn div_assign(&mut self, other: Self) {
                self.div_(&other)
            }
        }

        impl<$generics: Element> std::ops::AddAssign<&$impl_type> for $impl_type {
            fn add_assign(&mut self, other: &$impl_type) {
                self.add_(other)
            }
        }

        impl<$generics: Element> std::ops::SubAssign<&$impl_type> for $impl_type {
            fn sub_assign(&mut self, other: &$impl_type) {
                self.sub_(other)
            }
        }

        impl<$generics: Element> std::ops::MulAssign<&$impl_type> for $impl_type {
            fn mul_assign(&mut self, other: &$impl_type) {
                self.mul_(other)
            }
        }

        impl<$generics: Element> std::ops::DivAssign<&$impl_type> for $impl_type {
            fn div_assign(&mut self, other: &$impl_type) {
                self.div_(other)
            }
        }
        
        impl<$generics: Element> std::ops::AddAssign<E> for $impl_type {
            fn add_assign(&mut self, other: E) {
                self.add_scalar_(other);
            }
        }

        impl<$generics: Element> std::ops::SubAssign<E> for $impl_type {
            fn sub_assign(&mut self, other: E) {
                self.sub_scalar_(other);
            }
        }

        impl<$generics: Element> std::ops::MulAssign<E> for $impl_type {
            fn mul_assign(&mut self, other: E) {
                self.mul_scalar_(other);
            }
        }

        impl<$generics: Element> std::ops::DivAssign<E> for $impl_type {
            fn div_assign(&mut self, other: E) {
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
        let mut ts = NdArrayTensor::zeros(&[1, 2, 3, 4, 5]);
        ts += 1i8;
        ts *= 2.0f64;
        // Should be all 2
        ts.debug_print();
    }
}