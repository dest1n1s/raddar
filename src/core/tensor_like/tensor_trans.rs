use tch::Tensor;

use super::{TensorOps, TensorTrigon};

/// A trait for a tensor-like object that can perform basic algebraic operations on self.
/// 
/// These operations are largely similar to the operations in [TensorOps], but they are performed on self instead of another tensor, so they are more efficient and do not require to be put in a computational graph.
/// 
/// This trait will provide a default implementation for all the operations, but it's highly recommended to override them if you want to get better performance.
pub trait TensorOpsInplace: TensorOps {
    /// In-place version of [TensorOps::transpose].
    fn transpose_(&mut self, dim0: i64, dim1: i64) {
        *self = self.transpose(dim0, dim1);
    }

    /// In-place version of [TensorOps::abs].
    fn abs_(&mut self) {
        *self = self.abs();
    }

    /// In-place version of [TensorOps::square].
    fn square_(&mut self) {
        *self = self.square();
    }

    /// In-place version of [TensorOps::sqrt].
    fn sqrt_(&mut self) {
        *self = self.sqrt();
    }

    /// In-place version of [TensorOps::exp].
    fn exp_(&mut self) {
        *self = self.exp();
    }

    /// In-place version of [TensorOps::log].
    fn log_(&mut self) {
        *self = self.log();
    }
}

impl TensorOpsInplace for Tensor {
    fn transpose_(&mut self, dim0: i64, dim1: i64) {
        let _ = self.transpose_(dim0, dim1);
    }

    fn abs_(&mut self) {
        let _ = self.abs_();
    }

    fn square_(&mut self) {
        let _ = self.square_();
    }

    fn sqrt_(&mut self) {
        let _ = self.sqrt_();
    }

    fn exp_(&mut self) {
        let _ = self.exp_();
    }

    fn log_(&mut self) {
        let _ = self.log_();
    }
}

pub trait TensorTrigonInplace: TensorTrigon {
    /// In-place version of [TensorOps::sin].
    fn sin_(&mut self) {
        *self = self.sin();
    }

    /// In-place version of [TensorOps::cos].
    fn cos_(&mut self) {
        *self = self.cos();
    }

    /// In-place version of [TensorOps::tan].
    fn tan_(&mut self) {
        *self = self.tan();
    }

    /// In-place version of [TensorOps::asin].
    fn asin_(&mut self) {
        *self = self.asin();
    }

    /// In-place version of [TensorOps::acos].
    fn acos_(&mut self) {
        *self = self.acos();
    }

    /// In-place version of [TensorOps::atan].
    fn atan_(&mut self) {
        *self = self.atan();
    }

    /// In-place version of [TensorOps::sinh].
    fn sinh_(&mut self) {
        *self = self.sinh();
    }

    /// In-place version of [TensorOps::cosh].
    fn cosh_(&mut self) {
        *self = self.cosh();
    }

    /// In-place version of [TensorOps::tanh].
    fn tanh_(&mut self) {
        *self = self.tanh();
    }

    /// In-place version of [TensorOps::asinh].
    fn asinh_(&mut self) {
        *self = self.asinh();
    }

    /// In-place version of [TensorOps::acosh].
    fn acosh_(&mut self) {
        *self = self.acosh();
    }

    /// In-place version of [TensorOps::atanh].
    fn atanh_(&mut self) {
        *self = self.atanh();
    }
}

impl TensorTrigonInplace for Tensor {
    fn sin_(&mut self) {
        let _ = self.sin_();
    }

    fn cos_(&mut self) {
        let _ = self.cos_();
    }

    fn tan_(&mut self) {
        let _ = self.tan_();
    }

    fn asin_(&mut self) {
        let _ = self.asin_();
    }

    fn acos_(&mut self) {
        let _ = self.acos_();
    }

    fn atan_(&mut self) {
        let _ = self.atan_();
    }

    fn sinh_(&mut self) {
        let _ = self.sinh_();
    }

    fn cosh_(&mut self) {
        let _ = self.cosh_();
    }

    fn tanh_(&mut self) {
        let _ = self.tanh_();
    }

    fn asinh_(&mut self) {
        let _ = self.asinh_();
    }

    fn acosh_(&mut self) {
        let _ = self.acosh_();
    }

    fn atanh_(&mut self) {
        let _ = self.atanh_();
    }
}