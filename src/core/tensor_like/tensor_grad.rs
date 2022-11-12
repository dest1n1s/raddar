use std::ops::{Deref, DerefMut};

use tch::Tensor;

use super::TensorLike;

pub struct NoGradGuard<'a, T: TensorGrad> {
    pub tensor: &'a T,
    pub old_grad: bool
}

impl<'a, T: TensorGrad> NoGradGuard<'a, T> {
    pub fn new(tensor: &'a T) -> Self {
        let old_grad = tensor.requires_grad();
        tensor.set_requires_grad(false);
        Self { tensor, old_grad }
    }
}

impl<'a, T: TensorGrad> Drop for NoGradGuard<'a, T> {
    fn drop(&mut self) {
        self.tensor.set_requires_grad(self.old_grad);
    }
}

impl<'a, T: TensorGrad> Deref for NoGradGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.tensor
    }
}

pub struct NoGradGuardMut<'a, T: TensorGrad> {
    pub tensor: &'a mut T,
    pub old_grad: bool
}

impl<'a, T: TensorGrad> NoGradGuardMut<'a, T> {
    pub fn new(tensor: &'a mut T) -> Self {
        let old_grad = tensor.requires_grad();
        tensor.set_requires_grad(false);
        Self { tensor, old_grad }
    }
}

impl<'a, T: TensorGrad> Drop for NoGradGuardMut<'a, T> {
    fn drop(&mut self) {
        self.tensor.set_requires_grad(self.old_grad);
    }
}

impl<'a, T: TensorGrad> Deref for NoGradGuardMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.tensor
    }
}

impl<'a, T: TensorGrad> DerefMut for NoGradGuardMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor
    }
}

/// A trait for tensor-like objects that can perform backpropagation and compute gradients.
pub trait TensorGrad: TensorLike {
    /// Returns the gradient of the tensor.
    fn grad(&self) -> Self;

    /// Clears the gradient of the tensor.
    fn zero_grad(&mut self);

    /// Checks if the tensor requires gradient.
    fn requires_grad(&self) -> bool;

    /// Sets whether the tensor requires gradient.
    fn set_requires_grad(&self, requires_grad: bool) -> Self;

    /// In-place version of `set_requires_grad`.
    fn set_requires_grad_(&mut self, requires_grad: bool) {
        *self = self.set_requires_grad(requires_grad);
    }

    /// Gets a no-grad guard of the tensor that disables gradient computation.
    fn no_grad(&self) -> NoGradGuard<Self> {
        NoGradGuard::new(self)
    }

    /// Gets a mutable no-grad guard of the tensor that disables gradient computation.
    fn no_grad_mut(&mut self) -> NoGradGuardMut<Self> {
        NoGradGuardMut::new(self)
    }

    /// Do backpropagation on the tensor.
    fn backward(&self);
}

impl TensorGrad for Tensor {
    fn grad(&self) -> Self {
        self.grad()
    }

    fn zero_grad(&mut self) {
        self.zero_grad()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad()
    }

    fn set_requires_grad(&self, requires_grad: bool) -> Self {
        self.set_requires_grad(requires_grad)
    }

    fn backward(&self) {
        self.backward()
    }
}