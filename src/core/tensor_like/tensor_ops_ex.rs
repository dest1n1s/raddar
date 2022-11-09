use tch::Tensor;

use super::TensorOps;

/// A trait for a tensor-like object that can perform advanced mathematical operations.
///
/// Advanced mathematical operations include:
/// - Statistical operations
/// - Advanced Linear Algebra operations
/// - Other mathematical operations not included in [TensorOps]
///
/// Some of these operations may not strongly require extra low-level performance optimization (using gpu computing, cpu optimization or maybe just some parallel operations), so this trait gives a default implementation for them. But you can still override them if you want to.
pub trait TensorOpsEx: TensorOps {
    /// Draws binary random numbers (0 or 1) from a Bernoulli distribution.
    ///
    /// The input tensor should be a tensor of probabilities, and the output tensor will be a tensor of 0s and 1s.
    fn bernoulli(&self) -> Self;

    /// Computes the Cholesky decomposition of the symmetric positive-definite matrices.
    ///
    /// The input tensor should be a symmetric positive-definite matrix, or a batch of symmetric positive-definite matrices.
    ///
    /// The output tensor will be the lower triangular matrix of the Cholesky decomposition if `upper` is false, or the upper triangular matrix of the Cholesky decomposition if `upper` is true.
    fn cholesky(&self, upper: bool) -> Self;

    /// Computes the determinant of the square matrix or matrices.
    /// 
    /// The input tensor should be a square matrix, or a batch of square matrices.
    fn det(&self) -> Self;

    /// Computes the eigenvalues and eigenvectors of a square matrix or matrices.
    /// 
    /// The input tensor should be a square matrix, or a batch of square matrices.
    /// 
    /// The output tensor will be a tuple of two tensors. The first tensor will be the eigenvalues, and the second tensor will be the eigenvectors.
    fn eig(&self) -> (Self, Self);
}

impl TensorOpsEx for Tensor {
    fn bernoulli(&self) -> Self {
        self.bernoulli()
    }

    fn cholesky(&self, upper: bool) -> Self {
        self.cholesky(upper)
    }

    fn det(&self) -> Self {
        self.det()
    }

    fn eig(&self) -> (Self, Self) {
        self.eig(true)
    }
}
