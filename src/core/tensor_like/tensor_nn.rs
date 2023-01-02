use std::{borrow::Borrow, f64::consts::PI};

use tch::Tensor;

use crate::core::TensorGrad;

use super::{TensorOps, TensorStatistics, TensorTrigon};

pub enum Reduction {
    None,
    Mean,
    Sum,
}

impl From<Reduction> for tch::Reduction {
    fn from(reduction: Reduction) -> Self {
        match reduction {
            Reduction::None => tch::Reduction::None,
            Reduction::Mean => tch::Reduction::Mean,
            Reduction::Sum => tch::Reduction::Sum,
        }
    }
}

/// A trait for tensor-like objects that can perform neural network related operations, such as convolution, pooling, some activation functions and loss functions, etc.
///
/// Some of these operations may not strongly require extra low-level performance optimization (using gpu computing, cpu optimization or maybe just some parallel operations), so this trait gives a default implementation for them. But you can still override them if you want to.
pub trait TensorNN: TensorOps + TensorGrad {
    /// Computes the 1D convolution of the input tensor with the kernel.
    ///
    /// The input tensor should be a 3D tensor of shape `(batch_size, channels, length)`.
    ///
    /// The kernel should be a 3D tensor of shape `(out_channels, channels / groups, kernel_size)`.
    ///
    /// The `bias` parameter is optional, and should be a 1D tensor of shape (out_channels).
    ///
    /// The output tensor will be a 3D tensor of shape (batch_size, out_channels, out_length), or a 4D tensor of shape (batch_size, out_channels, out_height, out_width).
    ///
    /// The `stride` parameter specifies the stride of the convolution.
    ///
    /// The `padding` parameter specifies the padding of the convolution.
    ///
    /// The `dilation` parameter specifies the dilation of the convolution.
    ///
    /// The `groups` parameter specifies the number of blocked connections from input channels to output channels.
    fn conv1d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self;

    /// Computes the 2D convolution of the input tensor with the kernel.
    ///
    /// The input tensor should be a 4D tensor of shape `(batch_size, channels, height, width)`.
    ///
    /// The kernel should be a 4D tensor of shape `(out_channels, channels / groups, kernel_height, kernel_width)`.
    ///
    /// The `bias` parameter is optional, and should be a 1D tensor of shape (out_channels).
    ///
    /// The output tensor will be a 4D tensor of shape (batch_size, out_channels, out_height, out_width).
    ///
    /// The `stride` parameter specifies the stride of the convolution.
    ///
    /// The `padding` parameter specifies the padding of the convolution.
    ///
    /// The `dilation` parameter specifies the dilation of the convolution.
    ///
    /// The `groups` parameter specifies the number of blocked connections from input channels to output channels.
    fn conv2d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self;

    /// Computes the 3D convolution of the input tensor with the kernel.
    ///
    /// The input tensor should be a 5D tensor of shape `(batch_size, channels, depth, height, width)`.
    ///
    /// The kernel should be a 5D tensor of shape `(out_channels, channels / groups, kernel_depth, kernel_height, kernel_width)`.
    ///
    /// The `bias` parameter is optional, and should be a 1D tensor of shape (out_channels).
    ///
    /// The output tensor will be a 5D tensor of shape (batch_size, out_channels, out_depth, out_height, out_width).
    ///
    /// The `stride` parameter specifies the stride of the convolution.
    ///
    /// The `padding` parameter specifies the padding of the convolution.
    ///
    /// The `dilation` parameter specifies the dilation of the convolution.
    ///
    /// The `groups` parameter specifies the number of blocked connections from input channels to output channels.
    fn conv3d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self;

    /// Computes the 1D max pooling of the input tensor.
    ///
    /// The input tensor should be a 3D tensor of shape `(batch_size, channels, length)`.
    ///
    /// The output tensor will be a 3D tensor of shape (batch_size, channels, out_length).
    ///
    /// The `kernel_size` parameter specifies the size of the max pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the max pooling.
    ///
    /// The `padding` parameter specifies the padding of the max pooling.
    ///
    /// The `dilation` parameter specifies the dilation of the max pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    fn max_pool1d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self;

    /// Computes the 2D max pooling of the input tensor.
    ///
    /// The input tensor should be a 4D tensor of shape `(batch_size, channels, height, width)`.
    ///
    /// The output tensor will be a 4D tensor of shape (batch_size, channels, out_height, out_width).
    ///
    /// The `kernel_size` parameter specifies the size of the max pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the max pooling.
    ///
    /// The `padding` parameter specifies the padding of the max pooling.
    ///
    /// The `dilation` parameter specifies the dilation of the max pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    fn max_pool2d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self;

    /// Computes the 3D max pooling of the input tensor.
    ///
    /// The input tensor should be a 5D tensor of shape `(batch_size, channels, depth, height, width)`.
    ///
    /// The output tensor will be a 5D tensor of shape (batch_size, channels, out_depth, out_height, out_width).
    ///
    /// The `kernel_size` parameter specifies the size of the max pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the max pooling.
    ///
    /// The `padding` parameter specifies the padding of the max pooling.
    ///
    /// The `dilation` parameter specifies the dilation of the max pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    fn max_pool3d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self;

    /// Computes the 1D average pooling of the input tensor.
    ///
    /// The input tensor should be a 3D tensor of shape `(batch_size, channels, length)`.
    ///
    /// The output tensor will be a 3D tensor of shape (batch_size, channels, out_length).
    ///
    /// The `kernel_size` parameter specifies the size of the average pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the average pooling.
    ///
    /// The `padding` parameter specifies the padding of the average pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    ///
    /// The `count_include_pad` parameter specifies whether to include padding in the average computation.
    fn avg_pool1d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self;

    /// Computes the 2D average pooling of the input tensor.
    ///
    /// The input tensor should be a 4D tensor of shape `(batch_size, channels, height, width)`.
    ///
    /// The output tensor will be a 4D tensor of shape (batch_size, channels, out_height, out_width).
    ///
    /// The `kernel_size` parameter specifies the size of the average pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the average pooling.
    ///
    /// The `padding` parameter specifies the padding of the average pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    ///
    /// The `count_include_pad` parameter specifies whether to include padding in the average computation.
    ///
    /// The `divisor_override` parameter specifies the divisor to use in the average computation.
    fn avg_pool2d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Option<i64>,
    ) -> Self;

    /// Computes the 3D average pooling of the input tensor.
    ///
    /// The input tensor should be a 5D tensor of shape `(batch_size, channels, depth, height, width)`.
    ///
    /// The output tensor will be a 5D tensor of shape (batch_size, channels, out_depth, out_height, out_width).
    ///
    /// The `kernel_size` parameter specifies the size of the average pooling kernel.
    ///
    /// The `stride` parameter specifies the stride of the average pooling.
    ///
    /// The `padding` parameter specifies the padding of the average pooling.
    ///
    /// The `ceil_mode` parameter specifies whether to use ceil or floor to compute the output shape.
    ///
    /// The `count_include_pad` parameter specifies whether to include padding in the average computation.
    ///
    /// The `divisor_override` parameter specifies the divisor to use in the average computation.
    fn avg_pool3d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Option<i64>,
    ) -> Self;

    /// Computes the 1D adaptive max pooling of the input tensor.
    ///
    /// The input tensor should be a 3D tensor of shape `(batch_size, channels, length)`.
    ///
    /// The output tensor will be a 3D tensor of shape (batch_size, channels, output_size).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the length dimension.
    fn adaptive_max_pool1d(&self, output_size: &[i64]) -> (Self, Self);

    /// Computes the 2D adaptive max pooling of the input tensor.
    ///
    /// The input tensor should be a 4D tensor of shape `(batch_size, channels, height, width)`.
    ///
    /// The output tensor will be a 4D tensor of shape (batch_size, channels, output_size[0], output_size[1]).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the height and width dimensions.
    fn adaptive_max_pool2d(&self, output_size: &[i64]) -> (Self, Self);

    /// Computes the 3D adaptive max pooling of the input tensor.
    ///
    /// The input tensor should be a 5D tensor of shape `(batch_size, channels, depth, height, width)`.
    ///
    /// The output tensor will be a 5D tensor of shape (batch_size, channels, output_size[0], output_size[1], output_size[2]).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the depth, height and width dimensions.
    fn adaptive_max_pool3d(&self, output_size: &[i64]) -> (Self, Self);

    /// Computes the 1D adaptive average pooling of the input tensor.
    ///
    /// The input tensor should be a 3D tensor of shape `(batch_size, channels, length)`.
    ///
    /// The output tensor will be a 3D tensor of shape (batch_size, channels, output_size).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the length dimension.
    fn adaptive_avg_pool1d(&self, output_size: &[i64]) -> Self;

    /// Computes the 2D adaptive average pooling of the input tensor.
    ///
    /// The input tensor should be a 4D tensor of shape `(batch_size, channels, height, width)`.
    ///
    /// The output tensor will be a 4D tensor of shape (batch_size, channels, output_size[0], output_size[1]).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the height and width dimensions.
    fn adaptive_avg_pool2d(&self, output_size: &[i64]) -> Self;

    /// Computes the 3D adaptive average pooling of the input tensor.
    ///
    /// The input tensor should be a 5D tensor of shape `(batch_size, channels, depth, height, width)`.
    ///
    /// The output tensor will be a 5D tensor of shape (batch_size, channels, output_size[0], output_size[1], output_size[2]).
    ///
    /// The `output_size` parameter specifies the size of the output tensor in the depth, height and width dimensions.
    fn adaptive_avg_pool3d(&self, output_size: &[i64]) -> Self;

    /// Computes ReLU of the input tensor.
    fn relu(&self) -> Self {
        self.r#where(&self.gt_scalar(0.0), &Self::zeros(&[], self.env()))
    }

    /// Computes LeakyReLU of the input tensor.
    fn leaky_relu(&self, negative_slope: f64) -> Self {
        self.r#where(&self.gt_scalar(0.0), self * negative_slope)
    }

    /// Computes GELU of the input tensor.
    fn gelu(&self) -> Self;

    /// Computes softmax of the input tensor.
    fn softmax(&self, dim: i64) -> Self {
        let exp = self.exp();
        let sum = exp.sum_dim(&[dim], true);
        exp / sum
    }

    /// Computes log softmax of the input tensor.
    fn log_softmax(&self, dim: i64) -> Self {
        self.softmax(dim).log()
    }

    /// Computes cross entropy loss between the input tensor and the target tensor
    ///
    /// The input tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    /// The target tensor should be a 1D tensor of shape `(batch_size)`.
    ///
    /// The output tensor will be a 1D tensor of shape `(batch_size)`.
    ///
    /// The `reduction` parameter specifies the reduction to apply to the output:
    /// - `Reduction::None` means no reduction will be applied.
    /// - `Reduction::Mean` means the output will be the mean of the output over the batch.
    /// - `Reduction::Sum` means the output will be the sum of the output over the batch.
    ///
    /// The `ignore_index` parameter specifies the index to ignore when computing the loss.
    ///
    /// The `weight` parameter specifies the weight to apply to each class.
    ///
    /// The `label_smoothing` parameter specifies the amount of label smoothing to apply.
    fn cross_entropy_loss<T: Borrow<Self>, U: Borrow<Self>>(
        &self,
        target: T,
        weight: Option<U>,
        reduction: Reduction,
        ignore_index: i64,
        label_smoothing: f64,
    ) -> Self;

    /// Computes binary cross entropy loss between the input tensor and the target tensor
    ///
    /// The input tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    /// The target tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    ///
    /// The output tensor will be a 1D tensor of shape `(batch_size)`.
    ///
    /// The `reduction` parameter specifies the reduction to apply to the output:
    /// - `Reduction::None` means no reduction will be applied.
    /// - `Reduction::Mean` means the output will be the mean of the output over the batch.
    /// - `Reduction::Sum` means the output will be the sum of the output over the batch.
    ///
    /// The `weight` parameter specifies the weight to apply to each class.
    fn binary_cross_entropy_loss<T: Borrow<Self>, U: Borrow<Self>, V: Borrow<Self>>(
        &self,
        target: T,
        weight: Option<U>,
        reduction: Reduction,
    ) -> Self;

    /// Computes mean squared error loss between the input tensor and the target tensor
    ///
    /// The input tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    /// The target tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    ///
    /// The output tensor will be a 1D tensor of shape `(batch_size)`.
    ///
    /// The `reduction` parameter specifies the reduction to apply to the output:
    /// - `Reduction::None` means no reduction will be applied.
    /// - `Reduction::Mean` means the output will be the mean of the output over the batch.
    /// - `Reduction::Sum` means the output will be the sum of the output over the batch.
    fn mse_loss<T: Borrow<Self>, U: Borrow<Self>>(&self, target: T, reduction: Reduction) -> Self;

    /// During training, randomly zeroes some of the elements of the input tensor with probability `p`.
    ///
    /// The input tensor should be a 2D tensor of shape `(batch_size, num_classes)`.
    ///
    /// The output tensor will be a 2D tensor of shape `(batch_size, num_classes)`.
    ///
    /// The `p` parameter specifies the probability of an element to be zeroed.
    fn dropout(&self, p: f64, train: bool) -> Self;

    /// Computes the batch norm of the input tensor.
    ///
    /// The input tensor should be a tensor of shape `(batch_size, num_classes, *)`, where `*` means any number of additional dimensions.
    ///
    /// The output tensor will be a tensor of shape `(batch_size, num_classes, *)`, where `*` equals the additional dimensions of the input tensor.
    ///
    /// The `weight` parameter specifies the weight to apply to the normalized input.
    ///
    /// The `bias` parameter specifies the bias to apply to the normalized input.
    ///
    /// The `running_mean` parameter specifies the running mean of the input.
    ///
    /// The `running_var` parameter specifies the running variance of the input.
    ///
    /// The `training` parameter specifies whether the input is in training mode.
    ///
    /// The `momentum` parameter specifies the momentum to use for the running mean and variance.
    ///
    /// The `eps` parameter specifies the epsilon to use for the running mean and variance.
    ///
    /// The `cudnn_enabled` parameter specifies whether to use cuDNN for the batch norm.
    fn batch_norm<T: Borrow<Self>, U: Borrow<Self>, V: Borrow<Self>, W: Borrow<Self>>(
        &self,
        weight: Option<T>,
        bias: Option<U>,
        running_mean: Option<V>,
        running_var: Option<W>,
        training: bool,
        momentum: f64,
        eps: f64,
        cudnn_enabled: bool,
    ) -> Self;

    /// Computes the layer norm of the input tensor.
    ///
    /// The input tensor should be a tensor of shape `(batch_size, num_classes, *)`, where `*` means any number of additional dimensions.
    ///
    /// The output tensor will be a tensor of shape `(batch_size, num_classes, *)`, where `*` equals the additional dimensions of the input tensor.
    ///
    /// The `normalized_shape` parameter specifies the shape of the normalized input.
    ///
    /// The `weight` parameter specifies the weight to apply to the normalized input.
    ///
    /// The `bias` parameter specifies the bias to apply to the normalized input.
    ///
    /// The `eps` parameter specifies the epsilon to use for the normalization.
    ///
    /// The `cudnn_enabled` parameter specifies whether to use cuDNN for the layer norm.
    fn layer_norm<T: Borrow<Self>, U: Borrow<Self>>(
        &self,
        normalized_shape: &[i64],
        weight: Option<T>,
        bias: Option<U>,
        eps: f64,
        cudnn_enabled: bool,
    ) -> Self;
}

default impl<T: TensorStatistics + TensorTrigon + TensorGrad> TensorNN for T {
    /// A default implementation of `dropout` that uses `bernoulli`.
    fn dropout(&self, p: f64, train: bool) -> Self {
        if train {
            let mask = self.ones_like() * (1.0 - p);
            self * mask.bernoulli()
        } else {
            self.copy()
        }
    }

    fn gelu(&self) -> Self  {
        let z = (self + &self.pow_scalar(3) * 0.044715) * (2.0f64 / PI).sqrt();
        0.5 * self * (1 + z.tanh())
    }
}

impl TensorNN for Tensor {
    fn conv1d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        self.conv1d(kernel.borrow(), bias, stride, padding, dilation, groups)
    }

    fn conv2d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        self.conv2d(kernel.borrow(), bias, stride, padding, dilation, groups)
    }

    fn conv3d(
        &self,
        kernel: impl Borrow<Self>,
        bias: Option<impl Borrow<Self>>,
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        groups: i64,
    ) -> Self {
        self.conv3d(kernel.borrow(), bias, stride, padding, dilation, groups)
    }

    fn max_pool1d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self {
        self.max_pool1d(kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool2d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self {
        self.max_pool2d(kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn max_pool3d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        dilation: &[i64],
        ceil_mode: bool,
    ) -> Self {
        self.max_pool3d(kernel_size, stride, padding, dilation, ceil_mode)
    }

    fn avg_pool1d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self {
        self.avg_pool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)
    }

    fn avg_pool2d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Option<i64>,
    ) -> Self {
        self.avg_pool2d(
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    }

    fn avg_pool3d(
        &self,
        kernel_size: &[i64],
        stride: &[i64],
        padding: &[i64],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Option<i64>,
    ) -> Self {
        self.avg_pool3d(
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    }

    fn adaptive_max_pool1d(&self, output_size: &[i64]) -> (Self, Self) {
        self.adaptive_max_pool1d(output_size)
    }

    fn adaptive_max_pool2d(&self, output_size: &[i64]) -> (Self, Self) {
        self.adaptive_max_pool2d(output_size)
    }

    fn adaptive_max_pool3d(&self, output_size: &[i64]) -> (Self, Self) {
        self.adaptive_max_pool3d(output_size)
    }

    fn adaptive_avg_pool1d(&self, output_size: &[i64]) -> Self {
        self.adaptive_avg_pool1d(output_size)
    }

    fn adaptive_avg_pool2d(&self, output_size: &[i64]) -> Self {
        self.adaptive_avg_pool2d(output_size)
    }

    fn adaptive_avg_pool3d(&self, output_size: &[i64]) -> Self {
        self.adaptive_avg_pool3d(output_size)
    }

    fn relu(&self) -> Self {
        self.relu()
    }

    fn gelu(&self) -> Self {
        self.gelu("none")
    }

    fn softmax(&self, dim: i64) -> Self {
        self.softmax(dim, self.kind())
    }

    fn log_softmax(&self, dim: i64) -> Self {
        self.log_softmax(dim, self.kind())
    }

    fn cross_entropy_loss<T: Borrow<Self>, U: Borrow<Self>>(
        &self,
        target: T,
        weight: Option<U>,
        reduction: Reduction,
        ignore_index: i64,
        label_smoothing: f64,
    ) -> Self {
        self.cross_entropy_loss(
            target.borrow(),
            weight,
            reduction.into(),
            ignore_index,
            label_smoothing,
        )
    }

    fn binary_cross_entropy_loss<T: Borrow<Self>, U: Borrow<Self>, V: Borrow<Self>>(
        &self,
        target: T,
        weight: Option<U>,
        reduction: Reduction,
    ) -> Self {
        self.binary_cross_entropy(target.borrow(), weight, reduction.into())
    }

    fn mse_loss<T: Borrow<Self>, U: Borrow<Self>>(&self, target: T, reduction: Reduction) -> Self {
        self.mse_loss(target.borrow(), reduction.into())
    }

    fn dropout(&self, p: f64, train: bool) -> Self {
        self.dropout(p, train)
    }

    fn batch_norm<T: Borrow<Self>, U: Borrow<Self>, V: Borrow<Self>, W: Borrow<Self>>(
        &self,
        weight: Option<T>,
        bias: Option<U>,
        running_mean: Option<V>,
        running_var: Option<W>,
        training: bool,
        momentum: f64,
        eps: f64,
        cudnn_enabled: bool,
    ) -> Self {
        self.batch_norm(
            weight.as_ref().map(|w| w.borrow()),
            bias.as_ref().map(|w| w.borrow()),
            running_mean.as_ref().map(|w| w.borrow()),
            running_var.as_ref().map(|w| w.borrow()),
            training,
            momentum,
            eps,
            cudnn_enabled,
        )
    }

    fn layer_norm<T: Borrow<Self>, U: Borrow<Self>>(
        &self,
        normalized_shape: &[i64],
        weight: Option<T>,
        bias: Option<U>,
        eps: f64,
        cudnn_enabled: bool,
    ) -> Self {
        self.layer_norm(
            normalized_shape,
            weight.as_ref().map(|w| w.borrow()),
            bias.as_ref().map(|w| w.borrow()),
            eps,
            cudnn_enabled,
        )
    }
}
