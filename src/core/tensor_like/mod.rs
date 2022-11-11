pub mod tensor_ops;
pub mod tensor_ops_ex;
pub mod tensor_trans;
pub mod tensor_nn;
pub mod tensor_grad;

use std::{
    borrow::Borrow,
    fmt::Debug,
    path::Path,
};

use tch::{Device, Kind, Tensor, Scalar};

pub use tensor_ops::*;
pub use tensor_ops_ex::*;
pub use tensor_trans::*;
pub use tensor_nn::*;
pub use tensor_grad::*;

pub trait Element: Clone + tch::kind::Element + Into<Scalar> {}

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
/// - It should be associated with some env variable, which for example may be used to determine the device and data type of the tensor-like object, and a size indicating the length of each dimension.
/// - It should be able to be constructed with some factors, or other tensor-like objects.
/// - It should also be able to perform some other operations, such as reshaping, transposing, and slicing.
///
/// Of course, a real tensor-like object should also be able to perform some tensor operations, such as arithmetic operations, matrix multiplication, gradient calculation, and so on. However, this trait just focuses on the properties of tensor-like objects, and does not require further functions.
pub trait TensorLike: PartialEq<Self> + AsRef<Self> + Debug + Default {
    /// The type of data kind.
    type Kind;

    /// The type of device.
    type Device;

    /// The type of the env variable. We assume that the env variable is at least consists of the data kind and the device, and may also contain other information.
    type Env: Default = (Kind, Device);

    /// Get the env variable.
    fn env(&self) -> Self::Env;

    /// Convert the tensor-like object to a specific kind.
    fn to_kind(&self, kind: Self::Kind) -> Self;

    /// Convert the tensor-like object to a specific device.
    fn to_device(&self, device: Self::Device) -> Self;

    /// Convert the tensor-like object to a specific env.
    fn to_env(&self, env: Self::Env) -> Self;

    /// Get the shape of the tensor-like object.
    fn shape(&self) -> Vec<i64>;

    /// Create a tensor-like object from a slice.
    fn of_slice<T: Element>(data: &[T]) -> Self;

    /// Create a 2-D tensor-like object from a slice of slices.
    fn of_slice2<T: Element, U: AsRef<[T]>>(v: &[U]) -> Self
    {
        let inner: Vec<Self> = v.iter().map(|v| Self::of_slice(v.as_ref())).collect();
        Self::stack(&inner, 0)
    }

    /// Stack a list of tensor-like objects into a tensor-like object.
    fn stack<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self;

    /// Concatenate a list of tensor-like objects into a tensor-like object.
    fn cat<T: Borrow<Self>>(tensors: &[T], dim: i64) -> Self;

    /// Create a tensor-like object filled with zeros.
    fn zeros(size: &[i64], env: Self::Env) -> Self;

    /// Create a tensor-like object filled with ones.
    fn ones(size: &[i64], env: Self::Env) -> Self;

    /// Create a tensor-like object filled with uninitialized values.
    fn empty(size: &[i64], env: Self::Env) -> Self;

    /// Create a tensor-like object with the same size and env as the given tensor-like object, filled with zeros.
    fn zeros_like(&self) -> Self {
        Self::zeros(&self.shape(), self.env())
    }

    /// Create a tensor-like object with the same size and env as the given tensor-like object, filled with ones.
    fn ones_like(&self) -> Self {
        Self::ones(&self.shape(), self.env())
    }

    /// Create a tensor-like object with the same size and env as the given tensor-like object, filled with uninitialized values.
    fn empty_like(&self) -> Self {
        Self::empty(&self.shape(), self.env())
    }

    /// Create a 2-D tensor-like object with ones on the diagonal and zeros elsewhere.
    fn eye(size: i64, env: Self::Env) -> Self;

    /// Read named tensor-like objects from a .npz file.
    fn read_npz<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error>;

    /// Read named tensor-like objects from a .ot file.
    fn read_ot<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, anyhow::Error>;

    /// Copy the tensor-like object to self.
    fn copy_(&mut self, other: &Self);

    /// Copy to a new tensor-like object.
    fn copy(&self) -> Self {
        let mut result = self.empty_like();
        result.copy_(self);
        result
    }

    /// Fill the tensor-like object with given value.
    fn fill_(&mut self, value: impl Element);

    /// Uniformly fill the tensor-like object with values in the given range.
    fn uniform_(&mut self, low: f64, high: f64);

    /// Initialize the tensor-like object with values from kaiming uniform distribution.
    fn kaiming_uniform_(&mut self) {
        let fan_in: i64 = self.shape().iter().skip(1).product();
        let bound = (1.0 / fan_in as f64).sqrt();
        self.uniform_(-bound, bound);
    }

    /// Reshape the tensor-like object as the given size.
    fn reshape(&self, size: &[i64]) -> Self;

    /// Reshape the tensor-like object as the same size as the given tensor-like object.
    fn reshape_as(&self, other: &Self) -> Self {
        self.reshape(&other.shape())
    }

    /// View the tensor-like object as the given size.
    fn view(&self, size: &[i64]) -> Self;

    /// View the tensor-like object as the same size as the given tensor-like object.
    fn view_as(&self, other: &Self) -> Self {
        self.view(&other.shape())
    }
}

impl TensorLike for Tensor {
    type Kind = Kind;
    type Device = Device;

    fn env(&self) -> Self::Env {
        (self.kind(), self.device())
    }

    fn to_kind(&self, kind: Self::Kind) -> Self {
        self.to_kind(kind)
    }

    fn to_device(&self, device: Self::Device) -> Self {
        self.to_device(device)
    }

    fn to_env(&self, env: Self::Env) -> Self {
        self.to_kind(env.0).to_device(env.1)
    }

    fn shape(&self) -> Vec<i64> {
        self.size()
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

    fn copy_(&mut self, other: &Self) {
        Tensor::copy_(self, other)
    }

    fn fill_(&mut self, value: impl Element) {
        let _ = Tensor::fill_(self, value);
    }

    fn uniform_(&mut self, low: f64, high: f64) {
        let _ = Tensor::uniform_(self, low, high);
    }

    fn reshape(&self, size: &[i64]) -> Self {
        Tensor::reshape(self, size)
    }
    
    fn view(&self, size: &[i64]) -> Self {
        Tensor::view(self, size)
    }
}