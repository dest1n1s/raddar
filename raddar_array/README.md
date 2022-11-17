# raddar_array
This crate contains everything [`raddar`](https://github.com/dest1n1s/raddar) needs in a backend to work with tensors.

It works as a wrapper around [`ndarray`](https://github.com/rust-ndarray/ndarray) and [`tch-rs`](https://github.com/dest1n1s/tch-rs). It provides a `Tensor` struct that can be used to create tensors and perform operations on them. Since `ndarray` does not support automatic differentiation, it implements autograd for tensors too.

## Features
- `std`: Use the standard library. This is enabled by default.
- `ndarray-backend`: Use `ndarray` as the backend. This is enabled by default.
- `tch-backend`: Use `tch-rs` as the backend.