# Raddar

A Rust-native approach to deep learning.

Currently since there's no mature solution for n-dimensional array computing on gpu in rust, we temporarily use the `Tensor` and other CUDA toolkit from `tch`, which provides Rust bindings for `libtorch`. But we won't use high-level parts of it.

## Getting-Started

This crate requires CUDA and `libtorch` support. You need to:

- Install CUDA 11.3 from [NVIDIA](https://developer.nvidia.com/cuda-11-3-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local). After the installation, you need to set the `TORCH_CUDA_VERSION` environment variable to `11.3`, or `libtorch` cannot find your CUDA.
- Install `libtorch`. You can follow the instructions in [tch-rs](https://github.com/LaurentMazare/tch-rs/blob/main/README.md#getting-started). 
- (On Windows) Check if your rust use a MSVC-based toolchain. The GNU toolchain could not successfully compile `torch-sys`. You can check the current toolchain by running
  ```shell
  rustup toolchain list
  ```
  If not, run
  ```shell
  rustup toolchain install nightly-x86_64-pc-windows-msvc
  rustup toolchain default nightly-x86_64-pc-windows-msvc
  ```
  to switch the toolchain.
- You should now be able to run the project. Try ```device_test``` in ```tests/tch_test.rs``` to see if the all settings are correct.