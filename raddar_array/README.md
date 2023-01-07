# raddar_array
This crate contains everything [`raddar`](https://github.com/dest1n1s/raddar) needs in a backend to work with tensors.

It works as a wrapper around [`ndarray`](https://github.com/rust-ndarray/ndarray) and [`tch-rs`](https://github.com/dest1n1s/tch-rs). It provides a `Tensor` struct that can be used to create tensors and perform operations on them. Since `ndarray` does not support automatic differentiation, it implements autograd for tensors too.

## Features
- `std`: Use the standard library. This is enabled by default.
- `ndarray-backend`: Use `ndarray` as the backend. This is enabled by default.
- `tch-backend`: Use `tch-rs` as the backend. (**Not Implemented yet**)

## Examples
### Creating a tensor
```rust
let mut tensor = ArrayTensor::zeros(&[2, 2], TensorKind::F32);
```

### Performing operations on a tensor
```rust
// Be free to use any type of scalar defined in `TensorKind`.
// They are automatically converted to the correct type.
tensor += 1i32;
tensor *= 2.0f32;
tensor -= 3.0f64;
tensor /= 4.0f64;
```

### Performing operations on tensors
```rust
let mut tensor2 = ArrayTensor::ones(&[2, 2], TensorKind::F64);

// Be free to use any type of tensors here.
// They are automatically converted to the LEFT tensor's type.
// i.e. `tensor2` is converted to `tensor`'s type, and added to it.
tensor += &tensor2;

// You can also use:
tensor = &tensor + &tensor2;
```

### Slicing a tensor
```rust
let mut tensor2 = tensor.get(1);

// You can also use:
let mut tensor2 = tensor.slice(vec![IndexInfoItem::Single(1), IndexInfoItem::Range(0, None, 1)].into());
```

### Performing operations on tensors with autograd
```rust
let mut tensor = ArrayTensor::zeros(&[2, 2], TensorKind::F32);
let mut tensor2 = ArrayTensor::ones(&[2, 2], TensorKind::F32);

tensor.set_requires_grad(true);
tensor2.set_requires_grad(true);

// Optional: only needed if you need to call `backward` on the tensor multiple times.
tensor.zero_grad();
tensor2.zero_grad();

let mut tensor3 = &tensor + &tensor2;
// DO-NOT: in-place operations are not recommended with autograd!
// tensor += &tensor2;

let (mut result, _) = tensor3.ext_dim(0, false, true);

assert_eq!(result.size(), vec![2]);

result.backward();

println!("tensor grad: {:?}", tensor.grad());
println!("tensor2 grad: {:?}", tensor2.grad());
```