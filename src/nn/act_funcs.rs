use crate::nn::module::{Module, NonParameterModule};
use tch::Tensor;

#[derive(Debug)]
struct GeLU;

impl NonParameterModule for GeLU {}

impl Module for GeLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let z = (input + &input.pow_tensor_scalar(3) * 0.044715) * 0.797845;
        0.5 * input * (1 + z.tanh())
    }
}
