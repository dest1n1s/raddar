use std::f64::consts::PI;

use crate::nn::module::{Module, NonParameterModule};
use tch::Tensor;

#[derive(Debug)]
pub struct GeLU;

impl NonParameterModule for GeLU {}

impl Module for GeLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let z = (input + &input.pow_tensor_scalar(3) * 0.044715) * (2.0f64 / PI).sqrt();
        0.5 * input * (1 + z.tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_test() {
        let input = Tensor::of_slice2(&[[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
        let output = GeLU.forward(&input);
        let expected = Tensor::of_slice2(&[
            [0.8413],
            [2.9960],
            [5.0000],
            [3.9999],
            [8.0000],
            [10.0000],
            [1.9545],
            [6.0000],
        ]);
        assert!(f64::from((output - expected).square().sum(tch::Kind::Double)) < 1e-4);
    }
}
