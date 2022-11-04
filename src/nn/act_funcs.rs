use std::f64::consts::PI;

use raddar_derive::NonParameterModule;
use tch::Tensor;

use crate::nn::Module;

/// GeLU activation function.
///
/// See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
#[derive(Debug, NonParameterModule)]
pub struct GeLU;

/// ReLU activation function.
///
/// See [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf).
#[derive(Debug, NonParameterModule)]
pub struct ReLU;

/// Leaky ReLU activation function.
#[derive(Debug, NonParameterModule)]
pub struct LeakyReLU {
    pub lambda: f64,
}

impl LeakyReLU {
    pub fn new(lambda: f64) -> LeakyReLU {
        LeakyReLU { lambda: lambda }
    }
}

impl Module for GeLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let z = (input + &input.pow_tensor_scalar(3) * 0.044715) * (2.0f64 / PI).sqrt();
        0.5 * input * (1 + z.tanh())
    }
}
impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let y = input.zeros_like();
        let condition = input.ge(0);
        input.where_self(&condition, &y)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let y = -input * self.lambda;
        let condition = input.ge(0);
        input.where_self(&condition, &y)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor;

    use super::*;

    #[test]
    fn gelu_test() {
        let input = tensor!(&[[1.0], [3.0], [5.0], [4.0], [8.0], [10.0], [2.0], [6.0]]);
        let output = GeLU.forward(&input);
        let expected = tensor!(&[
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
