
use raddar_derive::Module;

use crate::{core::TensorNN, nn::Module};

/// GeLU activation function.
///
/// See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
#[derive(Debug, Module)]
#[module(paramless)]
pub struct GeLU;

/// ReLU activation function.
///
/// See [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf).
#[derive(Debug, Module)]
#[module(paramless)]
pub struct ReLU;

/// Leaky ReLU activation function.
#[derive(Debug, Module)]
#[module(paramless)]
pub struct LeakyReLU {
    pub lambda: f64,
}

impl LeakyReLU {
    pub fn new(lambda: f64) -> LeakyReLU {
        LeakyReLU { lambda: lambda }
    }
}

impl<Ts: TensorNN> Module<Ts> for GeLU {
    fn forward(&self, input: &Ts) -> Ts {
        input.gelu()
    }
}
impl<Ts: TensorNN> Module<Ts> for ReLU {
    fn forward(&self, input: &Ts) -> Ts {
        input.relu()
    }
}

impl<Ts: TensorNN> Module<Ts> for LeakyReLU {
    fn forward(&self, input: &Ts) -> Ts {
        input.leaky_relu(self.lambda)
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
