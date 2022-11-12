use raddar_derive::Module;

use crate::core::{Cellable, TensorCell, TensorNN};

use super::{module::Module, StateDict, Trainable};

// A simple fully-connected layer.
#[derive(Debug, Module)]
#[module(tensor_type="Ts", builder)]
pub struct Linear<Ts: TensorNN> {
    pub linear_weight: TensorCell<Ts>,
    pub linear_bias: Option<TensorCell<Ts>>,
    #[builder]
    pub input_dim: i64,
    #[builder]
    pub output_dim: i64,
    #[builder(default = "true")]
    pub bias: bool,
}

impl<Ts: TensorNN> Trainable<Ts> for Linear<Ts> {
    fn parameters(&self) -> StateDict<Ts> {
        let mut result = StateDict::new();
        result.insert("weight".to_owned(), self.linear_weight.clone());
        if let Some(bias) = &self.linear_bias {
            result.insert("bias".to_owned(), bias.clone());
        }
        result
    }
}

impl<Ts: TensorNN> Module<Ts> for Linear<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let weight = self.linear_weight.lock();
        if let Some(bias) = &self.linear_bias {
            let bias = bias.lock();
            input.matmul(&*weight) + &*bias
        } else {
            input.matmul(&*weight)
        }
    }
}

impl<Ts: TensorNN> Linear<Ts> {
    pub fn new(config: LinearConfig<Ts>) -> Linear<Ts> {
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        let bias = config.bias;
        let mut weight = Ts::empty(&[input_dim, output_dim], Ts::Env::default())
            .set_requires_grad(true);

        let mut linear_bias = Ts::empty(&[output_dim], Ts::Env::default()).set_requires_grad(true);

        weight.no_grad_mut().kaiming_uniform_();
        linear_bias.no_grad_mut().kaiming_uniform_();
        Linear {
            linear_weight: weight.cell(),
            linear_bias: if bias { Some(linear_bias.cell()) } else { None },
            input_dim,
            output_dim,
            bias,
        }
    }
}
