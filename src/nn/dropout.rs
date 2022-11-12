use raddar_derive::Module;

use crate::core::TensorNN;

use super::Module;

/// A dropout layer.
#[derive(Debug, Module)]
#[module(paramless, builder)]
pub struct Dropout {
    #[builder(default = "0.5")]
    p: f64,
    #[builder(default = "true")]
    train: bool,
}

impl<Ts: TensorNN> Module<Ts> for Dropout {
    fn forward(&self, input: &Ts) -> Ts {
        input.dropout(self.p, self.train)
    }
}

impl Dropout {
    pub fn new(config: DropoutConfig) -> Self {
        Self {
            p: config.p,
            train: config.train,
        }
    }
}
