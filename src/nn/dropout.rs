use crate::core::TensorNN;

use super::Module;
use raddar_derive::{ArchitectureBuilder, CallableModule, NonParameterModule};

/// A dropout layer.
#[derive(ArchitectureBuilder, Debug, CallableModule, NonParameterModule)]
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
