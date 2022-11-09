use super::Module;
use raddar_derive::{ArchitectureBuilder, NonParameterModule};
use tch::Tensor;

/// A dropout layer.
#[derive(ArchitectureBuilder, Debug, NonParameterModule)]
pub struct Dropout {
    #[builder(default = "0.5")]
    p: f64,
    #[builder(default = "true")]
    train: bool,
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
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
