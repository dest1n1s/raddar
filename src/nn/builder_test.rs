use raddar_derive::{CallableModule, PartialBuilder, NonParameterModule};
use tch::Tensor;
use derive_builder::Builder;

use super::Module;

#[derive(Debug, CallableModule, NonParameterModule, PartialBuilder)]
pub struct TestMaxPooling1D {
    pub not_build: i64,

    #[builder(default = "[3]")]
    pub kernel_size: [i64; 1],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 1],    

    #[builder(default = "[1]")]
    pub padding: [i64; 1],

    #[builder(default = "[1]")]
    pub dilation: [i64; 1],

    #[builder(default = "false")]
    pub ceil_mode: bool,
}

impl TestMaxPooling1D {
    pub fn new(config: TestMaxPooling1DConfig) -> Self {
        Self {
            not_build: 1,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            ceil_mode: config.ceil_mode,
        }
    }
}

impl Module for TestMaxPooling1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool1d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}