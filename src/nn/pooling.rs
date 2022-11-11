use raddar_derive::{NonParameterModule, CallableModule, ArchitectureBuilder};

use crate::core::TensorNN;

use super::Module;

/// A max pooling layer in 1 dimension.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct MaxPooling1D {
    #[builder]
    pub kernel_size: [i64; 1],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 1],    

    #[builder(default = "[0]")]
    pub padding: [i64; 1],

    #[builder(default = "[1]")]
    pub dilation: [i64; 1],

    #[builder(default = "false")]
    pub ceil_mode: bool,
}

impl MaxPooling1D {
    pub fn new(config: MaxPooling1DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            ceil_mode: config.ceil_mode,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for MaxPooling1D {
    fn forward(&self, input: &Ts) -> Ts {
        input.max_pool1d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}

/// A max pooling layer in 2 dimensions.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct MaxPooling2D {
    #[builder]
    pub kernel_size: [i64; 2],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 2],    

    #[builder(default = "[0, 0]")]
    pub padding: [i64; 2],

    #[builder(default = "[1, 1]")]
    pub dilation: [i64; 2],

    #[builder(default = "false")]
    pub ceil_mode: bool,
}

impl MaxPooling2D {
    pub fn new(config: MaxPooling2DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            ceil_mode: config.ceil_mode,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for MaxPooling2D {
    fn forward(&self, input: &Ts) -> Ts {
        input.max_pool2d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}

/// A max pooling layer in 3 dimensions.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct MaxPooling3D {
    #[builder]
    pub kernel_size: [i64; 3],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 3],    

    #[builder(default = "[0, 0, 0]")]
    pub padding: [i64; 3],

    #[builder(default = "[1, 1, 1]")]
    pub dilation: [i64; 3],

    #[builder(default = "false")]
    pub ceil_mode: bool,
}

impl MaxPooling3D {
    pub fn new(config: MaxPooling3DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            ceil_mode: config.ceil_mode,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for MaxPooling3D {
    fn forward(&self, input: &Ts) -> Ts {
        input.max_pool3d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}

/// An average pooling layer in 1 dimension.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AveragePooling1D {
    #[builder(default = "[3]")]
    pub kernel_size: [i64; 1],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 1],    

    #[builder(default = "[1]")]
    pub padding: [i64; 1],

    #[builder(default = "false")]
    pub ceil_mode: bool,

    #[builder(default = "false")]
    pub count_include_pad: bool,
}

impl AveragePooling1D {
    pub fn new(config: AveragePooling1DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            ceil_mode: config.ceil_mode,
            count_include_pad: config.count_include_pad,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AveragePooling1D {
    fn forward(&self, input: &Ts) -> Ts {
        input.avg_pool1d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
    }
}

/// An average pooling layer in 2 dimensions.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]

pub struct AveragePooling2D {
    #[builder(default = "[3, 3]")]
    pub kernel_size: [i64; 2],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 2],    

    #[builder(default = "[0, 0]")]
    pub padding: [i64; 2],

    #[builder(default = "false")]
    pub ceil_mode: bool,

    #[builder(default = "false")]
    pub count_include_pad: bool,

    #[builder(default = "None")]
    pub divisor_override: Option<i64>,
}

impl AveragePooling2D {
    pub fn new(config: AveragePooling2DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            ceil_mode: config.ceil_mode,
            count_include_pad: config.count_include_pad,
            divisor_override: config.divisor_override,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AveragePooling2D {
    fn forward(&self, input: &Ts) -> Ts {
        input.avg_pool2d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
    }
}

/// An average pooling layer in 3 dimensions.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AveragePooling3D {
    #[builder(default = "[3, 3, 3]")]
    pub kernel_size: [i64; 3],

    #[builder(default = "self.kernel_size.unwrap().clone()")]
    pub stride: [i64; 3],

    #[builder(default = "[0, 0, 0]")]
    pub padding: [i64; 3],

    #[builder(default = "false")]
    pub ceil_mode: bool,

    #[builder(default = "false")]
    pub count_include_pad: bool,

    #[builder(default = "None")]
    pub divisor_override: Option<i64>,
}

impl AveragePooling3D {
    pub fn new(config: AveragePooling3DConfig) -> Self {
        Self {
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            ceil_mode: config.ceil_mode,
            count_include_pad: config.count_include_pad,
            divisor_override: config.divisor_override,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AveragePooling3D {
    fn forward(&self, input: &Ts) -> Ts {
        input.avg_pool3d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
    }
}

/// An adaptive max pooling layer in 1 dimension, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveMaxPooling1D {
    #[builder(default = "[1]")]
    pub output_size: [i64; 1],
}

impl AdaptiveMaxPooling1D {
    pub fn new(config: AdaptiveMaxPooling1DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveMaxPooling1D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_max_pool1d(&self.output_size).0
    }
}

/// An adaptive max pooling layer in 2 dimensions, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveMaxPooling2D {
    #[builder(default = "[1, 1]")]
    pub output_size: [i64; 2],
}

impl AdaptiveMaxPooling2D {
    pub fn new(config: AdaptiveMaxPooling2DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveMaxPooling2D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_max_pool2d(&self.output_size).0
    }
}

/// An adaptive max pooling layer in 3 dimensions, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveMaxPooling3D {
    #[builder(default = "[1, 1, 1]")]
    pub output_size: [i64; 3],
}

impl AdaptiveMaxPooling3D {
    pub fn new(config: AdaptiveMaxPooling3DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveMaxPooling3D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_max_pool3d(&self.output_size).0
    }
}

/// An adaptive average pooling layer in 1 dimension, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveAveragePooling1D {
    #[builder(default = "[1]")]
    pub output_size: [i64; 1],
}

impl AdaptiveAveragePooling1D {
    pub fn new(config: AdaptiveAveragePooling1DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveAveragePooling1D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_avg_pool1d(&self.output_size)
    }
}

/// An adaptive average pooling layer in 2 dimensions, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveAveragePooling2D {
    #[builder(default = "[1, 1]")]
    pub output_size: [i64; 2],
}

impl AdaptiveAveragePooling2D {
    pub fn new(config: AdaptiveAveragePooling2DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveAveragePooling2D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_avg_pool2d(&self.output_size)
    }
}

/// An adaptive average pooling layer in 3 dimensions, which outputs a fixed size vector.
#[derive(Debug, CallableModule, NonParameterModule, ArchitectureBuilder)]
pub struct AdaptiveAveragePooling3D {
    #[builder(default = "[1, 1, 1]")]
    pub output_size: [i64; 3],
}

impl AdaptiveAveragePooling3D {
    pub fn new(config: AdaptiveAveragePooling3DConfig) -> Self {
        Self {
            output_size: config.output_size,
        }
    }
}

impl<Ts: TensorNN> Module<Ts> for AdaptiveAveragePooling3D {
    fn forward(&self, input: &Ts) -> Ts {
        input.adaptive_avg_pool3d(&self.output_size)
    }
}