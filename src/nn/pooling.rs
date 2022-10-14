use derive_builder::Builder;
use raddar_derive::{CallableModule, NonParameterModule};
use tch::Tensor;

use super::Module;

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct MaxPooling1D {
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

impl MaxPooling1D {
    pub fn new(kernel_size: [i64; 1], stride: [i64; 1], padding: [i64; 1], dilation: [i64; 1], ceil_mode: bool) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
}

impl Module for MaxPooling1D {
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

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct MaxPooling2D {
    #[builder(default = "[3, 3]")]
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
    pub fn new(kernel_size: [i64; 2], stride: [i64; 2], padding: [i64; 2], dilation: [i64; 2], ceil_mode: bool) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
}

impl Module for MaxPooling2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool2d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned", build_fn(skip))]
pub struct MaxPooling3D {
    #[builder(default = "[3, 3, 3]")]
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
    pub fn new(kernel_size: [i64; 3], stride: [i64; 3], padding: [i64; 3], dilation: [i64; 3], ceil_mode: bool) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
}

impl Module for MaxPooling3D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.max_pool3d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.ceil_mode,
        )
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
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
    pub fn new(kernel_size: [i64; 1], stride: [i64; 1], padding: [i64; 1], ceil_mode: bool, count_include_pad: bool) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        }
    }
}

impl Module for AveragePooling1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.avg_pool1d(
            &self.kernel_size,
            &self.stride,
            &self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
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
    pub fn new(kernel_size: [i64; 2], stride: [i64; 2], padding: [i64; 2], ceil_mode: bool, count_include_pad: bool, divisor_override: Option<i64>) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        }
    }
}

impl Module for AveragePooling2D {
    fn forward(&self, input: &Tensor) -> Tensor {
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

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
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
    pub fn new(kernel_size: [i64; 3], stride: [i64; 3], padding: [i64; 3],  ceil_mode: bool, count_include_pad: bool, divisor_override: Option<i64>) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        }
    }
}

impl Module for AveragePooling3D {
    fn forward(&self, input: &Tensor) -> Tensor {
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

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveMaxPooling1D {
    #[builder(default = "[1]")]
    pub output_size: [i64; 1],
}

impl AdaptiveMaxPooling1D {
    pub fn new(output_size: [i64; 1]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveMaxPooling1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_max_pool1d(&self.output_size).0
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveMaxPooling2D {
    #[builder(default = "[1, 1]")]
    pub output_size: [i64; 2],
}

impl AdaptiveMaxPooling2D {
    pub fn new(output_size: [i64; 2]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveMaxPooling2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_max_pool2d(&self.output_size).0
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveMaxPooling3D {
    #[builder(default = "[1, 1, 1]")]
    pub output_size: [i64; 3],
}

impl AdaptiveMaxPooling3D {
    pub fn new(output_size: [i64; 3]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveMaxPooling3D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_max_pool3d(&self.output_size).0
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveAveragePooling1D {
    #[builder(default = "[1]")]
    pub output_size: [i64; 1],
}

impl AdaptiveAveragePooling1D {
    pub fn new(output_size: [i64; 1]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveAveragePooling1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_avg_pool1d(&self.output_size)
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveAveragePooling2D {
    #[builder(default = "[1, 1]")]
    pub output_size: [i64; 2],
}

impl AdaptiveAveragePooling2D {
    pub fn new(output_size: [i64; 2]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveAveragePooling2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_avg_pool2d(&self.output_size)
    }
}

#[derive(Debug, CallableModule, NonParameterModule, Builder)]
#[builder(pattern = "owned")]
pub struct AdaptiveAveragePooling3D {
    #[builder(default = "[1, 1, 1]")]
    pub output_size: [i64; 3],
}

impl AdaptiveAveragePooling3D {
    pub fn new(output_size: [i64; 3]) -> Self {
        Self {
            output_size,
        }
    }
}

impl Module for AdaptiveAveragePooling3D {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.adaptive_avg_pool3d(&self.output_size)
    }
}