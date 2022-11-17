#![feature(trait_alias)]
#[cfg(feature = "ndarray-backend")]
pub mod ndarr;

pub mod tensor;
#[cfg(feature = "ndarray-backend")]
pub type ArrayTensor = ndarr::NdArrayTensor;
#[cfg(test)]
mod tests {
    use crate::{
        ndarr::NdArrayTensor,
        tensor::{TensorKind, TensorMethods, AutoGradTensorMethods},
    };

    #[test]
    fn it_works() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::zeros(&[2, 2], TensorKind::F32);
        ts2 *= 2.0f64;
        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);
        let ts3 = &ts + &ts2;
        let ts4 = &ts3 + &ts;
        let mut ts4 = &ts4 - &ts3;
        ts4.backward();
        // ts4 = ts + ts2 + ts - (ts + ts2),
        // so the gradient of `ts` should be all 1.0,
        // and the gradient of `ts2` should be all 0.0.
        ts.grad().debug_print();
        ts2.grad().debug_print();
    }
}
