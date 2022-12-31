#![feature(trait_alias)]
#![feature(anonymous_lifetime_in_impl_trait)]
#[cfg(feature = "ndarray-backend")]
pub mod ndarr;

pub mod tensor;
#[cfg(feature = "ndarray-backend")]
pub type ArrayTensor = ndarr::NdArrayTensor;
#[cfg(test)]
mod tests {
    use crate::{
        ndarr::NdArrayTensor,
        tensor::{
            index::{IndexInfo, IndexInfoItem},
            ArrayMethods, AutoGradTensorMethods, TensorKind, TensorMethods,
        },
    };

    #[test]
    fn it_works() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::zeros(&[2, 2], TensorKind::F32);
        ts2 += 1;
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

    #[test]
    fn simple_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::zeros(&[2], TensorKind::F32);
        ts2 += 1;
        ts2 *= 2.0f64;
        ts2.broadcast(&[10, 2]).debug_print();
        ts.t()
            .broadcast(&[10, 2, 2])
            .broadcast(&[10, 10, 2, 2])
            .debug_print();
        ts.set_requires_grad(true);

        let ts = ts.t().t();
        let ts_1 = ts.slice(IndexInfo {
            infos: vec![IndexInfoItem::Single(0), IndexInfoItem::Range(0, 2, 1)],
        });
        let mut ts3 = &ts_1 + &ts2;

        ts3.backward();

        ts.grad().debug_print();
        ts.grad().t().debug_print();
        ts.debug_print();
    }

    #[test]
    fn mul_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::ones(&[2, 2], TensorKind::F32);

        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = &ts + &ts2;
       
        ts3 = &ts3 * 2;
        ts3 = &ts3 * &ts2;
        ts3 = &ts3 / &ts2;
        ts3.backward();

        ts3.debug_print();

        ts.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn cobroadcast_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::ones(&[2], TensorKind::F32);

        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = &ts + &ts2;
        ts3 = &ts3 * 2;
        
        ts3.backward();

        ts3.debug_print();

        ts.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn sum_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::ones(&[2], TensorKind::F32);

        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = ts.sum_dim(&[0], false);
        ts3 = &ts3 + &ts2;

        ts3.backward();

        ts3.debug_print();
        
        ts.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn unsqueeze_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        ts *= 2.0f64;

        ts.unsqueeze(0).debug_print();
        ts.unsqueeze(1).debug_print();
        ts.unsqueeze(2).debug_print();

        ts.unsqueeze(0).unsqueeze(0).debug_print();

        ts.unsqueeze_(0);
        ts.debug_print();
    }

    #[test]
    fn matmul_test() {
        let mut ts = NdArrayTensor::ones(&[2], TensorKind::F32);
        ts *= 2.0f64;
        ts.debug_print();
        let mut ts2 = NdArrayTensor::ones(&[2], TensorKind::F32);

        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = ts.matmul(&ts2);
        ts3 = &ts3 * 2;
        
        ts3.backward();

        let it: f32 = ts3.item();
        println!("{:?}", it);

        ts.grad().debug_print();
        ts2.grad().debug_print();
    }
}
