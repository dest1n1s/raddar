#![feature(trait_alias)]
#![feature(anonymous_lifetime_in_impl_trait)]
#[cfg(feature = "ndarray-backend")]
pub mod ndarr;

pub mod tensor;
#[cfg(feature = "ndarray-backend")]
pub type ArrayTensor = ndarr::NdArrayTensor;

pub trait AnyNum = num::NumCast + num::Num + PartialOrd + Copy + 'static;

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
            infos: vec![IndexInfoItem::Single(0), IndexInfoItem::Range(0, None, 1)],
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
        let mut ts = NdArrayTensor::ones(&[2, 2, 2], TensorKind::F32);
        ts *= 2.0f64;
        ts.debug_print();
        let mut ts2 = NdArrayTensor::ones(&[2, 2, 4], TensorKind::F32);
        let mut ts2_2 = ts2.get(1);
        ts2_2 *= 2.0f64;
        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = ts.matmul(&ts2);
        ts3 = &ts3 * 2;

        ts3.backward();

        ts.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn self_ref_test() {
        let mut ts = NdArrayTensor::ones(&[2], TensorKind::F32);
        ts *= 2.0f64;
        ts.debug_print();
        ts += &ts.name_clone();
        ts.debug_print();
    }

    #[test]
    fn ln_pow_test() {
        let mut ts = NdArrayTensor::ones(&[2], TensorKind::F32);
        ts *= 2.0f64;
        let mut ts2 = NdArrayTensor::ones(&[2], TensorKind::F64);
        ts.set_requires_grad(true);
        ts2.set_requires_grad(true);

        let mut ts3 = &ts + &ts2;
        ts3 = ts3.pow(&ts);

        ts3.backward();

        ts3.debug_print();

        ts.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn argmax_test() {
        let ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        let mut ts2 = ts.get(0);
        ts2 *= 2.0f64;
        ts.debug_print();
        ts.argext_dim(0, false, true).debug_print();
    }

    #[test]
    fn scatter_test() {
        let mut dst = NdArrayTensor::zeros(&[3, 5], TensorKind::F32);
        let src = NdArrayTensor::ones(&[2, 5], TensorKind::I64);
        let index = NdArrayTensor::zeros(&[2, 5], TensorKind::I64);
        index.get(0).get(1).assign_scalar(1);
        index.get(0).get(2).assign_scalar(2);
        index.get(1).get(0).assign_scalar(2);
        index.get(1).get(3).assign_scalar(1);
        index.get(1).get(4).assign_scalar(2);

        dst.scatter_dim_(0, &index, &src, crate::tensor::ScatterReduction::Add);

        dst.debug_print();
    }

    #[test]
    fn max_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        let mut ts1 = ts.get(0);
        ts1 *= 2.0f64;
        ts.set_requires_grad(true);
        let (mut ts2, _) = ts.ext_dim(0, false, true);

        ts2.backward();

        ts.grad().debug_print();
    }

    #[test]
    fn abs_test() {
        let mut ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        let mut ts1 = ts.get(0);
        ts1 *= -2.0f64;
        ts.set_requires_grad(true);
        let mut ts2 = ts.abs();

        ts2.backward();

        ts.grad().debug_print();
    }

    #[test]
    fn cmp_test() {
        let ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        let mut ts1 = ts.get(0);
        ts1 *= -2.0f64;
        let ts2 = ts.cmp(&ts1, crate::tensor::CmpMode::GT);

        ts2.debug_print();
    }

    #[test]
    fn cat_test() {
        let ts = NdArrayTensor::ones(&[2, 2], TensorKind::F32);
        let mut ts1 = ts.get(0);
        ts1 *= -2.0f64;
        let mut ts2 = ts.get(1);
        ts2 *= 2.0f64;
        ts1.set_requires_grad(true);
        ts2.set_requires_grad(true);
        let mut ts3 = NdArrayTensor::cat(&[&ts1, &ts2], 0);

        ts3 = &ts3 * 2;

        ts3.backward();

        ts1.grad().debug_print();
        ts2.grad().debug_print();
    }

    #[test]
    fn reshape_test(){
        let mut ts = NdArrayTensor::ones(&[2, 3], TensorKind::F32);
        ts *= 2.0f64;
        ts.set_requires_grad(true);

        let mut ts2 = ts.reshape(&[6]);

        ts2.backward();

        ts.grad().debug_print();
    }

    #[test]
    fn linear_fit_test() {
        let x = NdArrayTensor::zeros(&[100], TensorKind::F32);
        let y = NdArrayTensor::zeros(&[100], TensorKind::F32);
        let mut w = NdArrayTensor::zeros(&[1], TensorKind::F32);
        let mut b = NdArrayTensor::zeros(&[1], TensorKind::F32);
        let learning_rate = 1e-4f64;
        for i in 0..100 {
            x.get(i).assign_scalar(i as f64);
            y.get(i).assign_scalar(2.0f64 * i as f64 + 2.0f64);
        }
        w.set_requires_grad(true);
        b.set_requires_grad(true);
        for _ in 0..30000 {
            w.zero_grad();
            b.zero_grad();
            let y_pred = &(&x * &w) + &b;
            let mut loss = &(&y_pred - &y).pow_scalar(2).sum() / 100;
            loss.backward();
            w -= &w.grad() * learning_rate;
            b -= &b.grad() * learning_rate;
        }
        w.debug_print();
        b.debug_print();
    }
}
