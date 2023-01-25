use std::sync::{Arc, Mutex, Weak};
use higher_order_closure::hrtb;
use crate::{
    go_backward,
    ndarr::{
        lens::{CompositeLen, LookThrough, ViewLens},
        ops::add_grad,
        KindedArrayD, NdArrayTensor, NdArrayTensorInternal, ViewMethods, ViewsImmut,
    },
    tensor::{ops::Operation, AutoGradTensorMethods, TensorMethods},
};

pub(crate) struct WhereOp {
    output: Weak<Mutex<NdArrayTensorInternal>>,
    condition: Arc<Mutex<NdArrayTensorInternal>>,
    x: Arc<Mutex<NdArrayTensorInternal>>,
    y: Arc<Mutex<NdArrayTensorInternal>>,
}

impl WhereOp {
    pub fn forward(
        condition: &NdArrayTensor,
        x: &NdArrayTensor,
        y: &NdArrayTensor,
    ) -> NdArrayTensor {
        // let mut output: NdArrayTensor = borrow_three_tensor_internals!(
        //     condition.internal.as_ref().unwrap(),
        //     x.internal.as_ref().unwrap(),
        //     y.internal.as_ref().unwrap(),
        //     inputs,
        //     {
        //         inputs
        //             .1
        //             .as_view()
        //             .r#where(&*inputs.0.as_view(), &*inputs.2.as_view())
        //     }
        // )
        // .into();

        let mut output: NdArrayTensor = ViewLens::with_tensor(x)
            .and(condition)
            .and(y)
            .look_through(hrtb!(|inputs: ViewsImmut<'_, '_>| -> KindedArrayD {
                inputs[0].r#where(&*inputs[1], &*inputs[2])
            }))
            .into();
        if x.requires_grad() || y.requires_grad() {
            output.i().op = Some(Arc::new(WhereOp {
                output: output.i_ref(),
                condition: condition.i_copy(),
                x: x.i_copy(),
                y: y.i_copy(),
            }));
            output.set_requires_grad(true);
        }
        output
    }
}

impl Operation for WhereOp {
    fn backward(&self, grad: NdArrayTensor) {
        add_grad(self.output.clone(), grad.name_clone());

        let shape = self.x.lock().unwrap().as_view().size();
        let dtype = self.x.lock().unwrap().as_view().kind();

        let mask = NdArrayTensor::zeros(&shape, dtype);

        let cond: NdArrayTensor = (&self.condition).into();

        let grad_a = grad.r#where(&cond, &mask);
        let grad_b = mask.r#where(&cond, &grad);

        go_backward!(self.x, grad_a);
        go_backward!(self.y, grad_b);
    }
}
