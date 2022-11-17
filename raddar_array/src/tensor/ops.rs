use crate::ArrayTensor;

pub trait Operation
{
    fn backward(&self, grad: ArrayTensor);
}
