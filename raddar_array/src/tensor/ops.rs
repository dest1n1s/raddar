use crate::{ArrayTensor, ndarr::Element};

pub trait Operation<E: Element>
{
    fn backward(&self, grad: ArrayTensor<E>);
}
