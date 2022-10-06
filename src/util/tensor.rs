#[macro_export]
macro_rules! tensor_vec {
    ($($x:expr),* $(,)?) => {
        {
            vec![$(std::sync::Arc::new(tch::Tensor::of_slice(& $x)),)*]
        }
    };
}