/// Create a [`Vec<Arc<Tensor>>`] from an array of 1d arrays.
#[macro_export]
macro_rules! tensor_vec {
    ($($x:expr),* $(,)?) => {
        {
            vec![$(std::sync::Arc::new(tch::Tensor::of_slice(& $x)),)*]
        }
    };
}

/// Decide if two tensors are equal.
/// 
/// Defaultly, the tensors are considered equal if they have the same shape and values whose MLE is less than 1e-6.
/// 
/// You can also explicitly specify the tolerance by passing a third argument.
#[macro_export]
macro_rules! tensor_eq {
    ($a:expr, $b:expr) => {
        {
            $a.size() == $b.size() && f64::from(($a - $b).square().sum(tch::Kind::Double)) < 1e-6
        }
    };
    ($a:expr, $b:expr, $c:expr) => {
        {
            $a.size() == $b.size() && f64::from(($a - $b).square().sum(tch::Kind::Double)) < $c
        }
    };
}

/// Assert if two tensors are equal.
/// 
/// Defaultly, the tensors are considered equal if they have the same shape and values whose MLE is less than 1e-6.
/// 
/// You can also explicitly specify the tolerance by passing a third argument.
#[macro_export]
macro_rules! assert_tensor_eq {
    ($a:expr, $b:expr) => {
        assert!(raddar::tensor_eq!($a, $b));
    };
    ($a:expr, $b:expr, $c:expr) => {
        assert!(raddar::tensor_eq!($a, $b, $c));
    };
}
