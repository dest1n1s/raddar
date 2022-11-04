use crate::nn::Module;
use raddar_derive::CallableModule;
use std::ops::{Deref, DerefMut};
use tch::Tensor;

use super::{Mod, Trainable, TrainableDict};

/// A module composed by a sequential of modules.
#[derive(Debug, CallableModule, Default)]
pub struct Sequential(Vec<Mod<dyn Module<Tensor, Tensor>>>);

#[derive(Debug, CallableModule, Default)]
pub struct NamedSequential(Vec<(String, Mod<dyn Module>)>);

impl Deref for Sequential {
    type Target = Vec<Mod<dyn Module<Tensor, Tensor>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Sequential {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl DerefMut for NamedSequential {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for NamedSequential {
    type Target = Vec<(String, Mod<dyn Module>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Mod<dyn Module<Tensor, Tensor>>>> for Sequential {
    fn from(seq: Vec<Mod<dyn Module<Tensor, Tensor>>>) -> Self {
        Sequential(seq)
    }
}

impl FromIterator<Mod<dyn Module<Tensor, Tensor>>> for Sequential {
    fn from_iter<I: IntoIterator<Item = Mod<dyn Module<Tensor, Tensor>>>>(iter: I) -> Self {
        Sequential(iter.into_iter().collect())
    }
}

impl From<Vec<(String, Mod<dyn Module>)>> for NamedSequential {
    fn from(seq: Vec<(String, Mod<dyn Module>)>) -> Self {
        NamedSequential(seq)
    }
}

impl FromIterator<(String, Mod<dyn Module<Tensor, Tensor>>)> for NamedSequential {
    fn from_iter<I: IntoIterator<Item = (String, Mod<dyn Module<Tensor, Tensor>>)>>(
        iter: I,
    ) -> Self {
        NamedSequential(iter.into_iter().collect())
    }
}

impl Trainable for Sequential {
    fn child_modules(&self) -> TrainableDict {
        let mut children = TrainableDict::new();
        for (i, module) in self.iter().enumerate() {
            children.insert(i.to_string(), module.clone());
        }
        children
    }
}

impl Module<Tensor, Tensor> for Sequential {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for module in self.iter() {
            x = module(&x)
        }
        x
    }
}

impl Trainable for NamedSequential {
    fn child_modules(&self) -> TrainableDict {
        let mut children = TrainableDict::new();
        for (name, module) in self.iter() {
            children.insert(name.clone(), module.clone());
        }
        children
    }
}

impl Module<Tensor, Tensor> for NamedSequential {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for (_, module) in self.iter() {
            x = module(&x)
        }
        x
    }
}

#[macro_export]
macro_rules! seq {
    ($($module:expr),* $(,)?) => {
        {
            $crate::nn::Mod::new($crate::nn::sequential::Sequential::from(vec![$($module as $crate::nn::Mod<dyn $crate::nn::Module<tch::Tensor, tch::Tensor>>,)*]))
        }
    };
}

#[macro_export]
macro_rules! named_seq {
    ($($name:expr => $module:expr),* $(,)?) => {
        {
            $crate::nn::Mod::new(vec![$(($name.to_string(), $module as $crate::nn::Mod<dyn $crate::nn::Module<Tensor, Tensor>>),)*]
                    .into_iter()
                    .collect::<$crate::nn::sequential::NamedSequential>())
        }
    };
}
