use crate::core::StateDict;
use crate::nn::Module;
use raddar_derive::CallableModule;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use tch::Tensor;

use super::Trainable;

/// A module composed by a sequential of modules.
#[derive(Debug, CallableModule)]
pub struct Sequential(Vec<Box<dyn Module>>);

#[derive(Debug, CallableModule)]
pub struct NamedSequential(BTreeMap<String, Box<dyn Module>>);

impl Deref for Sequential {
    type Target = Vec<Box<dyn Module>>;

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
    type Target = BTreeMap<String, Box<dyn Module>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Box<dyn Module>>> for Sequential {
    fn from(seq: Vec<Box<dyn Module>>) -> Self {
        Sequential(seq)
    }
}

impl FromIterator<Box<dyn Module>> for Sequential {
    fn from_iter<I: IntoIterator<Item = Box<dyn Module>>>(iter: I) -> Self {
        Sequential(iter.into_iter().collect())
    }
}

impl From<BTreeMap<String, Box<dyn Module>>> for NamedSequential {
    fn from(seq: BTreeMap<String, Box<dyn Module>>) -> Self {
        NamedSequential(seq)
    }
}

impl FromIterator<(String, Box<dyn Module>)> for NamedSequential {
    fn from_iter<I: IntoIterator<Item = (String, Box<dyn Module>)>>(iter: I) -> Self {
        NamedSequential(iter.into_iter().collect())
    }
}

impl Trainable for Sequential {
    fn parameters(&self) -> StateDict {
        let mut state_dict = StateDict::new();
        for (i, module) in self.iter().enumerate() {
            let child = module.parameters();
            state_dict.append_child(i.to_string(), child)
        }
        state_dict
    }
}

impl Module for Sequential {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for module in self.iter() {
            x = module.forward(&x)
        }
        x
    }
}

impl Trainable for NamedSequential {
    fn parameters(&self) -> StateDict {
        let mut state_dict = StateDict::new();
        for (name, module) in self.iter() {
            let child = module.parameters();
            state_dict.append_child(name.clone(), child)
        }
        state_dict
    }
}

impl Module for NamedSequential {
    fn forward(&self, input: &tch::Tensor) -> tch::Tensor {
        let mut x = input + 0;
        for (_, module) in self.iter() {
            x = module.forward(&x)
        }
        x
    }
}

#[macro_export]
macro_rules! seq {
    ($($module:expr),* $(,)?) => {
        {
            $crate::nn::sequential::Sequential::from(vec![$(Box::new($module) as Box<dyn $crate::nn::Module>,)*])
        }
    };
}

#[macro_export]
macro_rules! named_seq {
    ($($name:expr => $module:expr),* $(,)?) => {
        {
                vec![$(($name.to_string(), Box::new($module) as Box<dyn $crate::nn::Module>),)*]
                    .into_iter()
                    .collect::<$crate::nn::sequential::NamedSequential>()
        }
    };
}
