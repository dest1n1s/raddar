use crate::core::StateDictOrigin;
use crate::nn::Module;
use raddar_derive::CallableModule;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use tch::Tensor;

use super::{Trainable, Mod, ModuleDict};

/// A module composed by a sequential of modules.
#[derive(Debug, CallableModule)]
pub struct Sequential(Vec<Mod<dyn Module>>);

#[derive(Debug, CallableModule)]
pub struct NamedSequential(BTreeMap<String, Mod<dyn Module>>);

impl Deref for Sequential {
    type Target = Vec<Mod<dyn Module>>;

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
    type Target = BTreeMap<String, Mod<dyn Module>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec<Mod<dyn Module>>> for Sequential {
    fn from(seq: Vec<Mod<dyn Module>>) -> Self {
        Sequential(seq)
    }
}

impl FromIterator<Mod<dyn Module>> for Sequential {
    fn from_iter<I: IntoIterator<Item = Mod<dyn Module>>>(iter: I) -> Self {
        Sequential(iter.into_iter().collect())
    }
}

impl From<BTreeMap<String, Mod<dyn Module>>> for NamedSequential {
    fn from(seq: BTreeMap<String, Mod<dyn Module>>) -> Self {
        NamedSequential(seq)
    }
}

impl FromIterator<(String, Mod<dyn Module>)> for NamedSequential {
    fn from_iter<I: IntoIterator<Item = (String, Mod<dyn Module>)>>(iter: I) -> Self {
        NamedSequential(iter.into_iter().collect())
    }
}

impl Trainable for Sequential {
    fn child_modules(&self) -> ModuleDict {
        let mut children = ModuleDict::new();
        for (i, module) in self.iter().enumerate() {
            children.insert(i.to_string(), module.clone())
        }
        children
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
    fn child_modules(&self) -> ModuleDict {
        let mut children = StateDictOrigin::new();
        for (name, module) in self.iter() {
            children.children(name.clone(), module)
        }
        children
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
            $crate::nn::sequential::Sequential::from(vec![$($module as $crate::nn::Mod<dyn $crate::nn::Module>,)*])
        }
    };
}

#[macro_export]
macro_rules! named_seq {
    ($($name:expr => $module:expr),* $(,)?) => {
        {
                vec![$(($name.to_string(), $module as $crate::nn::Mod<dyn $crate::nn::Module>),)*]
                    .into_iter()
                    .collect::<$crate::nn::sequential::NamedSequential>()
        }
    };
}
