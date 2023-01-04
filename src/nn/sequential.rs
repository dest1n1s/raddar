use crate::{nn::Module, core::TensorNN};
use raddar_derive::Module;
use std::ops::{Deref, DerefMut};

use super::{Mod, Trainable, TrainableDict};

/// A module composed by a sequential of modules.
#[derive(Debug, Module)]
#[module(tensor_type="Ts")]
pub struct Sequential<Ts: TensorNN>(Vec<Mod<dyn Module<Ts>, Ts>>);

#[derive(Debug, Module)]
#[module(tensor_type="Ts")]
pub struct NamedSequential<Ts: TensorNN>(Vec<(String, Mod<dyn Module<Ts>, Ts>)>);

impl<Ts: TensorNN> Default for Sequential<Ts> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<Ts: TensorNN> Default for NamedSequential<Ts> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<Ts: TensorNN> Deref for Sequential<Ts> {
    type Target = Vec<Mod<dyn Module<Ts>, Ts>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Ts: TensorNN> DerefMut for Sequential<Ts> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<Ts: TensorNN> DerefMut for NamedSequential<Ts> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<Ts: TensorNN> Deref for NamedSequential<Ts> {
    type Target = Vec<(String, Mod<dyn Module<Ts>, Ts>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Ts: TensorNN> From<Vec<Mod<dyn Module<Ts>, Ts>>> for Sequential<Ts> {
    fn from(seq: Vec<Mod<dyn Module<Ts>, Ts>>) -> Self {
        Sequential(seq)
    }
}

impl<Ts: TensorNN> FromIterator<Mod<dyn Module<Ts>, Ts>> for Sequential<Ts> {
    fn from_iter<I: IntoIterator<Item = Mod<dyn Module<Ts>, Ts>>>(iter: I) -> Self {
        Sequential(iter.into_iter().collect())
    }
}

impl<Ts: TensorNN> From<Vec<(String, Mod<dyn Module<Ts>, Ts>)>> for NamedSequential<Ts> {
    fn from(seq: Vec<(String, Mod<dyn Module<Ts>, Ts>)>) -> Self {
        NamedSequential(seq)
    }
}

impl<Ts: TensorNN> FromIterator<(String, Mod<dyn Module<Ts>, Ts>)> for NamedSequential<Ts> {
    fn from_iter<I: IntoIterator<Item = (String, Mod<dyn Module<Ts>, Ts>)>>(
        iter: I,
    ) -> Self {
        NamedSequential(iter.into_iter().collect())
    }
}

impl<Ts: TensorNN> Trainable<Ts> for Sequential<Ts> {
    fn child_modules(&self) -> TrainableDict<Ts> {
        let mut children = TrainableDict::new();
        for (i, module) in self.iter().enumerate() {
            children.insert(i.to_string(), module.clone());
        }
        children
    }
}

impl<Ts: TensorNN> Module<Ts> for Sequential<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
        let mut x = input + 0;
        for module in self.iter() {
            x = module(&x)
        }
        x
    }
}

impl<Ts: TensorNN> Trainable<Ts> for NamedSequential<Ts> {
    fn child_modules(&self) -> TrainableDict<Ts> {
        let mut children = TrainableDict::new();
        for (name, module) in self.iter() {
            children.insert(name.clone(), module.clone());
        }
        children
    }
}

impl<Ts: TensorNN> Module<Ts> for NamedSequential<Ts> {
    fn forward(&self, input: &Ts) -> Ts {
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
            $crate::nn::Mod::new($crate::nn::sequential::Sequential::from(vec![$($module as $crate::nn::Mod<dyn $crate::nn::Module<_>, _>,)*]))
        }
    };
}

#[macro_export]
macro_rules! named_seq {
    ($($name:expr => $module:expr),* $(,)?) => {
        {
            $crate::nn::Mod::new(vec![$(($name.to_string(), $module as $crate::nn::Mod<dyn $crate::nn::Module<_>, _>),)*]
                    .into_iter()
                    .collect::<$crate::nn::sequential::NamedSequential<_>>())
        }
    };
}
