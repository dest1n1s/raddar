use std::{
    collections::BTreeMap,
    marker::Unsize,
    ops::{CoerceUnsized, Deref},
    path::Path,
    sync::{Arc, Weak},
};

use anyhow::Ok;
use parking_lot::RwLock;
use tch::{no_grad, Device, Tensor};

use crate::core::{Cellable, TensorCell};

pub type StateDict = BTreeMap<String, TensorCell>;
pub type ModuleDict = BTreeMap<String, Mod<dyn Trainable>>;

#[derive(Debug, Clone)]
pub enum ModuleMode {
    Train,
    Eval,
}
/// A trait for anything that has trainable parameters.
pub trait Trainable: std::fmt::Debug {
    /// Defines the trainable parameters of the module. This does not include the parameters in child modules.
    ///
    /// By default, this returns an empty map. If your module has trainable parameters, you should override this method.
    fn parameters(&self) -> StateDict {
        BTreeMap::new()
    }

    /// Defines the static tensors of the module. This does not include the static tensors in child modules.
    ///
    /// By default, this returns an empty map. If your module has static tensors, you should override this method.
    fn static_tensors(&self) -> StateDict {
        BTreeMap::new()
    }

    /// Defines the child modules of the module.
    ///
    /// By default, this returns an empty map. If your module has child modules, you should override this method.
    fn child_modules(&self) -> ModuleDict {
        BTreeMap::new()
    }

    /// Returns the size of the parameters of the module.
    fn parameter_size(&self) -> usize {
        self.parameters().len()
    }

    /// Load the parameters from another `StateDict`.
    /// 
    /// This method will load all parameters with the same name from the `StateDict` into the module.
    fn load(&self, parameters: StateDict) {
        for (name, other_parameter) in parameters {
            if let Some(parameter) = self.parameters().get(&name) {
                *parameter.lock() = other_parameter.lock().shallow_clone();
            }
        }
    }

    /// Returns all trainable parameters that is not freezed.
    fn training_parameters(&self) -> Vec<TensorCell> {
        self.parameters()
            .into_values()
            .filter(|tensor| tensor.lock().requires_grad())
            .collect()
    }

    /// Freeze the trainable parameters of the module.
    fn freeze(&self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(false);
            });
        }
    }

    /// Unfreeze the trainable parameters of the module.
    fn unfreeze(&self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(true);
            });
        }
    }

    /// Clear the gradients of the trainable parameters of the module.
    fn zero_grad(&self) {
        self.parameters().values().for_each(|param| {
            let mut param = param.lock();
            param.zero_grad();
        });
    }

    /// Initialize the trainable parameters of the module, with a certain distribution from `tch::nn::Init`.
    fn init(&self, init: tch::nn::Init) {
        no_grad(|| {
            for parameter in self.parameters().values() {
                parameter.lock().init(init);
            }
        });
    }
}

/// The wrapper for a trainable module. Any module that implements `Trainable` should be wrapped in this.
///
/// You can explicitly visit the underlying module with `.module`.
///
/// A [Mod] and its underlying module both implement `Trainable`, so you can call methods like `parameters` on a [Mod]. The difference is that calling such methods on a [Mod] will also take effect on the child modules of the underlying module.
///
/// For example, if you call `freeze` on a [Mod], the trainable parameters of the underlying module and its child modules will be freezed. However, if you call `freeze` on the underlying module, only the trainable parameters of the underlying module will be freezed.
#[derive(Debug)]
pub struct Mod<T: Trainable + ?Sized> {
    pub arc: Arc<ModData<T>>,
}

/// The data of a [Mod], which stores some states and metadata of the module.
#[derive(Debug)]
pub struct ModData<T: Trainable + ?Sized> {
    pub parent: RwLock<Option<Weak<ModData<dyn Trainable>>>>,
    pub children: RwLock<BTreeMap<String, Mod<dyn Trainable>>>,
    pub device: RwLock<Device>,
    pub mode: RwLock<ModuleMode>,
    pub module: T,
}

impl<T: Trainable + ?Sized> Clone for Mod<T> {
    fn clone(&self) -> Self {
        Self {
            arc: Arc::clone(&self.arc),
        }
    }
}

impl<T: Trainable + ?Sized> Deref for ModData<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl<T: Trainable + ?Sized> Deref for Mod<T> {
    type Target = ModData<T>;

    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T, U> CoerceUnsized<ModData<U>> for ModData<T>
where
    T: CoerceUnsized<U> + Trainable + ?Sized,
    U: Trainable + ?Sized,
{
}

impl<T, U> CoerceUnsized<Mod<U>> for Mod<T>
where
    T: Unsize<U> + Trainable + ?Sized,
    U: Trainable + ?Sized,
{
}

impl<T: Trainable + 'static> Mod<T> {
    /// Create a [Mod] wrapped module, and update the parent of child modules.
    pub fn new(module: T) -> Mod<T> {
        let this = Mod {
            arc: Arc::new(ModData {
                parent: RwLock::new(None),
                children: RwLock::new(module.child_modules()),
                device: RwLock::new(Device::Cpu),
                mode: RwLock::new(ModuleMode::Train),
                module,
            }),
        };
        this.children.write().iter_mut().for_each(|(_, child)| {
            child
                .parent
                .write()
                .replace(Arc::downgrade(&(this.arc.clone() as _)));
        });
        this
    }
}

impl<T: Trainable + ?Sized> Mod<T> {
    /// Move the parameters of the module to a certain device.
    pub fn to(self, device: Device) -> Self
    where
        Self: Sized,
    {
        self.parameters()
            .values()
            .chain(self.static_tensors().values())
            .for_each(|param| {
                let mut param = param.lock();
                let requires_grad = param.requires_grad();
                no_grad(|| {
                    *param = param.to(device).set_requires_grad(requires_grad);
                })
            });
        *self.device.write() = device;
        self
    }

    /// Change the mode of the module to `Train`.
    pub fn train(&self, affect_children: bool) {
        *self.mode.write() = ModuleMode::Train;
        if affect_children {
            self.children
                .read()
                .values()
                .for_each(|child| child.train(affect_children));
        }
    }

    /// Change the mode of the module to `Eval`.
    pub fn eval(&self, affect_children: bool) {
        *self.mode.write() = ModuleMode::Eval;
        if affect_children {
            self.children
                .read()
                .values()
                .for_each(|child| child.eval(affect_children));
        }
    }

    /// Load parameters from a numpy .npz file. This method won't load static tensors.
    /// 
    /// The tensors in the file should be named as the path to them.
    /// 
    /// For example, a model architecture like this:
    /// 
    /// ```text
    /// SomeModule {
    ///     "layer1": Linear {
    ///         "linear_weight": TensorCell,
    ///         "linear_bias": TensorCell,
    ///     },
    ///     "layer2": Linear {
    ///         "linear_weight": TensorCell,
    ///         "linear_bias": TensorCell,
    ///     },
    /// }
    /// ```
    /// 
    /// Should be saved in a .npz file like this:
    /// 
    /// ```text
    /// layer1.linear_weight.npy
    /// layer1.linear_bias.npy
    /// layer2.linear_weight.npy
    /// layer2.linear_bias.npy
    /// ```
    pub fn load_npz<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        self.load(
            Tensor::read_npz(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        );
        Ok(())
    }

    /// Load parameters from a .ot file. This type of file is used by OpenTorch. It's also the default format used by `StateDict::save`. This method won't load static tensors.
    pub fn load_ot<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        Ok(self.load(
            Tensor::load_multi(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }
}

impl<T: Trainable + ?Sized> Trainable for Mod<T> {
    /// Returns all trainable parameters in the module, including the parameters in child modules. The parameters are stored in a `BTreeMap` with their path as keys.
    /// 
    /// For example, a model architecture like this:
    /// 
    /// ```text
    /// SomeModule {
    ///     "layer1": Linear {
    ///         "linear_weight": TensorCell,
    ///         "linear_bias": TensorCell,
    ///     },
    ///     "layer2": Linear {
    ///         "linear_weight": TensorCell,
    ///         "linear_bias": TensorCell,
    ///     },
    /// }
    /// ```
    /// 
    /// will return a `BTreeMap` like this:
    /// 
    /// ```text
    /// BTreeMap {
    ///     "layer1.linear_weight": TensorCell,
    ///     "layer1.linear_bias": TensorCell,
    ///     "layer2.linear_weight": TensorCell,
    ///     "layer2.linear_bias": TensorCell,
    /// }
    /// ```
    fn parameters(&self) -> StateDict {
        let mut parameters = self.module.parameters();
        for (name, child) in self.children.read().iter() {
            for (child_name, child_parameter) in child.parameters() {
                parameters.insert(format!("{}.{}", name, child_name), child_parameter);
            }
        }
        parameters
    }

    /// Returns all static tensors in the module, including the static tensors in child modules. The static tensors are stored in a `BTreeMap` with their path as keys.
    fn static_tensors(&self) -> StateDict {
        let mut static_tensors = self.module.static_tensors();
        for (name, child) in self.children.read().iter() {
            for (child_name, child_static_tensor) in child.static_tensors() {
                static_tensors.insert(format!("{}.{}", name, child_name), child_static_tensor);
            }
        }
        static_tensors
    }
}

/// A module is a neural network layer, which can be seen as a function from `Tensor` to `Tensor`, with some trainable parameters.
pub trait Module: Trainable {
    /// The forward function for Module.
    fn forward(&self, input: &Tensor) -> Tensor;
}

/// A module without trainable parameters.
pub trait NonParameterModule: Module {}

impl<T: NonParameterModule> Trainable for T {}
