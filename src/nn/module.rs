use std::{
    collections::BTreeMap,
    marker::Unsize,
    ops::{CoerceUnsized, Deref},
    sync::{Arc, Weak},
};

use parking_lot::RwLock;
use tch::{no_grad, Device, Tensor};

use crate::core::TensorCell;

pub type StateDict = BTreeMap<String, TensorCell>;
pub type ModuleDict = BTreeMap<String, Mod<dyn Trainable>>;

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
    fn load(&mut self, parameters: StateDict) {
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
    fn freeze(&mut self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(false);
            });
        }
    }

    /// Unfreeze the trainable parameters of the module.
    fn unfreeze(&mut self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            no_grad(|| {
                *tensor = tensor.set_requires_grad(true);
            });
        }
    }

    /// Move the parameters of the module to a certain device.
    fn to(self, device: Device) -> Self
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
        self
    }

    /// Clear the gradients of the trainable parameters of the module.
    fn zero_grad(&self) {
        self.parameters().values().for_each(|param| {
            let mut param = param.lock();
            param.zero_grad();
        });
    }

    /// Initialize the trainable parameters of the module, with a certain distribution from `tch::nn::Init`.
    fn init(&mut self, init: tch::nn::Init) {
        no_grad(|| {
            for parameter in self.parameters().values() {
                parameter.lock().init(init);
            }
        });
    }
}

#[derive(Debug)]
pub struct ModData<T: Trainable + ?Sized> {
    pub parent: RwLock<Option<Weak<ModData<dyn Trainable>>>>,
    pub children: RwLock<BTreeMap<String, Mod<dyn Trainable>>>,
    pub device: Device,
    pub module: T,
}

#[derive(Debug)]
pub struct Mod<T: Trainable + ?Sized> {
    pub arc: Arc<ModData<T>>,
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
                device: Device::Cpu,
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

impl<T: Trainable + ?Sized + Send + Sync> Trainable for Mod<T> {
    /// Returns all trainable parameters in the module, including the parameters in child modules.
    fn parameters(&self) -> StateDict {
        let mut parameters = self.module.parameters();
        for (name, child) in self.children.read().iter() {
            for (child_name, child_parameter) in child.parameters() {
                parameters.insert(format!("{}.{}", name, child_name), child_parameter);
            }
        }
        parameters
    }

    /// Returns all static tensors in the module, including the static tensors in child modules.
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

impl<T: NonParameterModule> Trainable for T {
    fn parameters(&self) -> StateDict {
        BTreeMap::new()
    }
}
