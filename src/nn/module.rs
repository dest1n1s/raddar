use std::{
    marker::Unsize,
    ops::{CoerceUnsized, Deref},
    path::Path,
    sync::{Arc, Weak},
};

use anyhow::Ok;
use linked_hash_map::LinkedHashMap;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    core::{Cellable, TensorCell, TensorNN, FromNpz, FromOt},
    util::DropGuard,
};

/// A `StateDict` is a collection of named tensors. It uses [LinkedHashMap] to preserve the insertion order of the tensors. This is useful when saving and loading the model.
/// 
/// A potentially substitute for `LinkedHashMap` is [IndexMap](https://docs.rs/indexmap/1.7.0/indexmap/map/struct.IndexMap.html). In comparision, `LinkedHashMap` can provide a more strict guarantee on the order of the tensors. For example, the order is preserved even when the tensors are removed from the [StateDict].
/// 
/// However, `LinkedHashMap` may cause a performance issue in iteration. This is because `LinkedHashMap` uses a doubly linked list to maintain the insertion order.
pub type StateDict<Ts: TensorNN> = LinkedHashMap<String, TensorCell<Ts>>;

/// A `TrainableDict` is a collection of named [Trainable]s. For the same reason as `StateDict`, it uses [LinkedHashMap] to preserve the insertion order of the `Trainable`s. See [StateDict] for more details.
pub type TrainableDict<Ts: TensorNN> = LinkedHashMap<String, Mod<dyn Trainable<Ts>, Ts>>;

/// A `ModuleDict` is a collection of named [Module]s. For the same reason as `StateDict`, it uses [LinkedHashMap] to preserve the insertion order of the `Module`s. See [StateDict] for more details.
/// 
/// It is recommended to use `ModuleDict` in the implemention of a `Module` to store child `Module`s with a repetitive pattern when you don't want an extra layer of abstraction.
pub type ModuleDict<Ts: TensorNN> = LinkedHashMap<String, Mod<dyn Module<Ts>, Ts>>;

#[derive(Debug, Clone, Copy)]
pub enum ModuleMode {
    Train,
    Eval,
}
/// A trait for anything that has trainable parameters.
pub trait Trainable<Ts: TensorNN>: std::fmt::Debug {
    /// Defines the trainable parameters of the module. This does not include the parameters in child modules.
    ///
    /// By default, this returns an empty map. If your module has trainable parameters, you should override this method.
    fn parameters(&self) -> StateDict<Ts> {
        LinkedHashMap::new()
    }

    /// Defines the static tensors of the module. This does not include the static tensors in child modules.
    ///
    /// By default, this returns an empty map. If your module has static tensors, you should override this method.
    fn static_tensors(&self) -> StateDict<Ts> {
        LinkedHashMap::new()
    }

    /// Defines the child modules of the module.
    ///
    /// By default, this returns an empty map. If your module has child modules, you should override this method.
    fn child_modules(&self) -> TrainableDict<Ts> {
        LinkedHashMap::new()
    }

    /// Returns the size of the parameters of the module.
    fn parameter_size(&self) -> usize {
        self.parameters().len()
    }

    /// Load the parameters from another `StateDict`.
    ///
    /// This method will load all parameters with the same name from the `StateDict` into the module.
    fn load(&self, parameters: StateDict<Ts>) {
        for (name, other_parameter) in parameters {
            if let Some(parameter) = self.parameters().get(&name) {
                *parameter.lock() = other_parameter.lock().copy();
            }
        }
    }

    /// Returns all trainable parameters that is not freezed.
    fn training_parameters(&self) -> Vec<TensorCell<Ts>> {
        self.parameters()
            .into_iter()
            .map(|(_, parameter)| parameter)
            .filter(|tensor| tensor.lock().requires_grad())
            .collect()
    }

    /// Freeze the trainable parameters of the module.
    fn freeze(&self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            let mut tensor = tensor.no_grad_mut();
            *tensor = tensor.set_requires_grad(false);
        }
    }

    /// Unfreeze the trainable parameters of the module.
    fn unfreeze(&self) {
        for tensor in self.parameters().values() {
            let mut tensor = tensor.lock();
            let mut tensor = tensor.no_grad_mut();
            *tensor = tensor.set_requires_grad(true);
        }
    }

    /// Clear the gradients of the trainable parameters of the module.
    fn zero_grad(&self) {
        self.parameters().values().for_each(|param| {
            let mut param = param.lock();
            param.zero_grad();
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
pub struct Mod<T: Trainable<Ts> + ?Sized, Ts: TensorNN> {
    pub arc: Arc<ModData<T, Ts>>,
}

/// The data of a [Mod], which stores some states and metadata of the module.
#[derive(Debug)]
pub struct ModData<T: Trainable<Ts> + ?Sized, Ts: TensorNN> {
    pub parent: RwLock<Option<Weak<ModData<dyn Trainable<Ts>, Ts>>>>,
    pub children: RwLock<LinkedHashMap<String, Mod<dyn Trainable<Ts>, Ts>>>,
    pub device: RwLock<Ts::Device>,
    pub mode: RwLock<ModuleMode>,
    pub module: RwLock<T>,
}

impl<T: Trainable<Ts> + ?Sized, Ts: TensorNN> Clone for Mod<T, Ts> {
    fn clone(&self) -> Self {
        Self {
            arc: Arc::clone(&self.arc),
        }
    }
}

impl<T: Trainable<Ts> + ?Sized, Ts: TensorNN> Deref for Mod<T, Ts> {
    type Target = ModData<T, Ts>;

    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl<T, U, Ts> CoerceUnsized<Mod<U, Ts>> for Mod<T, Ts>
where
    T: Unsize<U> + Trainable<Ts> + ?Sized,
    U: Trainable<Ts> + ?Sized,
    Ts: TensorNN,
{
}

impl<T: Trainable<Ts> + 'static, Ts: TensorNN> Mod<T, Ts> {
    /// Create a [Mod] wrapped module, and update the parent of child modules.
    pub fn new(module: T) -> Mod<T, Ts> {
        let this = Mod {
            arc: Arc::new(ModData {
                parent: RwLock::new(None),
                children: RwLock::new(module.child_modules()),
                device: RwLock::new(Ts::Device::default()),
                mode: RwLock::new(ModuleMode::Train),
                module: RwLock::new(module),
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

    /// Get a mutable reference to the underlying module.
    ///
    /// This method will update the parent of child modules when the mutable reference is dropped. The device of the child modules will also be checked and updated to the device of this module.
    pub fn module_mut(&self) -> DropGuard<RwLockWriteGuard<T>> {
        let this = self.clone();
        let module = self.module.write();

        // Use a [DropGuard] to update the parent of child modules before the mutable reference is dropped, so that when the write lock is released, the parent of child modules is already up-to-date.
        DropGuard::new(
            module,
            Box::new(move |module| {
                let mut children = this.children.write();
                let new_children = module.child_modules();
                *children = new_children;
                children.iter_mut().for_each(|(_, child)| {
                    // Check and update the device of child modules.
                    if child.device() != this.device() {
                        child.to_(this.device());
                    }

                    // Update the parent of child modules.
                    child
                        .parent
                        .write()
                        .replace(Arc::downgrade(&(this.arc.clone() as _)));
                });
            }),
        )
    }
}

impl<T: Trainable<Ts> + ?Sized, Ts: TensorNN> Mod<T, Ts> {
    /// Get a reference to the underlying module.
    pub fn module(&self) -> RwLockReadGuard<T> {
        self.module.read()
    }

    /// Move the parameters of the module to a certain device, and return a new [Mod].
    pub fn to(self, device: Ts::Device) -> Self
    where
        Self: Sized,
    {
        self.parameters()
            .values()
            .chain(self.static_tensors().values())
            .for_each(|param| {
                let mut param = param.lock();
                let mut param = param.no_grad_mut();
                *param = param.to_device(device);
            });

        self._set_device(device);
        self
    }

    /// Move the parameters of the module to a certain device.
    pub fn to_(&self, device: Ts::Device) {
        self.parameters()
            .values()
            .chain(self.static_tensors().values())
            .for_each(|param| {
                let mut param = param.lock();
                let mut param = param.no_grad_mut();
                *param = param.to_device(device);
            });

        self._set_device(device);
    }

    /// Update the device state of the module and its child modules, without practically moving the parameters.
    fn _set_device(&self, device: Ts::Device) {
        *self.device.write() = device;
        self.children
            .write()
            .iter()
            .for_each(|(_, child)| child._set_device(device));
    }

    /// Get the device of the module.
    pub fn device(&self) -> Ts::Device {
        self.device.read().clone()
    }

    /// Change the mode of the module to `Train`.
    ///
    /// If `affect_children` is `true`, the mode of the child modules will also be changed to `Train`. Otherwise, the mode of the child modules will not be changed.
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
    ///
    /// If `affect_children` is `true`, the mode of the child modules will also be changed to `Eval`. Otherwise, the mode of the child modules will not be changed.
    pub fn eval(&self, affect_children: bool) {
        *self.mode.write() = ModuleMode::Eval;
        if affect_children {
            self.children
                .read()
                .values()
                .for_each(|child| child.eval(affect_children));
        }
    }

    /// Get the mode of the module.
    pub fn mode(&self) -> ModuleMode {
        self.mode.read().clone()
    }

    /// Get the parent of the module.
    pub fn parent(&self) -> Option<Mod<dyn Trainable<Ts>, Ts>> {
        self.parent
            .read()
            .as_ref()
            .and_then(|parent| parent.upgrade())
            .map(Mod::from)
    }

    /// Get the children of the module.
    pub fn children(&self) -> LinkedHashMap<String, Mod<dyn Trainable<Ts>, Ts>> {
        self.children.read().clone()
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
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    ///     "layer2": Linear {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    /// }
    /// ```
    ///
    /// Should be saved in a .npz file like this:
    ///
    /// ```text
    /// layer1.weight.npy
    /// layer1.bias.npy
    /// layer2.weight.npy
    /// layer2.bias.npy
    /// ```
    pub fn load_npz<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> where Ts: FromNpz {
        self.load(
            Ts::from_npz(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        );
        Ok(())
    }

    /// Load parameters from a .ot file. This type of file is used by OpenTorch. It's also the default format used by `StateDict::save`. This method won't load static tensors.
    pub fn load_ot<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> where Ts: FromOt {
        Ok(self.load(
            Ts::from_ot(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }
}

impl<T: Trainable<Ts> + ?Sized, Ts: TensorNN> Trainable<Ts> for Mod<T, Ts> {
    /// Returns all trainable parameters in the module, including the parameters in child modules. The parameters are stored in a `LinkedHashMap` with their path as keys.
    ///
    /// For example, a model architecture like this:
    ///
    /// ```text
    /// SomeModule {
    ///     "layer1": Linear {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    ///     "layer2": Linear {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    /// }
    /// ```
    ///
    /// will return a `LinkedHashMap` like this:
    ///
    /// ```text
    /// LinkedHashMap {
    ///     "layer1.weight": TensorCell,
    ///     "layer1.bias": TensorCell,
    ///     "layer2.weight": TensorCell,
    ///     "layer2.bias": TensorCell,
    /// }
    /// ```
    fn parameters(&self) -> StateDict<Ts> {
        let mut parameters = self.module().parameters();
        for (name, child) in self.children.read().iter() {
            for (child_name, child_parameter) in child.parameters() {
                parameters.insert(format!("{}.{}", name, child_name), child_parameter);
            }
        }
        parameters
    }

    /// Returns all static tensors in the module, including the static tensors in child modules. The static tensors are stored in a `LinkedHashMap` with their path as keys.
    fn static_tensors(&self) -> StateDict<Ts> {
        let mut static_tensors = self.module().static_tensors();
        for (name, child) in self.children.read().iter() {
            for (child_name, child_static_tensor) in child.static_tensors() {
                static_tensors.insert(format!("{}.{}", name, child_name), child_static_tensor);
            }
        }
        static_tensors
    }
}

impl<T: Trainable<Ts> + ?Sized, Ts: TensorNN> From<Arc<ModData<T, Ts>>> for Mod<T, Ts> {
    fn from(data: Arc<ModData<T, Ts>>) -> Self {
        Self { arc: data }
    }
}

/// A module is a neural network layer, which can be seen as a function from `Tensor` to `Tensor`, with some trainable parameters.
pub trait Module<Ts: TensorNN, InputType = Ts, OutputType = Ts>: Trainable<Ts> {
    /// The forward function for Module.
    fn forward(&self, input: &InputType) -> OutputType;
}

impl<T, U, Ts: TensorNN> Fn<(&T,)> for Mod<dyn Module<Ts, T, U>, Ts> {
    extern "rust-call" fn call(&self, input: (&T,)) -> U {
        self.module().forward(input.0)
    }
}

impl<T, U, Ts: TensorNN> FnMut<(&T,)> for Mod<dyn Module<Ts, T, U>, Ts> {
    extern "rust-call" fn call_mut(&mut self, input: (&T,)) -> U {
        self.module().forward(input.0)
    }
}

impl<T, U, Ts: TensorNN> FnOnce<(&T,)> for Mod<dyn Module<Ts, T, U>, Ts> {
    type Output = U;

    extern "rust-call" fn call_once(self, input: (&T,)) -> U {
        self.module().forward(input.0)
    }
}

// /// A module without trainable parameters.
// pub trait NonParameterModule<Ts: TensorNN>: Module<Ts> {}

// impl<T: NonParameterModule<Ts>, Ts: TensorNN> Trainable<Ts> for T {}
