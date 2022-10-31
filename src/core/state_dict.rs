use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    ops::Deref,
    path::Path,
    sync::{Arc, RwLock, RwLockReadGuard, Weak},
};

use anyhow::{anyhow, Result, Ok};
use itertools::Itertools;
use tch::{no_grad, Tensor};

use super::{TensorCell, Cellable};

/// Indicates the current node is a leaf node (i.e. a [`TensorCell`]) or a child tree (i.e. another [`StateDict`]).
/// 
/// A `StaticTensor` means a Tensor that is not a parameter of a [`Module`]. It is used to store some const data independent to input, and need to be in the same device as the module.
/// You won't get a `StaticTensor` from `StateDict` when use call method such as `to_map` or `to_vec`. `StaticTensor` also won't be loaded in `load` or `from_map` method.
#[derive(Debug, Clone)]
pub enum StateValue {
    Tensor(TensorCell),
    StaticTensor(TensorCell),
    ChildStateDict(StateDict),
}

/// A [`StateDict`] is a tree of [`Tensor`]s. It stores the parameters of a model, including each parameter itself and the path to it.
/// 
/// For example, a [`StateDict`] of a model with 2 layers may look like this:
/// 
/// ```text
/// StateDict {
///     "layer1": StateDict {
///         "weight": TensorCell,
///         "bias": TensorCell,
///     },
///     "layer2": StateDict {
///         "weight": TensorCell,
///         "bias": TensorCell,
///     },
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StateDict {
    arc: Arc<StateDictData>,
}

/// The actual data of a [`StateDict`]. It is wrapped in an [`Arc`] to allow sharing between multiple [`StateDict`]s.
#[derive(Debug)]
pub struct StateDictData {
    pub name: RwLock<String>,
    pub parent: RwLock<Weak<StateDictData>>,
    pub parameters: RwLock<BTreeMap<String, StateValue>>,
}

impl Deref for StateDict {
    type Target = Arc<StateDictData>;

    fn deref(&self) -> &Self::Target {
        &self.arc
    }
}

impl StateDict {
    /// Creates an empty [`StateDict`].
    pub fn new() -> Self {
        let data = StateDictData {
            name: RwLock::new("".to_owned()),
            parent: RwLock::new(Weak::new()),
            parameters: RwLock::new(BTreeMap::new()),
        };
        Self {
            arc: Arc::new(data),
        }
    }

    /// Get a clone of the pointer to the inner [`StateDictData`].
    pub fn arc(&self) -> Arc<StateDictData> {
        self.arc.clone()
    }

    /// Builds a [`StateDict`] from a [`BTreeMap`] of [`TensorCell`]s. This method won't build static tensors.
    /// 
    /// To build a [`StateDict`] like this:
    /// 
    /// ```text
    /// StateDict {
    ///     "layer1": StateDict {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    ///     "layer2": StateDict {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    /// }
    /// ```
    /// 
    /// You should pass a [`BTreeMap`] like this:
    /// 
    /// ```text
    /// BTreeMap {
    ///     "layer1.weight": TensorCell,
    ///     "layer1.bias": TensorCell,
    ///     "layer2.weight": TensorCell,
    ///     "layer2.bias": TensorCell,
    /// }
    /// ```
    pub fn from_map(parameters: BTreeMap<String, TensorCell>) -> Self {
        let this = Self::new();
        let parameter_map = parameters;
        let mut parameters = BTreeMap::new();
        let mut current_child: Option<BTreeMap<String, TensorCell>> = None;
        let mut current_child_name = "".to_owned();
        for (key, value) in parameter_map {
            let mut split = key.split(".");
            let first = split.next().unwrap();
            if split.next().is_none() {
                parameters.insert(first.to_owned(), StateValue::Tensor(value.clone()));
            } else {
                if let Some(child) = current_child {
                    if first != current_child_name {
                        let child = StateDict::from_map(child);
                        *child.parent.write().unwrap() = Arc::downgrade(&this);
                        parameters.insert(current_child_name, StateValue::ChildStateDict(child));
                        current_child = None;
                        current_child_name = "".to_owned();
                    } else {
                        current_child = Some(child)
                    }
                }
                if current_child.is_none() {
                    current_child = Some(BTreeMap::new());
                    current_child_name = first.to_owned();
                }
                current_child
                    .as_mut()
                    .unwrap()
                    .insert(key.split(".").skip(1).join("."), value);
            }
        }
        if let Some(child) = current_child {
            let child = StateDict::from_map(child);
            *child.parent.write().unwrap() = Arc::downgrade(&this);
            parameters.insert(
                current_child_name.to_owned(),
                StateValue::ChildStateDict(child),
            );
        }
        *this.parameters.write().unwrap() = parameters;
        this
    }

    /// Loads a [`StateDict`] from a numpy .npz file. This method won't load static tensors.
    /// 
    /// The tensors in the file should be named as the path to them in the [`StateDict`].
    /// 
    /// For example, a [`StateDict`] like this:
    /// 
    /// ```text
    /// StateDict {
    ///     "layer1": StateDict {
    ///         "weight": TensorCell,
    ///         "bias": TensorCell,
    ///     },
    ///     "layer2": StateDict {
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
    pub fn from_npz<T: AsRef<Path>>(path: T) -> Result<StateDict> {
        Ok(Self::from_map(
            Tensor::read_npz(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }

    /// Loads a [`StateDict`] from a .ot file. This type of file is used by OpenTorch. It's also the default format used by `StateDict::save`. This method won't load static tensors.
    pub fn from_ot<T: AsRef<Path>>(path: T) -> Result<StateDict> {
        Ok(Self::from_map(
            Tensor::load_multi(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }

    /// Append a child [`StateDict`] to this [`StateDict`].
    pub fn append_child(&mut self, module_name: String, child: StateDict) {
        *child.parent.write().unwrap() = Arc::downgrade(&self.arc());
        self.parameters
            .write()
            .unwrap()
            .insert(module_name, StateValue::ChildStateDict(child));
    }

    /// Append a child `Tensor` to this [`StateDict`].
    pub fn append_tensor(&mut self, name: String, tensor: TensorCell) {
        self.parameters
            .write()
            .unwrap()
            .insert(name, StateValue::Tensor(tensor));
    }

    /// Append a child `StaticTensor` to this [`StateDict`].
    pub fn append_static_tensor(&mut self, name: String, tensor: TensorCell) {
        self.parameters
            .write()
            .unwrap()
            .insert(name, StateValue::StaticTensor(tensor));
    }
}

impl StateDictData {
    /// Get the path to this [`StateDict`] in the [`StateDict`] tree.
    pub fn path(&self) -> String {
        if let Some(parent) = self.parent.read().unwrap().upgrade() {
            format!("{}.{}", parent.path(), self.name.read().unwrap())
        } else {
            "root".to_owned()
        }
    }

    /// Get a [`RwLockReadGuard`] to the [`BTreeMap`] of parameters.
    pub fn parameters(&self) -> RwLockReadGuard<'_, BTreeMap<String, StateValue>> {
        self.parameters.read().unwrap()
    }

    /// Get a [`TensorCell`] from the [`StateDict`]. This method can only get tensors that are in the current [`StateDict`], not in child [`StateDict`]s.
    pub fn tensor(&self, key: &str) -> Result<TensorCell> {
        match self.parameters().get(key) {
            Some(StateValue::Tensor(tensor)) => Ok(tensor.clone()),
            _ => Err(anyhow!("No such parameter: {} in {}", key, self.path())),
        }
    }

    /// Get a child [`StateDict`] from the [`StateDict`]. This method can only get child [`StateDict`]s, not tensors, and not child [`StateDict`]s of child [`StateDict`]s.
    pub fn child_state_dict(&self, module_name: &str) -> Result<StateDict> {
        match self.parameters().get(module_name) {
            Some(StateValue::ChildStateDict(state_dict)) => Ok(state_dict.clone()),
            _ => Err(anyhow!(
                "No such module: {} in {}",
                module_name,
                self.path()
            )),
        }
    }

    /// Load a [`StateDict`] from another [`StateDict`]. This method only load the [`Tensor`], but not the entire [`StateDict`] architecture. Any parameter with the same path will be loaded. Any parameter in one [`StateDict`] but not found in the other will be ignored (without changing its value).
    pub fn load(&self, state_dict: StateDict) {
        for (key, value) in &*state_dict.parameters() {
            match self.parameters().get(key) {
                Some(StateValue::Tensor(tensor)) => {
                    let mut tensor = tensor.lock();
                    if let StateValue::Tensor(value) = value {
                        let value = value.lock();
                        no_grad(|| {
                            *tensor = value.shallow_clone();
                        });
                    }
                }
                Some(StateValue::ChildStateDict(child_state_dict)) => {
                    child_state_dict.load(state_dict.child_state_dict(key).unwrap());
                }
                _ => (),
            }
        }
    }

    /// Convert the [`StateDict`] to a [`BTreeMap`]. The returned [`BTreeMap`] doesn't contain static tensors.
    pub fn to_map(&self) -> BTreeMap<String, TensorCell> {
        let mut parameters = BTreeMap::new();
        for (key, value) in &*self.parameters() {
            match value {
                StateValue::Tensor(tensor) => {
                    parameters.insert(key.clone(), tensor.clone());
                }
                StateValue::ChildStateDict(state_dict) => {
                    let map: BTreeMap<String, TensorCell> = state_dict.to_map();
                    for (child_key, child_value) in map {
                        parameters.insert(format!("{}.{}", key, child_key), child_value);
                    }
                }
                _ => (),
            }
        }
        parameters
    }

    /// Convert the [`StateDict`] to a [`Vec<TensorCell>`]. This method doesn't save the information about the path to parameters, so it should only be used when you need to execute an operation to all parameters as stream. The returned [`Vec`] doesn't contain static tensors.
    pub fn to_vec(&self) -> Vec<TensorCell> {
        let mut parameters = Vec::new();
        for (_, value) in &*self.parameters() {
            match value {
                StateValue::Tensor(tensor) => {
                    parameters.push(tensor.clone());
                }
                StateValue::ChildStateDict(state_dict) => {
                    let vec: Vec<TensorCell> = state_dict.to_vec();
                    parameters.extend(vec);
                },
                _ => (),
            }
        }
        parameters
    }

    /// Get all static tensors in the [`StateDict`].
    pub fn static_tensors(&self) -> Vec<TensorCell> {
        let mut parameters = Vec::new();
        for (_, value) in &*self.parameters() {
            match value {
                StateValue::StaticTensor(tensor) => {
                    parameters.push(tensor.clone());
                }
                StateValue::ChildStateDict(state_dict) => {
                    let vec: Vec<TensorCell> = state_dict.static_tensors();
                    parameters.extend(vec);
                },
                _ => (),
            }
        }
        parameters
    }
}

impl Display for StateDict {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let map = self.to_map();
        for (key, value) in map {
            writeln!(f, "{}: {:?}", key, value.lock())?;
        }
        std::fmt::Result::Ok(())
    }
}
