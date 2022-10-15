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

#[derive(Debug, Clone)]
pub enum StateValue {
    Tensor(TensorCell),
    ChildStateDict(StateDict),
}

#[derive(Debug, Clone)]
pub struct StateDict {
    arc: Arc<StateDictData>,
}

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

    pub fn arc(&self) -> Arc<StateDictData> {
        self.arc.clone()
    }

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

    pub fn from_npz<T: AsRef<Path>>(path: T) -> Result<StateDict> {
        Ok(Self::from_map(
            Tensor::read_npz(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }

    pub fn from_ot<T: AsRef<Path>>(path: T) -> Result<StateDict> {
        Ok(Self::from_map(
            Tensor::load_multi(path)?
                .into_iter()
                .map(|(key, tensor)| (key, tensor.cell()))
                .collect(),
        ))
    }

    pub fn append_child(&mut self, module_name: String, child: StateDict) {
        *child.parent.write().unwrap() = Arc::downgrade(&self.arc());
        self.parameters
            .write()
            .unwrap()
            .insert(module_name, StateValue::ChildStateDict(child));
    }
}

impl StateDictData {
    pub fn path(&self) -> String {
        if let Some(parent) = self.parent.read().unwrap().upgrade() {
            format!("{}.{}", parent.path(), self.name.read().unwrap())
        } else {
            "root".to_owned()
        }
    }

    pub fn parameters(&self) -> RwLockReadGuard<'_, BTreeMap<String, StateValue>> {
        self.parameters.read().unwrap()
    }

    pub fn tensor(&self, key: &str) -> Result<TensorCell> {
        match self.parameters().get(key) {
            Some(StateValue::Tensor(tensor)) => Ok(tensor.clone()),
            _ => Err(anyhow!("No such parameter: {} in {}", key, self.path())),
        }
    }

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

    pub fn load(&self, state_dict: StateDict) {
        for (key, value) in &*state_dict.parameters() {
            match self.parameters().get(key) {
                Some(StateValue::Tensor(tensor)) => {
                    let mut tensor = tensor.lock().unwrap();
                    if let StateValue::Tensor(value) = value {
                        let value = value.lock().unwrap();
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
            }
        }
        parameters
    }

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
                }
            }
        }
        parameters
    }
}

impl Display for StateDict {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let map = self.to_map();
        for (key, value) in map {
            writeln!(f, "{}: {:?}", key, value.lock().unwrap())?;
        }
        std::fmt::Result::Ok(())
    }
}
