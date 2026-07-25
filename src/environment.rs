use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use crate::runtime::RuntimeVal;
use crate::types::ValueType;


pub type SharedEnv = Arc<Mutex<Environment>>;

#[derive(Debug, Clone)]
pub struct Environment {
    parent: Option<SharedEnv>,
    variables: HashMap<String, Box<dyn RuntimeVal + Send + Sync>>,
    var_types: HashMap<String, Option<ValueType>>,
    constants: HashSet<String>,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            parent: None,
            variables: HashMap::new(),
            var_types: HashMap::new(),
            constants: HashSet::new(),
        }
    }
}

impl Environment {
    pub fn new(parent: Option<SharedEnv>) -> Self {
        Environment {
            parent,
            variables: HashMap::new(),
            var_types: HashMap::new(),
            constants: HashSet::new(),
        }
    }

    pub fn new_shared(parent: Option<SharedEnv>) -> SharedEnv {
        Arc::new(Mutex::new(Environment::new(parent)))
    }

    pub fn set_var(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>, declared_type: Option<ValueType>) -> Box<dyn RuntimeVal + Send + Sync> {
        self.declare(var_name, new_value, declared_type, false)
    }

    pub fn set_const(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>, declared_type: Option<ValueType>) -> Box<dyn RuntimeVal + Send + Sync> {
        self.declare(var_name, new_value, declared_type, true)
    }

    fn declare(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>, declared_type: Option<ValueType>, is_const: bool) -> Box<dyn RuntimeVal + Send + Sync> {
        if let Some(ty) = declared_type.clone() {
            self.var_types.insert(var_name.clone(), Some(ty));
        }

        if self.var_types.contains_key(&var_name) {
            if let Some(expected) = self.var_types.get(&var_name).unwrap() {
                let actual = new_value.value_type()
                    .expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!(
                        "Type error: variable `{}` declared as `{:?}` but assigned `{:?}`",
                        var_name, expected, actual
                    );
                }
            }
            if is_const {
                self.constants.insert(var_name.clone());
            }
            self.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        self.var_types.insert(var_name.clone(), declared_type);
        if is_const {
            self.constants.insert(var_name.clone());
        }
        self.variables.insert(var_name, new_value.clone());
        new_value
    }

    pub fn assign_var(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>) -> Box<dyn RuntimeVal + Send + Sync> {
        if self.variables.contains_key(&var_name) {
            if self.constants.contains(&var_name) {
                panic!("Cannot assign to constant `{}`", var_name);
            }
            if let Some(Some(expected)) = self.var_types.get(&var_name) {
                let actual = new_value.value_type().expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!("Type error: variable `{}` declared as `{:?}` but assigned `{:?}`", var_name, expected, actual);
                }
            }
            self.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        if let Some(parent) = &self.parent {
            return parent.lock().unwrap().assign_var(var_name, new_value);
        }

        panic!("Cannot assign to undeclared variable '{}'", var_name);
    }

    pub fn lookup_var(&self, var_name: &str) -> Box<dyn RuntimeVal + Send + Sync> {
        if let Some(val) = self.variables.get(var_name) {
            return val.clone();
        }
        if let Some(parent) = &self.parent {
            return parent.lock().unwrap().lookup_var(var_name);
        }
        panic!("Cannot resolve '{}' as it does not exist.", var_name);
    }
}