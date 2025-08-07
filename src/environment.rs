use std::collections::HashMap;
use crate::runtime::RuntimeVal;
use crate::types::ValueType;

#[derive(Debug, Clone)]
pub struct Environment {
    parent: Option<Box<Environment>>,
    variables: HashMap<String, Box<dyn RuntimeVal + Send + Sync>>,
    var_types: HashMap<String, Option<ValueType>>,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            parent: None,
            variables: HashMap::new(),
            var_types: HashMap::new(),
        }
    }
}

impl Environment {
    pub fn new(parent: Option<Box<Environment>>) -> Self {
        Environment {
            parent,
            variables: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    pub fn find_var_in_chain(&mut self, var_name: &str) -> Option<&mut Environment> {
        if self.variables.contains_key(var_name) {
            return Some(self);
        }
        if let Some(parent) = &mut self.parent {
            return parent.find_var_in_chain(var_name);
        }
        None
    }

    pub fn get_global_env(&mut self) -> &mut Environment {
        if self.parent.is_none() {
            self
        } else {
            self.parent.as_mut().unwrap().get_global_env()
        }
    }

    pub fn set_var(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>, declared_type: Option<ValueType>) -> Box<dyn RuntimeVal + Send + Sync> {
        // Type checking pour les déclarations
        if let Some(ty) = declared_type.clone() {
            self.var_types.insert(var_name.clone(), Some(ty));
        }

        // Si la variable existe déjà dans cet environnement, on la met à jour
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
            self.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        // Sinon, c'est une nouvelle déclaration dans cet environnement
        self.var_types.insert(var_name.clone(), declared_type);
        self.variables.insert(var_name, new_value.clone());
        new_value
    }
    pub fn assign_var(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>) -> Box<dyn RuntimeVal + Send + Sync> {
        if self.variables.contains_key(&var_name) {
            if let Some(Some(expected)) = self.var_types.get(&var_name) {
                let actual = new_value.value_type()
                    .expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!(
                        "Type error: variable `{}` declared as `{:?}` but assigned `{:?}`",
                        var_name, expected, actual
                    );
                }
            }
            self.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        if let Some(env_with_var) = self.find_var_in_chain(&var_name) {
            if let Some(Some(expected)) = env_with_var.var_types.get(&var_name) {
                let actual = new_value.value_type()
                    .expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!(
                        "Type error: variable `{}` declared as `{:?}` but assigned `{:?}`",
                        var_name, expected, actual
                    );
                }
            }
            env_with_var.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        let global_env = self.get_global_env();
        if global_env.variables.contains_key(&var_name) {
            if let Some(Some(expected)) = global_env.var_types.get(&var_name) {
                let actual = new_value.value_type()
                    .expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!(
                        "Type error: variable `{}` declared as `{:?}` but assigned `{:?}`",
                        var_name, expected, actual
                    );
                }
            }
            global_env.variables.insert(var_name, new_value.clone());
            return new_value;
        }

        panic!("Cannot assign to undeclared variable '{}'", var_name);
    }
    pub fn lookup_var(&mut self, var_name: String) -> Box<dyn RuntimeVal + Send + Sync> {
        let env = self.resolve(&var_name);
        env.variables.get(&var_name).unwrap().clone()
    }
    pub fn resolve(&mut self, var_name: &str) -> &mut Environment {
        if self.variables.contains_key(var_name) {
            self
        } else if let Some(ref mut parent_env) = self.parent {
            parent_env.resolve(var_name)
        } else {
            panic!("Cannot resolve '{}' as it does not exist.", var_name);
        }
    }
}
