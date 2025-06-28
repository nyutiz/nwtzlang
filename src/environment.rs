use std::collections::HashMap;
use crate::runtime::RuntimeVal;
use crate::types::ValueType;

#[derive(Debug, Clone)]
pub struct Environment {
    parent: Option<Box<Environment>>,
    variables: HashMap<String, Box<dyn RuntimeVal + Send + Sync>>,
    var_types: HashMap<String, Option<ValueType>>,
}

impl Environment {
    pub fn new(parent: Option<Box<Environment>>) -> Self {
        Environment {
            parent,
            variables: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    pub fn set_var(&mut self, var_name: String, new_value: Box<dyn RuntimeVal + Send + Sync>, declared_type: Option<ValueType>, ) -> Box<dyn RuntimeVal + Send + Sync> {
        if let Some(ty) = declared_type.clone() {
            self.var_types.insert(var_name.clone(), Some(ty));
        }

        if self.var_types.contains_key(&var_name) {
            if let Some(expected) = self.var_types.get(&var_name).unwrap() {
                let actual = new_value
                    .value_type()
                    .expect("RuntimeVal should always have a type");
                if &actual != expected {
                    panic!(
                        "Type error: variable `{}` declared as `{:?}` but assigned `{:?}`",
                        var_name, expected, actual
                    );
                }
            }
            self.variables.insert(var_name.clone(), new_value.clone());
        }
        else if let Some(parent) = &mut self.parent {
            return parent.set_var(var_name.clone(), new_value.clone(), declared_type);
        }
        else {
            self.var_types.insert(var_name.clone(), None);
            self.variables.insert(var_name.clone(), new_value.clone());
        }

        new_value
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
