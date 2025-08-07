use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use crate::ast::Stmt;
use crate::environment::Environment;
use crate::runtime::RuntimeVal;

pub static RESERVED_NAMES: Lazy<HashSet<String>> = Lazy::new(|| {
    ValueType::iter().map(|vt| format!("{:?}", vt)).collect()
});

#[derive(Debug, Clone, EnumIter, PartialEq)]
pub enum ValueType {
    Null,
    Integer,
    Boolean,
    Object,
    Array,
    NativeFn,
    Function,
    String,
}

#[derive(Debug, Clone)]
pub struct NullVal {
    pub r#type: Option<ValueType>,
}

#[derive(Debug, Clone)]
pub struct ArrayVal {
    pub r#type: Option<ValueType>,
    pub elements: Arc<Mutex<Vec<Box<dyn RuntimeVal + Send + Sync>>>>,
}



#[derive(Debug, Clone)]
pub struct IntegerVal {
    pub r#type: Option<ValueType>,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct BooleanVal {
    pub r#type: Option<ValueType>,
    pub value: bool,
}


#[derive(Debug, Clone)]
pub struct ObjectVal {
    pub r#type: Option<ValueType>,
    pub properties:  Arc<Mutex<HashMap<String, Box<dyn RuntimeVal + Send + Sync>>>>,
}



#[derive(Debug, Clone)]
pub struct FunctionVal {
    pub r#type: Option<ValueType>,
    pub name: String,
    pub parameters: Vec<String>,
    pub declaration_env: Arc<Mutex<Environment>>,
    pub body: Arc<Vec<Box<dyn Stmt>>>,
}

pub type FunctionCall =
Arc<dyn Fn(Vec<Box<dyn RuntimeVal + Send + Sync>>, &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> + Send + Sync>;
#[derive(Clone)]
pub struct NativeFnValue {
    pub r#type: Option<ValueType>,
    pub call: FunctionCall,
}

impl Debug for NativeFnValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "< fn >")
    }
}