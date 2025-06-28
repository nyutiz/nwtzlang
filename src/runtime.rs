use std::any::Any;
use std::fmt::Debug;
use crate::ast::StringVal;
use crate::types::{ArrayVal, BooleanVal, FunctionVal, IntegerVal, NullVal, ObjectVal, ValueType};
use crate::types::ValueType::{Function, NativeFn};

impl RuntimeVal for NullVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for IntegerVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for StringVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ArrayVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for crate::types::NativeFnValue {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(NativeFn)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for FunctionVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(Function)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for BooleanVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ObjectVal {
    fn value_type(&self) -> Option<ValueType> {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait RuntimeValClone {
    fn clone_box(&self) -> Box<dyn RuntimeVal + Send + Sync>;
}

impl<T> RuntimeValClone for T
where
    T: 'static + RuntimeVal + Clone,
{
    fn clone_box(&self) -> Box<dyn RuntimeVal + Send + Sync> {
        Box::from(self.clone())
    }
}

impl Clone for Box<dyn RuntimeVal + Send + Sync> {
    fn clone(&self) -> Box<dyn RuntimeVal + Send + Sync> {
        self.clone_box()
    }
}

pub trait RuntimeVal: Debug + RuntimeValClone + Send + Sync   {
    fn value_type(&self) -> Option<ValueType>;
    fn as_any(&self) -> &dyn Any;
}
