use std::any::Any;
use std::fmt::Debug;
use crate::environment::Environment;
use crate::evaluator::evaluate;
use crate::types::{FunctionVal, ValueType};

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {

    // Statements
    Program,
    VariableDeclaration,
    FunctionDeclaration,
    ImportAst,
    IfStatement,
    ForStatement,

    // Expressions
    AssignmentExpr,
    MemberExpr,
    CallExpr,

    // Literals
    Property,
    ObjectLiteral,
    NumericLiteral,
    NullLiteral,
    BooleanLiteral,
    Identifier,
    BinaryExpression,
    StringLiteral,
    ArrayLiteral,

}


impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    pub fn new() -> Self {
        Self {
            kind: NodeType::Program,
            body: Vec::new(),
        }
    }

    pub fn has_x(&mut self, name: &str, target_type: NodeType, env: &mut Environment) -> bool {
        if name.is_empty() {
            panic!("Name from has_x is empty");
        }

        let mut found = false;

        for stmt in &self.body {
            match stmt.kind() {
                NodeType::FunctionDeclaration => {
                    if target_type == NodeType::FunctionDeclaration {
                        if let Some(func_decl) = stmt.as_any().downcast_ref::<FunctionDeclaration>() {
                            if func_decl.name == name {
                                found = true;
                            }
                        }
                    }
                    evaluate(stmt.clone(), env);
                },
                NodeType::VariableDeclaration => {
                    if target_type == NodeType::VariableDeclaration {
                        if let Some(var_decl) = stmt.as_any().downcast_ref::<VariableDeclaration>() {
                            if var_decl.name == name {
                                found = true;
                            }
                        }
                    }
                    evaluate(stmt.clone(), env);
                },
                NodeType::ImportAst => {
                    evaluate(stmt.clone(), env);
                },
                _ => {}
            }
        }

        found
    }
    pub fn merge_imports(mut self) -> Self {
        let mut merged_body = Vec::new();

        for stmt in self.body.into_iter() {
            if let Some(import_node) = stmt.as_any().downcast_ref::<ImportAst>() {
                merged_body.extend(import_node.body.clone());
            } else {
                merged_body.push(stmt);
            }
        }

        self.body = merged_body;
        self
    }
}
impl VariableDeclaration {
    pub fn new(identifier: String, value: Option<Box<dyn Stmt>>, r#type :Option<ValueType>) -> Self {
        Self {
            kind: NodeType::VariableDeclaration,
            r#type,
            name: identifier,
            value,
        }
    }
}


pub trait Stmt: Debug + StmtClone + Send + Sync {
    fn kind(&self) -> NodeType;
    fn value(&self) -> Option<String>;
    fn as_any(&self) -> &dyn Any;
}
impl dyn Stmt {
    pub fn downcast<T: 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Stmt>>
    where
        T: Stmt,
    {
        if self.as_any().is::<T>() {
            let raw = Box::into_raw(self);
            Ok(unsafe { Box::from_raw(raw as *mut T) })
        } else {
            Err(self)
        }
    }
}

pub trait StmtClone {
    fn clone_box(&self) -> Box<dyn Stmt>;
}

impl<T> StmtClone for T
where
    T: 'static + Stmt + Clone,
{
    fn clone_box(&self) -> Box<dyn Stmt> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Stmt> {
    fn clone(&self) -> Box<dyn Stmt> {
        self.clone_box()
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Property {
    pub kind: NodeType,
    pub key: String,
    pub value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct ObjectLiteral {
    pub kind: NodeType,
    pub properties: Vec<Property>,
}


#[derive(Debug, Clone, PartialEq)]
pub struct LiteralExpr {
    pub kind: NodeType,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifierExpr {
    pub kind: NodeType,
    pub name: String,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct NullLiteral {
    pub kind: NodeType,
    pub value: String,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct BinaryExpr {
    pub kind: NodeType,
    pub left: Box<dyn Stmt>,
    pub right: Box<dyn Stmt>,
    pub operator: String,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct AssignmentExpr {
    pub kind: NodeType,
    pub assigne: Box<dyn Stmt>,
    pub value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct MemberExpr {
    pub kind: NodeType,
    pub object: Box<dyn Stmt>,
    pub property: Box<dyn Stmt>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct CallExpr {
    pub kind: NodeType,
    pub caller: Box<dyn Stmt>,
    pub args: Vec<Box<dyn Stmt>>,
}
#[derive(Debug)]
#[derive(Clone)]
pub struct Program {
    pub kind: NodeType,
    pub body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct VariableDeclaration {
    pub kind: NodeType,
    pub r#type: Option<ValueType>,
    pub name: String,
    pub value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct FunctionDeclaration {
    pub kind: NodeType,
    pub parameters: Vec<String>,
    pub name: String,
    pub body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug, Clone)]
pub struct StringVal {
    pub r#type: Option<ValueType>,
    pub kind: NodeType,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct ForStatement {
    pub kind: NodeType,
    pub initializer: Option<Box<dyn Stmt>>,
    pub condition: Option<Box<dyn Stmt>>,
    pub increment: Option<Box<dyn Stmt>>,
    pub body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug, Clone)]
pub struct IfStatement {
    pub kind: NodeType,
    pub condition: Box<dyn Stmt>,
    pub then_branch: Vec<Box<dyn Stmt>>,
    pub else_branch: Option<Vec<Box<dyn Stmt>>>,
}
#[derive(Debug, Clone)]
pub struct ArrayLiteral {
    pub kind: NodeType,
    pub elements: Vec<Box<dyn Stmt>>,
}

#[derive(Debug, Clone)]
pub struct BooleanLiteral {
    pub kind: NodeType,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct ImportAst {
    pub kind: NodeType,
    pub body: Vec<Box<dyn Stmt>>,
}


impl Stmt for BooleanLiteral {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> { Some(self.value.to_string()) }
    fn as_any(&self) -> &dyn Any { self }
}


impl Stmt for IfStatement {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> { None }
    fn as_any(&self) -> &dyn Any { self }
}

impl Stmt for ForStatement {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> { None }
    fn as_any(&self) -> &dyn Any { self }
}

impl Stmt for ArrayLiteral {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> {None}
    fn as_any(&self) -> &dyn Any { self }
}
impl Stmt for FunctionDeclaration {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> {
        None
    }    fn as_any(&self) -> &dyn Any { self }
}

impl Stmt for FunctionVal {
    fn kind(&self) -> NodeType { NodeType::FunctionDeclaration }
    fn value(&self) -> Option<String> {
        None
    }    fn as_any(&self) -> &dyn Any { self }
}
impl Stmt for LiteralExpr {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Stmt for ObjectLiteral {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Stmt for ImportAst {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Stmt for StringVal {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        Some(self.value.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for MemberExpr {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for CallExpr {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for Property {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }

    fn value(&self) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for IdentifierExpr {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for BinaryExpr{
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for AssignmentExpr{
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for NullLiteral{
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        Option::from(self.value.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for Program {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for VariableDeclaration {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
    fn value(&self) -> Option<String> {
        None
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}