#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

use std::any::Any;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::{fs, io};
use std::process::exit;
use std::rc::Rc;
use std::slice::SliceIndex;
use logos::Logos;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::nwtz::ValueType::{Boolean, NativeFn, Null, Number};
use std::sync::Arc;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use crate::nwtz::NodeType::Identifier;
use crate::nwtz::Token::{LBrace, RBrace, RBracket, Semicolon};

pub trait Stmt: Debug + StmtClone {
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


pub trait RuntimeValClone {
    fn clone_box(&self) -> Box<dyn RuntimeVal>;
}

impl<T> RuntimeValClone for T
where
    T: 'static + RuntimeVal + Clone,
{
    fn clone_box(&self) -> Box<dyn RuntimeVal> {
        Box::from(self.clone())
    }
}

impl Clone for Box<dyn RuntimeVal> {
    fn clone(&self) -> Box<dyn RuntimeVal> {
        self.clone_box()
    }
}

pub trait RuntimeVal: Debug + RuntimeValClone  {
    fn value_type(&self) -> ValueType;
    fn as_any(&self) -> &dyn Any;
}

impl Debug for NativeFnValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<native-fn>")
    }
}

#[derive(Clone, Debug, PartialEq, Logos)]
pub enum Token {
    #[regex(r"[ \t\n\r\f]+", logos::skip)]
    #[regex(r"//[^\n\r]*", logos::skip)]
    #[token("with")]
    With,
    #[token("obj")]
    Obj,
    #[token("fn")]
    Fn,
    #[token("impl")]
    Impl,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("new")]
    New,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("whl")]
    While,
    #[token("in")]
    In,
    #[token("for")]
    For,
    //#[token("log")]
    //Log,
    //#[token("main")]
    //Main,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("%")]
    Percent,
    #[token("=")]
    Equal,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token("/")]
    Slash,
    #[token(".")]
    Dot,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(">")]
    Greater,
    #[token("<")]
    Lower,
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice().to_string())]
    StringLiteral(String),
    #[regex(r"\d+\.\d+", |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[regex(r"\d+", |lex| lex.slice().parse::<i32>().unwrap())]
    Integer(i32),
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,
    EOF,
}

#[derive(Debug, Clone, EnumIter)]
pub enum ValueType {
    Null,
    Number,
    Boolean,
    Object,
    Array,
    NativeFn,
    Function,
    String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {

    // Statements
    Program,
    VariableDeclaration,
    FunctionDeclaration,
    ImportAst,
    IfStatement,

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

static RESERVED_NAMES: Lazy<HashSet<String>> = Lazy::new(|| {
    ValueType::iter().map(|vt| format!("{:?}", vt)).collect()
});

#[derive(Debug, Clone)]
pub struct Parameter {
    pub param_type: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Environment {
    parent: Option<Box<Environment>>,
    variables: HashMap<String, Box<dyn RuntimeVal>>,
}

#[derive(Debug, Clone)]
pub struct ImportAst {
    pub kind: NodeType,
    pub body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug, Clone)]
pub struct NullVal {
    pub r#type: ValueType,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct ArrayVal {
    pub r#type: ValueType,
    pub elements: Rc<RefCell<Vec<Box<dyn RuntimeVal>>>>,
}

#[derive(Debug, Clone)]
pub struct BooleanLiteral {
    kind: NodeType,
    value: bool,
}

#[derive(Debug, Clone)]
pub struct NumberVal {
    pub r#type: ValueType,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct BooleanVal {
    pub r#type: ValueType,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub struct StringLiteralExpr {
    r#type: ValueType,
    kind: NodeType,
    pub(crate) value: String,
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
pub struct ObjectVal {
    pub r#type: ValueType,
    pub properties:  Rc<RefCell<HashMap<String, Box<dyn RuntimeVal>>>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Property {
    kind: NodeType,
    key: String,
    value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct ObjectLiteral {
    kind: NodeType,
    properties: Vec<Property>,
}


#[derive(Debug, Clone, PartialEq)]
pub struct LiteralExpr {
    kind: NodeType,
    value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifierExpr {
    pub kind: NodeType,
    pub name: String,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct NullLiteral {
    kind: NodeType,
    value: String,
}



#[derive(Debug)]
#[derive(Clone)]
pub struct BinaryExpr {
    kind: NodeType,
    left: Box<dyn Stmt>,
    right: Box<dyn Stmt>,
    operator: String,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct AssignmentExpr {
    kind: NodeType,
    assigne: Box<dyn Stmt>,
    value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct MemberExpr {
    kind: NodeType,
    object: Box<dyn Stmt>,
    property: Box<dyn Stmt>,
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
    kind: NodeType,
    body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct VariableDeclaration {
    kind: NodeType,
    identifier: String,
    value: Option<Box<dyn Stmt>>,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct FunctionDeclaration {
    kind: NodeType,
    parameters: Vec<String>,
    name: String,
    body: Vec<Box<dyn Stmt>>,
}

#[derive(Debug, Clone)]
pub struct FunctionVal {
    value_type: ValueType,
    name: String,
    parameters: Vec<String>,
    declaration_env: Rc<RefCell<Environment>>,
    body: Arc<Vec<Box<dyn Stmt>>>,
}

pub type FunctionCall =
Arc<dyn Fn(Vec<Box<dyn RuntimeVal>>, &mut Environment) -> Box<dyn RuntimeVal> + Send + Sync>;
#[derive(Clone)]
pub struct NativeFnValue {
    pub value_type: ValueType,
    pub call: FunctionCall,
}
impl Environment {
    pub fn new(parent: Option<Box<Environment>>) -> Self {
        Environment {
            parent,
            variables: HashMap::new(),
        }
    }
    
    pub fn declare_var(&mut self, var_name: String, value: Box<dyn RuntimeVal>) -> Box<dyn RuntimeVal> {
        
        //if self.variables.contains_key(&var_name) {
        //    panic!("Cannot declare variable {}. It is already defined.", var_name);
        //}
        self.variables.insert(var_name.clone(), value.clone());
        value
    }
    
    pub fn assign_var(&mut self, var_name: String, value: Box<dyn RuntimeVal>) -> Box<dyn RuntimeVal> {
        let env = self.resolve(&var_name);
        env.variables.insert(var_name.clone(), value.clone());
        value
    }
    pub fn lookup_var(&mut self, var_name: String) -> Box<dyn RuntimeVal> {
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
impl Program {
    pub fn new() -> Self {
        Self {
            kind: NodeType::Program,
            body: Vec::new(),
        }
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
    pub fn new(identifier: String, value: Option<Box<dyn Stmt>>) -> Self {
        Self {
            kind: NodeType::VariableDeclaration,
            identifier,
            value,
        }
    }
}
impl RuntimeVal for NullVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for NumberVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for StringLiteralExpr {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ArrayVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for NativeFnValue {
    fn value_type(&self) -> ValueType {
        self.value_type.clone()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl RuntimeVal for FunctionVal {
    fn value_type(&self) -> ValueType {
        self.value_type.clone()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl RuntimeVal for BooleanVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ObjectVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Stmt for BooleanLiteral {
    fn kind(&self) -> NodeType { self.kind.clone() }
    fn value(&self) -> Option<String> { Some(self.value.to_string()) }
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Stmt for StringLiteralExpr {
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

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    pub fn produce_ast(&mut self) -> Program {
        let mut program = Program::new();
        while self.not_eof() {
            program.body.push(self.parse_stmt());
        }
        program
    }

    fn not_eof(&self) -> bool {
        self.position < self.tokens.len() && self.tokens[self.position] != Token::EOF
    }

    fn at(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn eat(&mut self) -> Token {
        let token = self.tokens[self.position].clone();
        self.position += 1;
        token
    }

    fn eat_if(&mut self, expected: Token) -> bool {
        if *self.at() == expected {
            self.eat();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> &Token {
        if self.position + 1 < self.tokens.len() {
            &self.tokens[self.position + 1]
        } else {
            &Token::EOF
        }
    }

    fn expect(&mut self, expected: Token, err_msg: &str) -> Token {
        let token = self.eat();
        if token != expected {
            panic!("Parser Error: {}\nGot: {:?}, Expected: {:?}", err_msg, token, expected);
        }
        token
    }

    fn parse_stmt(&mut self) -> Box<dyn Stmt> {

        match self.at() {
            Token::Identifier(_) if *self.peek() == Token::Semicolon => {
                let name = if let Token::Identifier(n) = self.eat() { n } else { unreachable!() };
                self.eat();
                Box::from(VariableDeclaration::new(name, None))
            }
            Token::Identifier(_) if *self.peek() == Token::Equal => {
                self.parse_variable_declaration()
            }
            Token::Fn => {
                self.parse_func_declaration()
            }
            Token::With => {
                self.parse_with_declaration()
            }
            Token::Obj => {
                self.parse_obj_declaration()
            }
            Token::If => {
                self.parse_if_statement()
            }
            _ => self.parse_expr(),
        }
    }

    fn parse_with_declaration(&mut self) -> Box<dyn Stmt> {
        self.eat();

        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected import name following with keyword");
        };

        //println!("{:?}", name);

        if name.starts_with("_") {
            let mut new_name = name[1..].to_string();
            new_name.push_str(".nwtz");
            let import = fs::read_to_string(&new_name)
                .expect("Erreur lors de la lecture du fichier");
            //println!("{}", import);
            let tokens = tokenize(import);
            let mut external_parser = Parser::new(tokens);
            let external_ast = external_parser.produce_ast();
            self.expect(Semicolon, "';' after import");
            return Box::from(ImportAst {
                kind: NodeType::ImportAst,
                body: external_ast.body,
            });
        } else {
            unimplemented!("Chargement depuis une installation locale ou via le web");
        }


    }

    fn parse_if_statement(&mut self) -> Box<dyn Stmt> {
        self.eat();

        let left_token = self.eat();
        let left_expr: Box<dyn Stmt> = match left_token {
            Token::Identifier(name) => Box::from(IdentifierExpr {
                kind: NodeType::Identifier,
                name,
            }),
            Token::Integer(number) => Box::from(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value: number as f64,
            }),
            other => {
                self.expect(Token::Identifier(String::new()), "Variable or integer expected as left-hand side");
                unreachable!();
            }
        };

        let op_token = self.eat();
        let operator: String = match op_token {
            Token::Equal => "=".to_string(),
            Token::Greater => ">".to_string(),
            Token::Lower => "<".to_string(),
            _ => {
                self.expect(Token::Equal, "Expected an operator (=, >, or <)");
                unreachable!();
            }
        };


        let right_token = self.eat();
        let right_expr: Box<dyn Stmt> = match right_token {
            Token::Identifier(name) => Box::from(IdentifierExpr {
                kind: NodeType::Identifier,
                name,
            }),
            Token::Integer(number) => Box::from(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value: number as f64,
            }),
            _ => {
                self.expect(Token::Identifier(String::new()), "Variable or integer expected as right-hand side");
                unreachable!();
            }
        };

        let condition = Box::from(BinaryExpr {
            kind: NodeType::BinaryExpression,
            left: left_expr,
            right: right_expr,
            operator,
        });
        
        condition

    }

    fn parse_func_declaration(&mut self) -> Box<dyn Stmt> {

        self.eat();

        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected function name following fn keyword");
        };

        let args = self.parse_args();
        let mut params = Vec::new();
        for arg in args {
            if let Some(ident) = arg.as_any().downcast_ref::<IdentifierExpr>() {
                params.push(ident.name.clone());
            } else {
                panic!("Inside function declaration expected parameters to be identifiers");
            }
        }

        self.expect(LBrace, "Expected function body following declaration");
        let mut body = Vec::new();
        while self.not_eof() && *self.at() != RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(RBrace, "Closing brace expected inside function declaration");

        Box::from(FunctionDeclaration{
            kind: NodeType::FunctionDeclaration,
            parameters: params,
            name,
            body,
        })
    }

    fn parse_obj_declaration(&mut self) -> Box<dyn Stmt> {
        self.eat();
        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected identifier after 'obj'");
        };

        self.expect(Token::LBrace, "Expected '{' after object name");

        let mut properties: Vec<Property> = Vec::new();

        while self.not_eof() && *self.at() != Token::RBrace {
            let key = if let Token::Identifier(key) = self.eat() {
                key
            } else {
                panic!("Expected property key in object literal");
            };

            self.expect(Token::Colon, "Expected ':' after property key");
            let value = self.parse_expr();

            properties.push(Property {
                kind: NodeType::Property,
                key,
                value: Some(value),
            });

            if *self.at() == Token::Comma {
                self.eat();
            }
        }

        self.expect(RBrace, "Expected '}' to close object literal");
        self.expect(Token::Semicolon, "Expected ';' after object declaration");

        let obj_literal = Box::from(ObjectLiteral {
            kind: NodeType::ObjectLiteral,
            properties,
        });

        Box::from(VariableDeclaration::new(name, Some(obj_literal)))
    }


    fn parse_variable_declaration(&mut self) -> Box<dyn Stmt> {
        let identifier = match self.eat() {
            Token::Identifier(name) => name,
            token => panic!("Parser Error: Expected identifier, got {:?}", token),
        };

        let value = if *self.at() == Token::Equal {
            self.eat();
            Some(self.parse_expr())
        } else {
            None
        };
        self.expect(Token::Semicolon, "Expected semicolon after variable declaration");
        Box::from(VariableDeclaration {
            kind: NodeType::VariableDeclaration,
            identifier,
            value,
        })
    }

    fn parse_assignment_expr(&mut self) -> Box<dyn Stmt> {
        let left = self.parse_object_expr();

        if *self.at() == Token::Equal {
            self.eat();
            let value = self.parse_assignment_expr();
            return Box::from(AssignmentExpr {
                kind: NodeType::AssignmentExpr,
                assigne: left,
                value: Some(value),
            });
        }

        left
    }

    fn parse_object_expr(&mut self) -> Box<dyn Stmt> {
        if *self.at() != Token::LBrace {
            return self.parse_additive_expr()
        }

        self.eat();
        let mut properties: Vec<Property> = Vec::new();
        while self.not_eof() && *self.at() != RBrace{
            let key = if let Token::Identifier(name) = self.eat() {
                if name == "String"{
                    
                }
                name
            } else {
                panic!("Parser Error: Object literal key expected, got {:?}", self.at());
            };

            if *self.at() == Token::Comma{
                self.eat();
                properties.push(Property {
                    kind: NodeType::Property,
                    key,
                    value: None,
                });
                continue
            }
            else if *self.at() == Token::RBrace{
                properties.push(Property {
                    kind: NodeType::Property,
                    key,
                    value: None,
                });
                continue
            }

            self.expect(Token::Colon, "Missing colon following identifier in ObjectExpr");

            let value = self.parse_expr();

            properties.push(Property{
                kind: NodeType::Property,
                key,
                value: Option::from(value),
            });

            if *self.at() != RBrace{
                self.expect(Token::Comma, "Expected comma or Closing Bracket following Property");
            }

        }

        self.expect(RBrace, "Object literal missing closing brace");
        Box::from(ObjectLiteral{
            kind: NodeType::ObjectLiteral,
            properties,
        })
    }

    fn parse_expr(&mut self) -> Box<dyn Stmt> {
        self.parse_assignment_expr()
    }

    fn parse_additive_expr(&mut self) -> Box<dyn Stmt> {
        let mut left = self.parse_multiplicative_expr();

        while let Token::Plus | Token::Minus = self.at() {
            let operator = match self.eat() {
                Token::Plus => "+".to_string(),
                Token::Minus => "-".to_string(),
                _ => unreachable!(),
            };

            let right = self.parse_multiplicative_expr();
            left = Box::from(BinaryExpr {
                kind: NodeType::BinaryExpression,
                left,
                right,
                operator,
            });
        }

        left
    }

    fn parse_multiplicative_expr(&mut self) -> Box<dyn Stmt> {
        let mut left = self.parse_call_member_expr();

        while let Token::Star | Token::Slash | Token::Percent = self.at() {
            let operator = match self.eat() {
                Token::Star => "*".to_string(),
                Token::Slash => "/".to_string(),
                Token::Percent => "%".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_call_member_expr();

            left = Box::from(BinaryExpr {
                kind: NodeType::BinaryExpression,
                left,
                right,
                operator,
            });
        }

        left
    }

    fn parse_call_member_expr(&mut self) -> Box<dyn Stmt>{
        let member = self.parse_member_expr();

        if *self.at() == Token::LParen{
            return self.parse_call_expr(member)
        }

        member
    }

    fn parse_call_expr(&mut self, caller: Box<dyn Stmt>) -> Box<dyn Stmt>{

        let mut call_expr = Box::from(CallExpr{
            kind: NodeType::CallExpr,
            caller,
            args: self.parse_args(),
        });

        if *self.at() == Token::LParen{
            call_expr = self.parse_call_expr(call_expr).downcast().unwrap();
        }

        call_expr
    }

    fn parse_args(&mut self) -> Vec<Box<dyn Stmt>>{

        self.expect(Token::LParen, "Expected open parenthesis");
        let args = match *self.at() {
            Token::RParen => {
                Vec::new()
            }
            _ => {
                self.parse_arguments_list()
            }
        };

        self.expect(Token::RParen, "Missing close parenthesis");

        args
    }

    fn parse_arguments_list(&mut self) -> Vec<Box<dyn Stmt>>{

        let mut args = Vec::new();
        args.insert(0, self.parse_assignment_expr());

        while let Token::Comma = self.at() {
            self.eat();
            args.push(self.parse_assignment_expr());
        }


        args
    }
    
    fn parse_array_expr(&mut self) -> Box<dyn Stmt> {
        //self.expect(Token::LBracket, "Expected '[' to start array literal");
        
        
        let mut elements: Vec<Box<dyn Stmt>> = Vec::new();

        if *self.at() != Token::RBracket {
            // On parse le premier élément
            elements.push(self.parse_expr());
            // Et les suivants séparés par des virgules
            while *self.at() == Token::Comma {
                self.eat();
                elements.push(self.parse_expr());
            }
        }
        self.expect(Token::RBracket, "Expected ']' to close array literal");

        Box::from(ArrayLiteral {
            kind: NodeType::ArrayLiteral,
            elements,
        })
        
    }

    fn parse_member_expr(&mut self) -> Box<dyn Stmt>{

        let mut object = self.parse_primary_expr();

        while *self.at() == Token::Dot || *self.at() == Token::LBracket {
            let operator = self.eat();
            let property: Box<dyn Stmt>;

            if operator == Token::Dot {
                property = self.parse_primary_expr();

                if property.kind() != NodeType::Identifier {
                    panic!("Cannot use dot operator without right hand side being a identifier {:?}", self.at());
                }
            } else {
                property = self.parse_expr();
                self.expect(Token::RBracket, "Missing closing bracket in computed value");
            }

            object = Box::from(MemberExpr{
                kind: NodeType::MemberExpr,
                object,
                property,
            });
        }

        object
    }


    fn parse_primary_expr(&mut self) -> Box<dyn Stmt> {
        match self.eat() {
            Token::Integer(value) => Box::from(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value: value as f64,
            }),
            Token::Float(value) => Box::from(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value,
            }),
            Token::Identifier(name) => Box::from(IdentifierExpr {
                kind: Identifier,
                name,
            }),
            Token::StringLiteral(value) =>{
                let unquoted = value[1..value.len()-1].to_string();

                Box::from(StringLiteralExpr {
                    r#type: ValueType::String,
                    kind: NodeType::StringLiteral,
                    value: unquoted,
                })
            },
            Token::Null => Box::from(NullLiteral {
                kind: NodeType::NullLiteral,
                value: "null".to_string(),
            }),
            Token::LParen => {
                let expr = self.parse_expr();
                self.expect(Token::RParen, "Expected `)`.");
                expr
            },
            Token::False => Box::from(BooleanLiteral {
                kind: NodeType::BooleanLiteral,
                value: false,
            }),
            Token::True => Box::from(BooleanLiteral {
                kind: NodeType::BooleanLiteral,
                value: true,
            }),
            Token::LBracket => self.parse_array_expr(),
            token => Box::from(IdentifierExpr {
                kind: NodeType::Identifier,
                name: "null".to_string(),
            }),
        }
    }
}



pub fn tokenize(source: String) -> Vec<Token> {
    let mut lexer = Token::lexer(&source);
    let mut tokens: Vec<Token> = Vec::new();

    while let Some(token_result) = lexer.next() {
        match token_result {
            Ok(token) => tokens.push(token),
            Err(_) => {
                eprintln!(
                    "Lexer Error: Unexpected token at position {}",
                    lexer.span().start
                );
                continue;
            }
        }
    }

    tokens.push(Token::EOF);
    tokens
}

pub fn evaluate(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    match ast_node.kind() {
        NodeType::NumericLiteral => {
            let literal = ast_node.as_any().downcast_ref::<LiteralExpr>()
                .expect("Expected a LiteralExpr");
            Box::from(NumberVal {
                r#type: ValueType::Number,
                value: literal.value,
            })
        },
        NodeType::BooleanLiteral => {
            let bool_node = ast_node.as_any().downcast_ref::<BooleanLiteral>().unwrap();
            Box::from(BooleanVal { r#type: Boolean, value: bool_node.value })
        },
        NodeType::NullLiteral => {
            mk_null()
        },
        NodeType::BinaryExpression => {
            eval_binary_expr(ast_node, env)
        },
        NodeType::Program => {
            eval_program(ast_node, env)
        },
        Identifier => {
            eval_identifier(ast_node, env)
        },
        NodeType::VariableDeclaration => {
            eval_var_declaration(ast_node, env)
        },
        NodeType::StringLiteral => {
            let str = ast_node.as_any().downcast_ref::<StringLiteralExpr>()
                .expect("Expected a LiteralExpr");
            Box::from(StringLiteralExpr {
                r#type: ValueType::String,
                kind: NodeType::StringLiteral,
                value: str.value.clone(),
            })
        },
        NodeType::AssignmentExpr => {
            eval_assignment(ast_node, env) 
        },
        NodeType::ObjectLiteral => {
            eval_object_expr(ast_node, env)
        },
        NodeType::CallExpr => {
            eval_call_expr(ast_node, env)
        },
        NodeType::FunctionDeclaration => {
            eval_function_declaration(ast_node, env)
        },
        NodeType::MemberExpr => {
            eval_member_expr(ast_node, env)
        },
        NodeType::Property => panic!("This ast node has not yet been setup for interpretation {:#?}", ast_node),
        NodeType::ArrayLiteral => {
            eval_array_expr(ast_node, env)
        },
        NodeType::ImportAst => {
            let import_ast = ast_node.as_any().downcast_ref::<ImportAst>()
                .expect("Expected ImportAst node");
            let mut last: Box<dyn RuntimeVal> = mk_null();
            for stmt in import_ast.body.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last

        },
        NodeType::IfStatement => eval_if_statement(ast_node, env),
    }
}

pub fn eval_if_statement(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let if_stmt = ast_node.as_any().downcast_ref::<IfStatement>()
        .expect("Expected an IfStatement node");

    let cond_val = evaluate(if_stmt.condition.clone(), env);
    let condition_is_true = if let Some(boolean) = cond_val.as_any().downcast_ref::<BooleanVal>() {
        boolean.value
    } else {
        panic!("La condition du if n'est pas un booléen");
    };

    if condition_is_true {
        let mut then_closure = || {
            let mut last: Box<dyn RuntimeVal> = mk_null();
            for stmt in if_stmt.then_branch.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last
        };
        then_closure()
    } else if let Some(else_branch) = &if_stmt.else_branch {
        let mut else_closure = || {
            let mut last: Box<dyn RuntimeVal> = mk_null();
            for stmt in else_branch.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last
        };
        else_closure()
    } else {
        mk_null()
    }
}


pub fn eval_array_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let array_node = node.downcast::<ArrayLiteral>()
        .expect("Expected an ArrayLiteral node");

    let elements: Vec<Box<dyn RuntimeVal>> = array_node
        .elements
        .into_iter()
        .map(|expr| evaluate(expr, env))
        .collect();

    Box::from(ArrayVal {
        r#type: ValueType::Array,
        elements: Rc::new(RefCell::new(elements)),
    })
}



pub fn eval_assignment(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let assignment = node
        .downcast::<AssignmentExpr>()
        .expect("Expected an AssignmentExpr node");

    let new_value = evaluate(
        assignment.value.expect("Assignment missing RHS"),
        env,
    );

    if let Some(ident) = assignment.assigne.as_any().downcast_ref::<IdentifierExpr>() {
        return env.assign_var(ident.name.clone(), new_value);
    } else if let Some(member) = assignment.assigne.as_any().downcast_ref::<MemberExpr>() {
        let object_val = evaluate(member.object.clone(), env);
        // Affectation sur un objet
        if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
            let prop_name = if let Some(ident) = member.property.as_any().downcast_ref::<IdentifierExpr>() {
                ident.name.clone()
            } else {
                panic!("Member expression expected an identifier for non-computed property");
            };
            obj.properties.borrow_mut().insert(prop_name, new_value.clone());
            return new_value;
        }
        // Affectation sur un tableau
        else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
            let index_val = evaluate(member.property.clone(), env);
            if let Some(num) = index_val.as_any().downcast_ref::<NumberVal>() {
                let index = num.value as usize;
                if index < arr.elements.borrow().len() {
                    arr.elements.borrow_mut()[index] = new_value.clone();
                    return new_value;
                } else {
                    panic!(
                        "Index out of bounds: {} is greater than the array size {}",
                        index,
                        arr.elements.borrow().len()
                    );
                }
            } else {
                panic!("The index of an array must be a number");
            }
        }
        else {
            panic!("Assignment target is not an object or an array");
        }
    } else {
        panic!("Invalid LHS in assignment: {:?}", assignment.assigne);
    }
}



pub fn eval_var_declaration(declaration: Box<dyn Stmt>, env: &mut Environment, ) -> Box<dyn RuntimeVal> {
    let var_declaration = declaration
        .downcast::<VariableDeclaration>()
        .expect("Expected a VariableDeclaration node");
    let value = match var_declaration.value {
        Some(expr) => evaluate(expr, env),
        None => mk_null(),
    };
    env.declare_var(var_declaration.identifier, value.clone())
}

pub fn eval_member_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let member = ast_node.downcast::<MemberExpr>()
        .expect("Expected a MemberExpr node");

    let object_val = evaluate(member.object, env);

    if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
        let prop_name = if let Some(ident) = member.property.as_any().downcast_ref::<IdentifierExpr>() {
            ident.name.clone()
        } else {
            panic!("Member expression expected an identifier for a non-computed property");
        };

        match obj.properties.borrow_mut().get(&prop_name) {
            Some(val) => val.clone(),
            None => panic!("The property '{}' does not exist in the object", prop_name),
        }
    } else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
        let index_val = evaluate(member.property, env);
        if let Some(num) = index_val.as_any().downcast_ref::<NumberVal>() {
            let index = num.value as usize;
            if index < arr.elements.borrow().len() {
                arr.elements.borrow()[index].clone()
            } else {
                panic!("Index out of bounds: {} is greater than the array size {}", index, arr.elements.borrow().len());
            }
        } else {
            panic!("The index of an array must be a number");
        }
    } else {
        panic!("Member access on a type that is neither an object nor an array");
    }
}

pub fn eval_binary_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let binary_expr = ast_node.downcast::<BinaryExpr>()
        .expect("Expected a BinaryExpr node");

    let lhs = evaluate(binary_expr.left, env);
    let rhs = evaluate(binary_expr.right, env);

    if let (Some(lhs_num), Some(rhs_num)) = (
        lhs.as_any().downcast_ref::<NumberVal>(),
        rhs.as_any().downcast_ref::<NumberVal>()
    ) {
        return eval_numeric_binary_expr(
            lhs_num,
            rhs_num,
            &binary_expr.operator,
        );
    }

    Box::from(NullVal {
        r#type: Null,
        value: "null".to_string(),
    })
}
pub fn eval_numeric_binary_expr(lhs: &NumberVal, rhs: &NumberVal, operator: &str, ) -> Box<dyn RuntimeVal> {
    let lhs_value = lhs.value;
    let rhs_value = rhs.value;

    let result = match operator {
        "+" => lhs_value + rhs_value,
        "-" => lhs_value - rhs_value,
        "*" => lhs_value * rhs_value,
        "/" => lhs_value / rhs_value, // TODO: Division by zero checks
        "%" => lhs_value % rhs_value,
        "=" => {
            return Box::from(BooleanVal {
                r#type: Boolean,
                value: lhs_value == rhs_value,
            });
        },
        _ => panic!("Unknown binary operator: {}", operator),
    };

    Box::from(NumberVal {
        r#type: Number,
        value: result,
    })
}
pub fn eval_program(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let program = ast_node.downcast::<Program>()
        .expect("Expected a Program node");

    let mut last_evaluated: Box<dyn RuntimeVal> = mk_null();

    for stmt in program.body.into_iter().filter(|s| {
        !(matches!(s.kind(), NodeType::Identifier) && s.as_any().downcast_ref::<IdentifierExpr>().unwrap().name == "null")
    }) {
        last_evaluated = evaluate(stmt, env);
    }


    last_evaluated
}

pub fn eval_identifier(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let identifier_expr = ast_node.downcast::<IdentifierExpr>()
        .expect("Expected an IdentifierExpr node");

    if RESERVED_NAMES.contains(&identifier_expr.name) {
        return mk_null();
    }

    env.lookup_var(identifier_expr.name)
}


pub fn eval_object_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let obj = node
        .downcast::<ObjectLiteral>()
        .expect("Expected ObjectLiteral");

    let mut props: HashMap<String, Box<dyn RuntimeVal>> = HashMap::new();

    for Property { key, value, .. } in obj.properties {
        let runtime_val = if let Some(expr) = value {
            evaluate(expr, env)
        } else {
            env.lookup_var(key.clone())
        };
        props.insert(key, runtime_val);
    }

    Box::from(ObjectVal {
        r#type: ValueType::Object,
        properties: Rc::new(RefCell::new(props)),
    })
}

pub fn eval_call_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let call = node
        .downcast::<CallExpr>()
        .expect("Expected CallExpr");

    let args: Vec<Box<dyn RuntimeVal>> = call
        .args
        .into_iter()
        .map(|arg| evaluate(arg, env))
        .collect();

    let callee = evaluate(call.caller, env);

    if let Some(native) = callee.as_any().downcast_ref::<NativeFnValue>() {
        return (native.call)(args, env);
    }

    if let Some(func) = callee.as_any().downcast_ref::<FunctionVal>() {
        let decl_env = func.declaration_env.borrow_mut();
        let mut scope = Environment::new(Some(Box::new(decl_env.clone())));
        if args.len() != func.parameters.len() {
            panic!(
                "Function `{}` expected {} arguments but got {}",
                func.name,
                func.parameters.len(),
                args.len()
            );
        }
        
        for (param, arg_val) in func.parameters.iter().zip(args.into_iter()) {
            scope.declare_var(param.clone(), arg_val);
        }

        let mut result: Box<dyn RuntimeVal> = mk_null();
        for stmt in func.body.iter() {
            result = evaluate(stmt.clone(), &mut scope);
        }
        
        return result;
    }

    panic!("Cannot call value that is not a function: {:?}", callee);
}


pub fn eval_function_declaration(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal>{
    let func = node
        .downcast::<FunctionDeclaration>()
        .expect("Expected FunctionDeclaration");

    let function = FunctionVal{
        value_type: ValueType::Function,
        parameters: func.parameters,
        name: func.name.clone(),
        body: Arc::new(func.body),
        declaration_env: Rc::new(RefCell::new(env.clone())),
    };

    //let decl_env: &mut Environment = unsafe { &mut *function_val.declaration_env };

    env.declare_var(func.name, Box::from(function))
}


pub fn mk_number<T: Into<f64>>(number: T) -> Box<NumberVal> {
    Box::from(NumberVal {
        r#type: Number,
        value: number.into(),
    })
}


pub fn mk_null() -> Box<NullVal> {
    Box::from(NullVal{ r#type: ValueType::Null, value: "null".to_string() })
}

pub fn mk_bool(b: bool) -> Box<BooleanVal> {
    Box::from(BooleanVal{ r#type: ValueType::Boolean, value: b })
}

pub fn mk_native_fn(call: FunctionCall) -> Box<NativeFnValue> {
    Box::from(NativeFnValue{
        value_type: NativeFn,
        call
    })
}

pub fn mk_array(elements: Vec<Box<dyn RuntimeVal>>) -> Box<ArrayVal> {
    Box::from(ArrayVal {
        r#type: ValueType::Array,
        elements: Rc::new(elements.into()),
    })
}
