use std::any::Any;
use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::{fs, thread};
use logos::{Logos};
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use crate::NodeType::Identifier;
use crate::Token::{LBrace, RBrace, Semicolon};
use crate::ValueType::{Boolean, NativeFn, Null, Integer, Object, Function, Array};

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
    #[token("const")]
    Const,
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

static RESERVED_NAMES: Lazy<HashSet<String>> = Lazy::new(|| {
    ValueType::iter().map(|vt| format!("{:?}", vt)).collect()
});


#[derive(Debug, Clone)]
pub struct Environment {
    parent: Option<Box<Environment>>,
    variables: HashMap<String, Box<dyn RuntimeVal + Send + Sync>>,
    var_types: HashMap<String, Option<ValueType>>,
}

#[derive(Debug, Clone)]
pub struct ImportAst {
    pub kind: NodeType,
    pub body: Vec<Box<dyn Stmt>>,
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
pub struct BooleanLiteral {
    kind: NodeType,
    value: bool,
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
pub struct ObjectVal {
    pub r#type: Option<ValueType>,
    pub properties:  Arc<Mutex<HashMap<String, Box<dyn RuntimeVal + Send + Sync>>>>,
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
            match self.var_types.get(&var_name).unwrap() {
                Some(expected) => {
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
                None => {}
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
    pub fn new(identifier: String, value: Option<Box<dyn Stmt>>, r#type :Option<ValueType>) -> Self {
        Self {
            kind: NodeType::VariableDeclaration,
            r#type,
            name: identifier,
            value,
        }
    }
}
impl RuntimeVal for NullVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(self.r#type.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for IntegerVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(self.r#type.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for StringVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(self.r#type.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ArrayVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(self.r#type.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl RuntimeVal for NativeFnValue {
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
        Option::from(self.r#type.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl RuntimeVal for ObjectVal {
    fn value_type(&self) -> Option<ValueType> {
        Option::from(self.r#type.clone())
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

        let (imports, without_imports): (Vec<_>, Vec<_>) =
            program.body
                .into_iter()
                .partition(|stmt| stmt.kind() == NodeType::ImportAst);

        let (fns, rest): (Vec<_>, Vec<_>) =
            without_imports
                .into_iter()
                .partition(|stmt| stmt.kind() == NodeType::FunctionDeclaration);

        program.body = Vec::new();
        program.body.extend(imports);
        program.body.extend(fns);
        program.body.extend(rest);

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
            Token::Identifier(_) if *self.peek() == Semicolon => {
                let name = if let Token::Identifier(n) = self.eat() { n } else { unreachable!() };
                self.eat();
                Box::from(VariableDeclaration::new(name, None, None))
            }
            Token::Identifier(_) if *self.peek() == Token::Equal || *self.peek() == Token::Colon => {
                self.parse_var_declaration()
            }
            Token::Const => {
                // Faire en sorte que Const soit obligatoirement définie avec un Valuetype
                todo!()
                //self.parse_variable_declaration()
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
            Token::For => {
                self.parse_for_statement()
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
            Box::from(ImportAst {
                kind: NodeType::ImportAst,
                body: external_ast.body,
            })
        } else {
            unimplemented!("Chargement depuis une installation locale ou via le web");
        }


    }

    fn parse_for_statement(&mut self) -> Box<dyn Stmt> {
        /*
        for (i = 0 ; i < 10 ; i = i+1) {}

        pub struct ForStatement {
            pub kind: NodeType,
            pub initializer: Option<Box<dyn Stmt>>,
            pub condition: Option<Box<dyn Stmt>>,
            pub increment: Option<Box<dyn Stmt>>,
            pub body: Vec<Box<dyn Stmt>>,
        }
         */

        self.eat();

        self.expect(Token::LParen, "`(` attendu après `for`");

        let initializer: Option<Box<dyn Stmt>>  = if !matches!(self.at(), &Semicolon) {
            Some(self.parse_var_declaration())
        } else {
            None
        };
        //self.expect(Semicolon, "`;` attendu après l'initialisation de i");

        let condition: Option<Box<dyn Stmt>>  = if !matches!(self.at(), &Semicolon) {
            let left: Box<dyn Stmt> = match self.eat() {
                Token::Identifier(name) =>
                    Box::new(IdentifierExpr { kind: Identifier, name }),
                Token::Integer(n) =>
                    Box::new(LiteralExpr    { kind: NodeType::NumericLiteral, value: n as f64 }),
                t => panic!("Expr. attendue en condition, pas {:?}", t),
            };

            let op = match self.eat() {
                Token::Equal   => "==",
                Token::Greater => ">",
                Token::Lower   => "<",
                _ => panic!("Opérateur attendu (=, > ou <)"),
            }.to_string();

            let right: Box<dyn Stmt> = match self.eat() {
                Token::Identifier(name) =>
                    Box::new(IdentifierExpr { kind: Identifier, name }),
                Token::Integer(n) =>
                    Box::new(LiteralExpr { kind: NodeType::NumericLiteral, value: n as f64 }),
                t => panic!("Expr. attendue après opérateur, pas {:?}", t),
            };

            Some(Box::new(BinaryExpr {
                kind: NodeType::BinaryExpression,
                left,
                operator: op,
                right,
            }))
        } else {
            None
        };
        self.expect(Semicolon, "`;` attendu après la condition du for");

        let increment: Option<Box<dyn Stmt>> = if !matches!(self.at(), &Token::RParen) {
            Some(self.parse_var_declaration())
        } else {
            None
        };
        self.expect(Token::RParen, "`)` attendu après l’incrément du for");

        self.expect(LBrace, "`{` attendu pour ouvrir le corps du for");

        let mut body = Vec::new();
        while self.not_eof() && *self.at() != RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(RBrace, "`}` attendu pour fermer le for");

        Box::new(ForStatement {
            kind: NodeType::ForStatement,
            initializer,
            condition,
            increment,
            body,
        })
    }

    fn parse_if_statement(&mut self) -> Box<dyn Stmt> {
        self.eat();

        let left_expr: Box<dyn Stmt> = match self.eat() {
            Token::Identifier(name) =>
                Box::new(IdentifierExpr { kind: Identifier, name }),
            Token::Integer(n) =>
                Box::new(LiteralExpr { kind: NodeType::NumericLiteral, value: n as f64 }),
            t => panic!("Variable ou entier attendu après `if` pas {:?}", t),
        };

        let operator = match self.eat() {
            Token::Equal   => "=",
            Token::Greater => ">",
            Token::Lower   => "<",
            _ => panic!("Opérateur attendu (=, > ou <)"),
        }.to_string();

        let right_expr: Box<dyn Stmt> = match self.eat() {
            Token::Identifier(name) =>
                Box::new(IdentifierExpr { kind: Identifier, name }),
            Token::Integer(n) =>
                Box::new(LiteralExpr    { kind: NodeType::NumericLiteral, value: n as f64 }),
            _ => panic!("Variable ou entier attendu après l’opérateur"),
        };

        let condition = Box::new(BinaryExpr {
            kind: NodeType::BinaryExpression,
            left:  left_expr,
            right: right_expr,
            operator,
        });

        self.expect(LBrace, "‘{’ attendu après la condition du if");
        let mut then_branch = Vec::new();
        while self.not_eof() && *self.at() != RBrace {
            then_branch.push(self.parse_stmt());
        }
        self.expect(RBrace, "‘}’ attendu pour clôturer le bloc then");

        let else_branch = if self.eat_if(Token::Else) {
            self.expect(LBrace, "‘{’ attendu après else");
            let mut else_vec = Vec::new();
            while self.not_eof() && *self.at() != RBrace {
                else_vec.push(self.parse_stmt());
            }
            self.expect(RBrace, "‘}’ attendu pour clôturer le bloc else");
            Some(else_vec)
        } else {
            None
        };

        Box::new(IfStatement {
            kind: NodeType::IfStatement,
            condition,
            then_branch,
            else_branch,
        })
    }


    fn parse_func_declaration(&mut self) -> Box<dyn Stmt> {

        self.eat();

        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected function name following fn keyword, got {:?}", self.at());
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

        self.expect(LBrace, "Expected '{' after object name");

        let mut properties: Vec<Property> = Vec::new();

        while self.not_eof() && *self.at() != RBrace {
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
        self.expect(Semicolon, "Expected ';' after object declaration");

        let obj_literal = Box::from(ObjectLiteral {
            kind: NodeType::ObjectLiteral,
            properties,
        });

        Box::from(VariableDeclaration::new(name, Some(obj_literal), Some(Object)))
    }


    fn parse_var_declaration(&mut self) -> Box<dyn Stmt> {
        let identifier = match self.eat() {
            Token::Identifier(name) => name,
            token => panic!("Parser Error: Expected identifier, got {:?}", token),
        };

        let var_type = if *self.at() == Token::Colon {
            self.eat();
            let ty_name = if let Token::Identifier(name) = self.eat() {
                name
            } else {
                panic!("Parser Error: Expected type name after `:`, got {:?}", self.at());
            };
            let vt = ValueType::iter()
                .find(|vt| format!("{:?}", vt) == ty_name)
                .unwrap_or_else(|| panic!("Unknown type literal `{}`", ty_name));

            Some(vt)
        } else {
            None
        };


        let value = if *self.at() == Token::Equal {
            self.eat();

            if let Some(expected) = var_type.as_ref() {
                match self.at() {
                    Token::StringLiteral(_) if *expected != ValueType::String => {
                        panic!(
                            "Parser Error: `{}` declared as `{:?}` but got a String",
                            identifier, expected
                        );
                    }
                    Token::Integer(_) if *expected != Integer => {
                        panic!(
                            "Parser Error: `{}` declared as `{:?}` but got an Integer",
                            identifier, expected
                        );
                    }
                    Token::True | Token::False if *expected != Boolean => {
                        panic!(
                            "Parser Error: `{}` declared as `{:?}` but got a Boolean",
                            identifier, expected
                        );
                    }
                    Token::Null if *expected != Null => {
                        panic!(
                            "Parser Error: `{}` declared as `{:?}` but got Null",
                            identifier, expected
                        );
                    }
                    _ => {

                    }
                }
            }
            Some(self.parse_expr())
        } else {
            None
        };
        
        self.expect(Semicolon, "Expected semicolon after variable declaration");
                
        Box::from(VariableDeclaration {
            kind: NodeType::VariableDeclaration,
            r#type: var_type,
            name: identifier,
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
        if *self.at() != LBrace {
            return self.parse_additive_expr()
        }

        self.eat();
        let mut properties: Vec<Property> = Vec::new();
        while self.not_eof() && *self.at() != RBrace{
            let key = if let Token::Identifier(name) = self.eat() {
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
            else if *self.at() == RBrace{
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
        //self.expect(Token::LBracket, "Expected '[' to start array literal"); ]


        let mut elements: Vec<Box<dyn Stmt>> = Vec::new();

        if *self.at() != Token::RBracket {
            elements.push(self.parse_expr());
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

                if property.kind() != Identifier {
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

                Box::from(StringVal {
                    r#type: Option::from(ValueType::String),
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
            _token => Box::from(IdentifierExpr {
                kind: Identifier,
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

pub fn evaluate(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    
    match ast_node.kind() {
        NodeType::NumericLiteral => {
            let literal = ast_node.as_any().downcast_ref::<LiteralExpr>()
                .expect("Expected a LiteralExpr");
            Box::from(IntegerVal {
                r#type: Some(Integer),
                value: literal.value,
            })
        },
        NodeType::BooleanLiteral => {
            let bool_node = ast_node.as_any().downcast_ref::<BooleanLiteral>().unwrap();
            Box::from(BooleanVal { r#type: Some(Boolean), value: bool_node.value })
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
            let str = ast_node.as_any().downcast_ref::<StringVal>()
                .expect("Expected a LiteralExpr");
            Box::from(StringVal {
                r#type: Some(ValueType::String),
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
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in import_ast.body.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last

        },
        NodeType::IfStatement => eval_if_statement(ast_node, env),
        NodeType::ForStatement => eval_for_statement(ast_node, env),
    }
}

pub fn eval_for_statement(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let for_stmt = ast_node
        .as_any()
        .downcast_ref::<ForStatement>()
        .expect("Expected a ForStatement node");

    if let Some(init_stmt) = &for_stmt.initializer {
        evaluate(init_stmt.clone(), env);
    }

    let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();

    loop {
        if let Some(cond_stmt) = &for_stmt.condition {
            let cond_val = evaluate(cond_stmt.clone(), env);
            let keep_going = cond_val
                .as_any()
                .downcast_ref::<BooleanVal>()
                .map(|b| b.value)
                .unwrap_or_else(|| panic!("Condition not Boolean"));
            if !keep_going {
                break;
            }
        }

        for stmt in &for_stmt.body {
            let val = evaluate(stmt.clone(), env);
            //println!("DEBUG ─ evaluate renvoie : {:?}", val);
            last = val;
        }


        if let Some(inc_stmt) = &for_stmt.increment {
            evaluate(inc_stmt.clone(), env);
        }
    }

    last
}


pub fn eval_if_statement(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
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
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in if_stmt.then_branch.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last
        };
        then_closure()
    } else if let Some(else_branch) = &if_stmt.else_branch {
        let mut else_closure = || {
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
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


pub fn eval_array_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let array_node = node.downcast::<ArrayLiteral>()
        .expect("Expected an ArrayLiteral node");

    let elements: Vec<Box<dyn RuntimeVal + Send + Sync>> = array_node
        .elements
        .into_iter()
        .map(|expr| evaluate(expr, env))
        .collect();

    Box::from(ArrayVal {
        r#type: Option::from(Array),
        elements: Arc::new(Mutex::new(elements)),
    })
}



pub fn eval_assignment(node: Box<dyn Stmt>, env: &mut Environment, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let assignment = node
        .downcast::<AssignmentExpr>()
        .expect("Expected an AssignmentExpr node");

    let new_value = evaluate(
        assignment.value.expect("Assignment missing RHS"),
        env,
    );

    if let Some(ident) = assignment.assigne.as_any().downcast_ref::<IdentifierExpr>() {
        let var_name = ident.name.clone();
        
        let result = env.set_var(var_name, new_value.clone(), None);

        result
    }
    else if let Some(member) = assignment.assigne.as_any().downcast_ref::<MemberExpr>() {
        let object_val = evaluate(member.object.clone(), env);

        if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
            let prop_name = if let Some(ident) =
                member.property.as_any().downcast_ref::<IdentifierExpr>()
            {
                ident.name.clone()
            } else {
                panic!("Member expression expected an identifier for non-computed property");
            };
            obj.properties.lock().unwrap().insert(prop_name, new_value.clone());
            new_value
        }
        else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
            let index_val = evaluate(member.property.clone(), env);
            if let Some(num) = index_val.as_any().downcast_ref::<IntegerVal>() {
                let index = num.value as usize;
                let mut elems = arr.elements.lock().unwrap();
                if index < elems.len() {
                    elems[index] = new_value.clone();
                    new_value
                } else {
                    panic!(
                        "Index out of bounds: {} is greater than the array size {}",
                        index,
                        elems.len()
                    );
                }
            } else {
                panic!("The index of an array must be a number");
            }
        }
        else {
            panic!("Assignment target is not an object or an array");
        }
    }
    else {
        panic!("Invalid LHS in assignment: {:?}", assignment.assigne);
    }
}



pub fn eval_var_declaration(declaration: Box<dyn Stmt>, env: &mut Environment, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let var_declaration = declaration
        .downcast::<VariableDeclaration>()
        .expect("Expected a VariableDeclaration node");

    let declared_type = var_declaration.r#type.clone();

    let value = match var_declaration.value {
        Some(expr) => evaluate(expr, env),
        None => mk_null(),
    };

    env.set_var(var_declaration.name.clone(), value.clone(), declared_type)
}
pub fn eval_member_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let member = ast_node.downcast::<MemberExpr>()
        .expect("Expected a MemberExpr node");

    let object_val = evaluate(member.object, env);

    if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
        let prop_name = if let Some(ident) = member.property.as_any().downcast_ref::<IdentifierExpr>() {
            ident.name.clone()
        } else {
            panic!("Member expression expected an identifier for a non-computed property");
        };

        match obj.properties.lock().unwrap().get(&prop_name) {
            Some(val) => val.clone(),
            None => panic!("The property '{}' does not exist in the object", prop_name),
        }
    } else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
        let index_val = evaluate(member.property, env);
        if let Some(num) = index_val.as_any().downcast_ref::<IntegerVal>() {
            let index = num.value as usize;
            if index < arr.elements.lock().unwrap().len() {
                arr.elements.lock().unwrap()[index].clone()
            } else {
                panic!("Index out of bounds: {} is greater than the array size {}", index, arr.elements.lock().unwrap().len());
            }
        } else {
            panic!("The index of an array must be a number");
        }
    } else {
        panic!("Member access on a type that is neither an object nor an array");
    }
}

pub fn eval_binary_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let binary_expr = ast_node.downcast::<BinaryExpr>()
        .expect("Expected a BinaryExpr node");

    let lhs = evaluate(binary_expr.left, env);
    let rhs = evaluate(binary_expr.right, env);

    if let (Some(lhs_num), Some(rhs_num)) = (
        lhs.as_any().downcast_ref::<IntegerVal>(),
        rhs.as_any().downcast_ref::<IntegerVal>()
    ) {
        return eval_numeric_binary_expr(
            lhs_num,
            rhs_num,
            &binary_expr.operator,
        );
    }

    Box::from(NullVal {
        r#type: Some(Null),
    })
}
pub fn eval_numeric_binary_expr(lhs: &IntegerVal, rhs: &IntegerVal, operator: &str, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let lhs_value = lhs.value;
    let rhs_value = rhs.value;

    match operator {
        "+" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value + rhs_value }),
        "-" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value - rhs_value }),
        "*" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value * rhs_value }),
        "/" => {
            if rhs_value == 0.0 {
                panic!("Division par zéro interdite");
            }
            Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value / rhs_value })
        },
        "%" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value % rhs_value }),

        ">"  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value  > rhs_value }),
        "<"  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value  < rhs_value }),
        ">=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value >= rhs_value }),
        "<=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value <= rhs_value }),
        "!=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value != rhs_value }),
        "=="  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value == rhs_value }),

        _ => panic!("Unknown binary operator: {}", operator),
    }
}

pub fn eval_program(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let program = ast_node.downcast::<Program>()
        .expect("Expected a Program node");

    let mut last_evaluated: Box<dyn RuntimeVal + Send + Sync> = mk_null();

    for stmt in program.body.into_iter().filter(|s| {
        !(matches!(s.kind(), Identifier) && s.as_any().downcast_ref::<IdentifierExpr>().unwrap().name == "null")
    }) {
        last_evaluated = evaluate(stmt, env);
    }


    last_evaluated
}

pub fn eval_identifier(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let identifier_expr = ast_node.downcast::<IdentifierExpr>()
        .expect("Expected an IdentifierExpr node");

    if RESERVED_NAMES.contains(&identifier_expr.name) {
        return mk_null();
    }

    env.lookup_var(identifier_expr.name)
}


pub fn eval_object_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let obj = node
        .downcast::<ObjectLiteral>()
        .expect("Expected ObjectLiteral");

    let mut props: HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();

    for Property { key, value, .. } in obj.properties {
        let runtime_val = if let Some(expr) = value {
            evaluate(expr, env)
        } else {
            env.lookup_var(key.clone())
        };
        props.insert(key, runtime_val);
    }

    Box::from(ObjectVal {
        r#type: Option::from(Object),
        properties: Arc::new(Mutex::new(props)),
    })
}

pub fn eval_call_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let call = node
        .downcast::<CallExpr>()
        .expect("Expected CallExpr");

    let args: Vec<Box<dyn RuntimeVal + Send + Sync>> = call
        .args
        .into_iter()
        .map(|arg| evaluate(arg, env))
        .collect();

    let callee = evaluate(call.caller, env);

    if let Some(native) = callee.as_any().downcast_ref::<NativeFnValue>() {
        return (native.call)(args, env);
    }

    if let Some(func) = callee.as_any().downcast_ref::<FunctionVal>() {
        let decl_env = func.declaration_env.lock().unwrap();
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
            scope.set_var(param.clone(), arg_val, None);
        }

        let mut result: Box<dyn RuntimeVal + Send + Sync> = mk_null();
        for stmt in func.body.iter() {
            result = evaluate(stmt.clone(), &mut scope);
        }

        return result;
    }

    panic!("Cannot call value that is not a function: {:?}", callee);
}


pub fn eval_function_declaration(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync>{
    let func = node
        .downcast::<FunctionDeclaration>()
        .expect("Expected FunctionDeclaration");

    let function = FunctionVal{
        r#type: Option::from(Function),
        parameters: func.parameters,
        name: func.name.clone(),
        body: Arc::new(func.body),
        declaration_env: Arc::new(Mutex::new(env.clone())),
    };

    //let decl_env : &mut Environment = unsafe {&mut *function_val.declaration_env} ;

    env.set_var(func.name, Box::from(function), Option::from(Function))
}


pub fn mk_number<T: Into<f64>>(number: T) -> Box<IntegerVal> {
    Box::from(IntegerVal {
        r#type: Option::from(Integer),
        value: number.into(),
    })
}


pub fn mk_null() -> Box<NullVal> {
    Box::from(NullVal{ r#type: Option::from(Null) })
}

pub fn mk_bool(b: bool) -> Box<BooleanVal> {
    Box::from(BooleanVal{ r#type: Option::from(Boolean), value: b })
}

pub fn mk_native_fn(call: FunctionCall) -> Box<NativeFnValue> {
    Box::from(NativeFnValue{
        r#type: Option::from(NativeFn),
        call
    })
}

pub fn _mk_array(elements: Vec<Box<dyn RuntimeVal + Send + Sync>>) -> Box<ArrayVal> {
    Box::from(ArrayVal {
        r#type: Option::from(Array),
        elements: Arc::new(Mutex::new(elements.into())),
    })
}

pub fn native_fs_read(args: Vec<Box<dyn RuntimeVal + Send + Sync>>, _env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let path = args.into_iter().next()
        .and_then(|v| v.as_any().downcast_ref::<StringVal>().map(|s| s.value.clone()))
        .expect("_fs_read attend un string");
    let content = fs::read_to_string(&path)
        .expect("Erreur lecture fichier");
    Box::new(StringVal {
        r#type: Some(ValueType::String),
        kind: NodeType::StringLiteral,
        value: content,
    })
}

macro_rules! register_natives {
    ( $($name:expr => $func:path),* $(,)? ) => {
        pub fn native_registry() -> HashMap<&'static str, FunctionCall> {
            let mut m = HashMap::new();
            $(
                m.insert($name, Arc::new($func) as FunctionCall);
            )*
            m
        }
    }
}

register_natives!(
    "_fs_read" => native_fs_read,
);


pub fn make_global_env() -> Environment {
    let mut env = Environment::new(None);

    env.set_var("null".to_string(), mk_null(), Option::from(Null));
    env.set_var("true".to_string(), mk_bool(true), Option::from(Boolean));
    env.set_var("false".to_string(), mk_bool(false), Option::from(Boolean));

    env.set_var(
        "time".to_string(),
        mk_native_fn(Arc::new(|_args: Vec<Box<dyn RuntimeVal + Send + Sync>>, _scope: &mut Environment| {
            mk_number(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as f64 / 1000.0)
        })), Option::from(NativeFn),
    );
    
    let registry = native_registry();
    for (name, func) in registry {
        env.set_var(
            name.to_string(),
            mk_native_fn(func.clone()),
            Option::from(NativeFn)
        );
    }
    env
}

pub fn interpreter_to_vec_string(mut env: &mut Environment, input: String) -> Vec<String> {
    let output = Arc::new(Mutex::new(Vec::<String>::new()));
    let output_for_native = output.clone();

    env.set_var(
        "log".to_string(),
        mk_native_fn(Arc::new(move |args, _| {
            let mut guard = output_for_native.lock().unwrap();
            for arg in args {
                guard.push(match_arg_to_string(&*arg));
            }
            mk_null()
        })),
        Option::from(NativeFn)
    );

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    let _ = evaluate(Box::new(ast), &mut env);
    output.lock().unwrap().clone()
}

pub fn interpreter_to_stream(env: &mut Environment, input: String, ) -> UnboundedReceiver<String> {
    let (tx, rx): (UnboundedSender<String>, UnboundedReceiver<String>) = unbounded_channel();

    let tx_for_native = tx.clone();
    env.set_var(
        "log".to_string(),
        mk_native_fn(Arc::new(move |args, _| {
            for arg in args {
                let s = match_arg_to_string(&*arg);
                let _ = tx_for_native.send(s);
            }
            mk_null()
        })),
        Some(NativeFn),
    );

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();

    let mut env_for_task = env.clone();
    tokio::spawn(async move {
        let _ = evaluate(Box::new(ast), &mut env_for_task);
        drop(tx);
    });

    rx
}

pub fn match_arg_to_string(arg: &dyn RuntimeVal) -> String {
    if let Some(sv) = arg.as_any().downcast_ref::<StringVal>() {
        sv.value.clone()
    } else if let Some(iv) = arg.as_any().downcast_ref::<IntegerVal>() {
        iv.value.to_string()
    } else if let Some(bv) = arg.as_any().downcast_ref::<BooleanVal>() {
        bv.value.to_string()
    } else if let Some(av) = arg.as_any().downcast_ref::<ArrayVal>() {
        av.elements
            .lock()
            .unwrap()
            .iter()
            .map(|e| match_arg_to_string(e.as_ref()))
            .collect::<Vec<_>>()
            .join(",")
    } else if let Some(f) = arg.as_any().downcast_ref::<FunctionVal>() {
        format!("{:?}", f.body)
    } else if arg.as_any().downcast_ref::<NullVal>().is_some() {
        "null".into()
    } else {
        format!("{:?}", arg)
    }
}

pub fn drive_stream(mut rx: UnboundedReceiver<String>) {
    thread::spawn(move || {
        while let Some(msg) = rx.blocking_recv() {
            println!("{}", msg);
        }
    });
}


#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;
    use crate::{evaluate, make_global_env, mk_native_fn, mk_null, tokenize, IntegerVal, Parser, match_arg_to_string};
    use crate::ValueType::{NativeFn};

    #[test]
    fn main() {
        let mut env = make_global_env();
        let output = Arc::new(Mutex::new(Vec::<String>::new()));
        //let output_for_native = output.clone();
        println!("adfa");

        env.set_var(
            "log".to_string(),
            mk_native_fn(Arc::new(move |args, _| {
                //let mut _guard = output_for_native.lock().unwrap();
                for arg in args {
                    //guard.push(match_arg_to_string(&*arg));
                    println!("{}", match_arg_to_string(&*arg));
                }
                mk_null()
            })),
            Option::from(NativeFn)
        );

        env.set_var(
            "sleep".to_string(),
            mk_native_fn(Arc::new(move |args, _| {
                let secs = args.get(0)
                    .expect("sleep: un argument attendu")
                    .as_any()
                    .downcast_ref::<IntegerVal>()
                    .expect("sleep: l’argument doit être un nombre")
                    .value;

                //tokio::time::sleep(Duration::from_secs_f64(secs));

                thread::sleep(Duration::from_secs_f64(secs));
                mk_null()
            })),
            Option::from(NativeFn)
        );


        //let input = fs::read_to_string("code.nwtz").unwrap();

        let input = r#"

log("Hello");
sleep(5);
log("world");

for (i = 0; i < 3; i = i+1;){
    log(i);
    sleep(1);
}

"#.to_string();

        let tokens = tokenize(input);
        let mut parser = Parser::new(tokens);
        let ast = parser.produce_ast();
        //println!("AST{:#?}\n\n", ast);
        let _ = evaluate(Box::new(ast), &mut env);
        println!("EVALUATED {:#?}", output.lock().unwrap().clone())


    }
}

/*
loop {

            print!("> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.is_empty() {

                let file = fs::read_to_string("code.nwtz").unwrap();
                let tokens = tokenize(file.clone());
                let mut parser = Parser::new(tokens);
                let ast = parser.produce_ast();
                //println!("{:#?}", ast);

                let _result = evaluate(Box::new(ast), &mut env);
            }


            let tokens = tokenize(input.to_string());
            let mut parser = Parser::new(tokens);
            let program = parser.produce_ast().merge_imports();

            let _res = evaluate(Box::new(program), &mut env);
            //println!("{:#?}", res);
        }
 */


