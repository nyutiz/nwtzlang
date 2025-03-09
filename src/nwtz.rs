#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::io;
use std::process::exit;
use logos::Logos;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::nwtz::ValueType::{Null, Number};

#[derive(Clone, Debug, PartialEq, Logos)]
pub enum Token {
    #[regex(r"[ \t\n\r\f]+", logos::skip)]
    #[regex(r"//[^\n\r]*", logos::skip)]
    #[token("with")]
    With,
    #[token("obj")]
    Obj,
    #[token("func")]
    Func,
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
    #[token("while")]
    While,
    #[token("in")]
    In,
    #[token("for")]
    For,
    #[token("log")]
    Log,
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
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice().to_string())]
    StringLiteral(String),
    #[regex(r"\d+\.\d+", |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[regex(r"\d+", |lex| lex.slice().parse::<i32>().unwrap())]
    Integer(i32),
    //#[token("true")]
    //True,
    //#[token("false")]
    //False,
    #[token("null")]
    Null,
    EOF,
}

#[derive(Debug, Clone)]
pub enum ValueType {
    Null,
    Number,
    Boolean,
}

pub trait RuntimeValClone {
    fn clone_box(&self) -> Box<dyn RuntimeVal>;
}

impl<T> RuntimeValClone for T
where
    T: 'static + RuntimeVal + Clone,
{
    fn clone_box(&self) -> Box<dyn RuntimeVal> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn RuntimeVal> {
    fn clone(&self) -> Box<dyn RuntimeVal> {
        self.clone_box()
    }
}

#[derive(Debug)]
pub struct Environment {
    parent: Option<Box<Environment>>,
    variables: HashMap<String, Box<dyn RuntimeVal>>,
}

impl Environment {
    pub fn new(parent: Option<Box<Environment>>) -> Self {
        Environment {
            parent,
            variables: HashMap::new(),
        }
    }
    
    pub fn declare_var(&mut self, var_name: String, value: Box<dyn RuntimeVal>) -> Box<dyn RuntimeVal> {
        if self.variables.contains_key(&var_name) {
            panic!("Cannot declare variable {}. It is already defined.", var_name);
        }
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


pub trait RuntimeVal: Debug + RuntimeValClone  {
    fn value_type(&self) -> ValueType;
    fn as_any(&self) -> &dyn Any;
}
#[derive(Debug, Clone)]
pub struct NullVal {
    pub r#type: ValueType,
    pub value: String,
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


impl RuntimeVal for BooleanVal {
    fn value_type(&self) -> ValueType {
        self.r#type.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Program,
    NumericLiteral,
    NullLiteral,
    Identifier,
    BinaryExpression,
}

pub trait Stmt: Debug {
    fn kind(&self) -> NodeType;
    fn value(&self) -> Option<String>;
    fn as_any(&self) -> &dyn Any;
}


// Permet le downcasting d'un Box<dyn Stmt> en Box<T>
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
#[derive(Debug, Clone, PartialEq)]
pub struct LiteralExpr {
    kind: NodeType,
    value: f64,
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

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifierExpr {
    kind: NodeType,
    name: String,
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

#[derive(Debug)]
pub struct BinaryExpr {
    kind: NodeType,
    left: Box<dyn Stmt>,
    right: Box<dyn Stmt>,
    operator: String,
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

#[derive(Debug)]
pub struct NullLiteral {
    kind: NodeType,
    value: String,
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

#[derive(Debug)]
pub struct Program {
    kind: NodeType,
    body: Vec<Box<dyn Stmt>>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            kind: NodeType::Program,
            body: Vec::new(),
        }
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

    fn expect(&mut self, expected: Token, err_msg: &str) -> Token {
        let token = self.eat();
        if token != expected {
            panic!("Parser Error: {}\nGot: {:?}, Expected: {:?}", err_msg, token, expected);
        }
        token
    }

    fn parse_stmt(&mut self) -> Box<dyn Stmt> {
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> Box<dyn Stmt> {
        self.parse_additive_expr()
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
            left = Box::new(BinaryExpr {
                kind: NodeType::BinaryExpression,
                left,
                right,
                operator,
            });
        }

        left
    }

    fn parse_multiplicative_expr(&mut self) -> Box<dyn Stmt> {
        let mut left = self.parse_primary_expr();

        while let Token::Star | Token::Slash | Token::Percent = self.at() {
            let operator = match self.eat() {
                Token::Star => "*".to_string(),
                Token::Slash => "/".to_string(),
                Token::Percent => "%".to_string(),
                _ => unreachable!(),
            };
            let right = self.parse_primary_expr();

            left = Box::new(BinaryExpr {
                kind: NodeType::BinaryExpression,
                left,
                right,
                operator,
            });
        }

        left
    }

    fn parse_primary_expr(&mut self) -> Box<dyn Stmt> {
        match self.eat() {
            Token::Integer(value) => Box::new(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value: value as f64,
            }),
            Token::Float(value) => Box::new(LiteralExpr {
                kind: NodeType::NumericLiteral,
                value,
            }),
            Token::Identifier(name) => Box::new(IdentifierExpr {
                kind: NodeType::Identifier,
                name,
            }),
            Token::Null => Box::new(NullLiteral {
                kind: NodeType::NullLiteral,
                value: "null".to_string(),
            }),
            Token::LParen => {
                let expr = self.parse_expr();
                self.expect(Token::RParen, "Expected `)`.");
                expr
            }
            token => Box::new(IdentifierExpr {
                kind: NodeType::Identifier,
                name: "null".to_string(),
            }),
        }
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

    Box::new(NullVal {
        r#type: ValueType::Null,
        value: "null".to_string(),
    })
}
pub fn eval_numeric_binary_expr(
    lhs: &NumberVal,
    rhs: &NumberVal,
    operator: &str,
) -> Box<dyn RuntimeVal> {
    let lhs_value = lhs.value;
    let rhs_value = rhs.value;

    let result = match operator {
        "+" => lhs_value + rhs_value,
        "-" => lhs_value - rhs_value,
        "*" => lhs_value * rhs_value,
        "/" => lhs_value / rhs_value, // TODO: Division by zero checks
        "%" => lhs_value % rhs_value,
        _ => panic!("Unknown binary operator: {}", operator),
    };

    Box::new(NumberVal {
        r#type: ValueType::Number,
        value: result,
    })
}
pub fn eval_program(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let program = ast_node.downcast::<Program>()
        .expect("Expected a Program node");

    let mut last_evaluated: Box<dyn RuntimeVal> = mk_null();

    for statement in program.body {
        // Use a reference to env instead of moving it
        last_evaluated = evaluate(statement, env);
    }

    last_evaluated
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
            Box::new(NumberVal {
                r#type: ValueType::Number,
                value: literal.value,
            })
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
        NodeType::Identifier => {
            eval_identifier(ast_node, env)
        }
    }
}

fn eval_identifier(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal> {
    let identifier_expr = ast_node.downcast::<IdentifierExpr>()
        .expect("Expected a BinaryExpr node");
    let val = env.lookup_var(identifier_expr.name);
    
    val
}

pub fn mk_number<T: Into<f64>>(number: T) -> Box<NumberVal> {
    Box::new(NumberVal {
        r#type: Number,
        value: number.into(),
    })
}


pub fn mk_null() -> Box<NullVal> {
    Box::new(NullVal{ r#type: ValueType::Null, value: "null".to_string() })
}

pub fn mk_bool(b: bool) -> Box<BooleanVal> {
    Box::new(BooleanVal{ r#type: ValueType::Boolean, value: b })
}