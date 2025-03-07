#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

use std::io;
use std::io::Write;
use std::sync::Mutex;
use logos::Logos;
use regex::Regex;
use once_cell::sync::Lazy;

use crate::nwtz::PunctuationType::{FParenthesis, QuotationMark, SParenthesis};

pub struct Nwtz {
    input: String,
    regex: Vec<(Regex, Box<dyn Fn(&str) -> Token>)>,
}


#[derive(Clone, Debug, PartialEq)]
pub enum PunctuationType {
    FParenthesis,
    SParenthesis,
    QuotationMark,
    Generic,
}

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
    For,  // Added for for-loops
    #[token("log")]
    Log,  // Added for print statements

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
    #[token("true")]
    True,
    #[token("false")]
    False,
}

#[derive(Debug)]
pub enum Statement {
    Import { lib: String, file: String },
    Print(Expression),
    ObjectDeclaration {
        name: String,
        fields: Vec<(String, Type)>,
    },
    Implementation {
        name: String,
        methods: Vec<Statement>,
    },
    Function {
        name: String,
        return_type: Option<Type>,
        args: Vec<Argument>,
        body: Vec<Statement>,
    },
    VariableDeclaration {
        name: String,
        value: Expression,
    },
    If {
        condition: Expression,
        then_branch: Vec<Statement>,
        else_branch: Option<Vec<Statement>>,
    },
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    For {
        iterators: Vec<String>,
        iterable: Expression,
        body: Vec<Statement>,
    },
}

#[derive(Debug)]
pub enum Expression {
    Literal(Literal),
    Identifier(String),
    BinaryOp {
        left: Box<Expression>,
        op: Operator,
        right: Box<Expression>,
    },
}

#[derive(Debug)]
pub enum Literal {
    Integer(i32),
    Float(f64),
    String(String),
    Boolean(bool),
}

#[derive(Debug)]
pub enum Operator {
    Plus,
    Minus,
    Multiply,
    Divide,
    And,
    Or,
    Greater,
    Equal,
}

#[derive(Debug)]
pub enum Type {
    Integer,
    Float,
    String,
    Object,
    Boolean,
}

#[derive(Debug)]
pub struct Argument {
    pub name: String,
    pub arg_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Program,
    NumericLiteral,
    Identifier,
    BinaryExpression,
    CallExpression,
    UnaryExpression,
    FunctionDeclaration,
}

pub trait Stmt {
    fn kind(&self) -> NodeType;
}

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr;

impl Stmt for Expr {
    fn kind(&self) -> NodeType {
        NodeType::BinaryExpression
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpr {
    kind: NodeType,
}

impl Stmt for BinaryExpr {
    fn kind(&self) -> NodeType {
        self.kind.clone()
    }
}


pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}



impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, position: 0 }
    }

    fn current(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn expect_token(&mut self, expected: &Token, error_message: &str) -> Result<(), String> {
        if let Some(token) = self.current() {
            if std::mem::discriminant(token) == std::mem::discriminant(expected) {
                self.advance();
                Ok(())
            } else {
                Err(error_message.to_string())
            }
        } else {
            Err("Fin des tokens inattendue".to_string())
        }
    }

    fn expect_identifier(&mut self, error_message: &str) -> Result<String, String> {
        if let Some(Token::Identifier(id)) = self.current() {
            let name = id.clone();
            self.advance();
            Ok(name)
        } else {
            Err(error_message.to_string())
        }
    }

    fn peek_is_type(&self) -> bool {
        if let Some(Token::Identifier(id)) = self.current() {
            matches!(id.as_str(), "Integer" | "Float" | "String" | "Object" | "Boolean")
        } else {
            false
        }
    }

    fn parse_type(&mut self) -> Result<Type, String> {
        let token = self.current().cloned();

        if let Some(Token::Identifier(id)) = token {
            self.advance();

            match id.as_str() {
                "Integer" => Ok(Type::Integer),
                "Float" => Ok(Type::Float),
                "String" => Ok(Type::String),
                "Object" => Ok(Type::Object),
                "Boolean" => Ok(Type::Boolean),
                _ => Err(format!("Type inconnu : {}", id)),
            }
        } else {
            Err("Type attendu".to_string())
        }
    }

    fn parse_bracketed_identifier(&mut self, error_message: &str) -> Result<String, String> {
        self.expect_token(&Token::LBracket, "Expected '['")?;
        let id = self.expect_identifier(error_message)?;
        self.expect_token(&Token::RBracket, "Expected ']'")?;
        Ok(id)
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        self.parse_binary_expression(0)
    }

    fn parse_binary_expression(&mut self, precedence: u8) -> Result<Expression, String> {
        let mut left = self.parse_primary_expression()?;

        while let Some(token) = self.current() {
            let (op_precedence, op) = match token {
                Token::And => (1, Operator::And),
                Token::Or => (1, Operator::Or),
                Token::Greater => (2, Operator::Greater),
                Token::Equal => (2, Operator::Equal),
                Token::Slash => (3, Operator::Divide),
                _ => break,
            };

            if op_precedence < precedence {
                break;
            }

            self.advance();
            let right = self.parse_binary_expression(op_precedence + 1)?;
            left = Expression::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_primary_expression(&mut self) -> Result<Expression, String> {
        if let Some(token) = self.current().cloned() {
            match token {
                Token::Identifier(id) => {
                    self.advance();
                    Ok(Expression::Identifier(id))
                },
                Token::Integer(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Integer(value)))
                },
                Token::Float(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Float(value)))
                },
                Token::StringLiteral(value) => {
                    self.advance();
                    Ok(Expression::Literal(Literal::String(value)))
                },
                Token::True => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Boolean(true)))
                },
                Token::False => {
                    self.advance();
                    Ok(Expression::Literal(Literal::Boolean(false)))
                },
                Token::LParen => {
                    self.advance();
                    let expr = self.parse_expression()?;
                    self.expect_token(&Token::RParen, "Expected ')'")?;
                    Ok(expr)
                },
                _ => Err(format!("Unexpected token in expression: {:?}", token))
            }
        } else {
            Err("Unexpected end of tokens in expression".to_string())
        }
    }

    pub fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.current() {
            Some(Token::With)   => self.parse_import(),
            Some(Token::Obj)    => self.parse_obj_declaration(),
            Some(Token::Func)   => self.parse_function(),
            Some(Token::Impl)   => self.parse_implementation(),
            Some(Token::If)     => self.parse_if_statement(),
            Some(Token::While)  => self.parse_while_statement(),
            Some(Token::For)    => self.parse_for_statement(),
            Some(Token::Log)    => self.parse_print(),
            Some(Token::LBracket) => self.parse_variable_declaration(),
            other => Err(format!("Token inattendu : {:?}", other)),
        }
    }


    fn parse_import(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "with"
        let name = self.parse_bracketed_identifier("Expected library or file name")?;
        self.expect_token(&Token::Semicolon, "Expected ';' after import")?;
        Ok(Statement::Import { lib: name.clone(), file: name })
    }

    fn parse_obj_declaration(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "obj"
        let name = self.parse_bracketed_identifier("Expected object name")?;
        self.expect_token(&Token::LBrace, "Expected '{' after object name")?;
        let mut fields = Vec::new();
        while let Some(token) = self.current() {
            if let Token::RBrace = token {
                break;
            }
            let field_name = self.expect_identifier("Expected field name")?;
            self.expect_token(&Token::Colon, "Expected ':' after field name")?;
            let field_type = self.parse_type()?;
            // Optionally consume a comma
            if let Some(Token::Comma) = self.current() {
                self.advance();
            }
            fields.push((field_name, field_type));
        }
        self.expect_token(&Token::RBrace, "Expected '}' after object declaration")?;
        Ok(Statement::ObjectDeclaration { name, fields })
    }

    /// Function declaration: func [Return Type] [Name] ([Args]) { ... }
    fn parse_function(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "func"
        let return_type = if self.peek_is_type() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let name = self.parse_bracketed_identifier("Expected function name")?;
        self.expect_token(&Token::LParen, "Expected '(' after function name")?;
        let args = self.parse_arguments()?;
        self.expect_token(&Token::RParen, "Expected ')' after arguments")?;
        self.expect_token(&Token::LBrace, "Expected '{' before function body")?;
        let body = self.parse_block()?;
        Ok(Statement::Function { name, return_type, args, body })
    }

    fn parse_arguments(&mut self) -> Result<Vec<Argument>, String> {
        let mut args = Vec::new();
        while let Some(token) = self.current() {
            if let Token::RParen = token {
                break;
            }
            // For arguments we assume they are written as [argName]: Type
            let arg_name = self.parse_bracketed_identifier("Expected argument name")?;
            self.expect_token(&Token::Colon, "Expected ':' after argument name")?;
            let arg_type = self.parse_type()?;
            args.push(Argument { name: arg_name, arg_type });
            if let Some(Token::Comma) = self.current() {
                self.advance();
            }
        }
        Ok(args)
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        while let Some(token) = self.current() {
            if let Token::RBrace = token {
                break;
            }
            statements.push(self.parse_statement()?);
        }
        // Don't advance past the closing brace - leave it for the caller
        Ok(statements)
    }

    /// Implementation block: impl [Name] { func ... }
    fn parse_implementation(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "impl"
        let name = self.parse_bracketed_identifier("Expected object name in implementation")?;
        self.expect_token(&Token::LBrace, "Expected '{' after object name")?;
        let mut methods = Vec::new();
        while let Some(token) = self.current() {
            if let Token::RBrace = token {
                break;
            }
            // We expect functions inside an implementation block.
            if let Token::Func = token {
                let function = self.parse_function()?;
                methods.push(function);
            } else {
                return Err(format!("Unexpected token in implementation block: {:?}", token));
            }
        }
        self.expect_token(&Token::RBrace, "Expected '}' after implementation block")?;
        Ok(Statement::Implementation { name, methods })
    }

    /// Print statement: log(<expression>);
    fn parse_print(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "log"
        self.expect_token(&Token::LParen, "Expected '(' after log")?;
        let expr = self.parse_expression()?;
        self.expect_token(&Token::RParen, "Expected ')' after log argument")?;
        self.expect_token(&Token::Semicolon, "Expected ';' after log statement")?;
        Ok(Statement::Print(expr))
    }

    /// Variable declaration: [Var Name] = <expression>;
    fn parse_variable_declaration(&mut self) -> Result<Statement, String> {
        let name = self.parse_bracketed_identifier("Expected variable name")?;
        self.expect_token(&Token::Equal, "Expected '=' in variable declaration")?;
        let value = self.parse_expression()?;
        self.expect_token(&Token::Semicolon, "Expected ';' after variable declaration")?;
        Ok(Statement::VariableDeclaration { name, value })
    }

    /// If statement:
    /// if <condition> { <then_branch> } [else { <else_branch> }]
    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "if"
        let condition = self.parse_expression()?;
        self.expect_token(&Token::LBrace, "Expected '{' after if condition")?;
        let then_branch = self.parse_block()?;
        self.expect_token(&Token::RBrace, "Expected '}' after if body")?;

        let else_branch = if let Some(Token::Else) = self.current() {
            self.advance(); // consume "else"
            if let Some(Token::If) = self.current() {
                // Handle "else if" by parsing a new if statement
                let else_if = self.parse_if_statement()?;
                Some(vec![else_if])
            } else {
                self.expect_token(&Token::LBrace, "Expected '{' after else")?;
                let branch = self.parse_block()?;
                self.expect_token(&Token::RBrace, "Expected '}' after else body")?;
                Some(branch)
            }
        } else {
            None
        };

        Ok(Statement::If { condition, then_branch, else_branch })
    }

    /// While loop: while (<condition>) { <body> }
    fn parse_while_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "while"
        self.expect_token(&Token::LParen, "Expected '(' after while")?;
        let condition = self.parse_expression()?;
        self.expect_token(&Token::RParen, "Expected ')' after while condition")?;
        self.expect_token(&Token::LBrace, "Expected '{' after while condition")?;
        let body = self.parse_block()?;
        self.expect_token(&Token::RBrace, "Expected '}' after while body")?;
        Ok(Statement::While { condition, body })
    }

    /// For loop: for (<id1>, <id2>, ...) in <expression> { <body> }
    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume "for"
        self.expect_token(&Token::LParen, "Expected '(' after for")?;

        // Parse a list of iterator variables
        let mut iterators = Vec::new();
        iterators.push(self.expect_identifier("Expected iterator identifier")?);

        // Parse additional iterator variables if any
        while let Some(Token::Comma) = self.current() {
            self.advance(); // consume comma
            iterators.push(self.expect_identifier("Expected iterator identifier")?);
        }

        self.expect_token(&Token::RParen, "Expected ')' after iterator identifiers")?;
        self.expect_token(&Token::In, "Expected 'in' after for loop iterators")?;
        let iterable = self.parse_expression()?;
        self.expect_token(&Token::LBrace, "Expected '{' after for loop header")?;
        let body = self.parse_block()?;
        self.expect_token(&Token::RBrace, "Expected '}' after for loop body")?;

        Ok(Statement::For { iterators, iterable, body })
    }
}