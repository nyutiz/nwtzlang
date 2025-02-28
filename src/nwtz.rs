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
    FParenthesis, // First parentheses
    SParenthesis, // Second parentheses
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

    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,
    #[regex(r"\d+\.\d+")]
    Float,
    #[regex(r"\d+")]
    Integer,
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
    Implementation { name: String },
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
        iterator: String,
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
}

#[derive(Debug)]
pub enum Operator {
    Plus,
    Minus,
    Multiply,
    Divide,
}

#[derive(Debug)]
pub enum Type {
    Integer,
    Float,
    String,
    Object,
    Bool,
}

#[derive(Debug)]
pub struct Argument {
    pub name: String,
    pub arg_type: Type,
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

    /// Avance si le token courant correspond (en comparant la "discriminant")
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
        if let Some(&Token::Identifier(ref id)) = self.current() {
            let name = id.clone();
            self.advance();
            Ok(name)
        } else {
            Err(error_message.to_string())
        }
    }

    /// Vérifie si le token courant est un type connu (pour les définitions de fonction, etc.)
    fn peek_is_type(&self) -> bool {
        if let Some(&Token::Identifier(ref id)) = self.current() {
            matches!(id.as_str(), "Integer" | "Float" | "String" | "Object" | "Bool")
        } else {
            false
        }
    }

    /// Parse un type depuis un identifiant
    fn parse_type(&mut self) -> Result<Type, String> {
        if let Some(&Token::Identifier(ref id)) = self.current() {
            self.advance();
            match id.as_str() {
                "Integer" => Ok(Type::Integer),
                "Float" => Ok(Type::Float),
                "String" => Ok(Type::String),
                "Object" => Ok(Type::Object),
                "Bool" => Ok(Type::Bool),
                _ => Err(format!("Type inconnu : {}", id)),
            }
        } else {
            Err("Type attendu".to_string())
        }
    }

    /// Point d'entrée pour le parsing d'une instruction
    pub fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.current() {
            Some(Token::With) => self.parse_import(),
            Some(Token::Obj) => self.parse_obj_declaration(),
            Some(Token::Func) => self.parse_function(),
            Some(Token::Impl) => self.parse_implementation(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            other => Err(format!("Token inattendu : {:?}", other)),
        }
    }

    fn parse_import(&mut self) -> Result<Statement, String> {
        // Syntaxe attendue : with [LibName] [File];
        self.advance(); // consomme "with"
        // On attend un identifiant entre crochets
        self.expect_token(&Token::LBracket, "Expected '[' after 'with'")?;
        let lib = self.expect_identifier("Expected library name")?;
        self.expect_token(&Token::RBracket, "Expected ']' after library name")?;
        self.expect_token(&Token::LBracket, "Expected '[' before file name")?;
        let file = self.expect_identifier("Expected file name")?;
        self.expect_token(&Token::RBracket, "Expected ']' after file name")?;
        self.expect_token(&Token::Semicolon, "Expected ';' after import")?;
        Ok(Statement::Import { lib, file })
    }

    fn parse_obj_declaration(&mut self) -> Result<Statement, String> {
        // Syntaxe attendue : obj [Name] { var1: Type, var2: Type, ... }
        self.advance(); // consomme "obj"
        let name = self.expect_identifier("Expected object name")?;
        self.expect_token(&Token::LBrace, "Expected '{' after object name")?;
        let mut fields = Vec::new();
        while let Some(token) = self.current() {
            if let Token::RBrace = token {
                break;
            }
            let field_name = self.expect_identifier("Expected field name")?;
            self.expect_token(&Token::Colon, "Expected ':' after field name")?;
            let field_type = self.parse_type()?;
            // Optionnellement, consommer la virgule
            if let Some(Token::Comma) = self.current() {
                self.advance();
            }
            fields.push((field_name, field_type));
        }
        self.expect_token(&Token::RBrace, "Expected '}' after object declaration")?;
        Ok(Statement::ObjectDeclaration { name, fields })
    }

    fn parse_function(&mut self) -> Result<Statement, String> {
        // Syntaxe : func [Return Type] [Name] ([Args]) { ... }
        self.advance(); // consomme "func"
        let return_type = if self.peek_is_type() {
            Some(self.parse_type()?)
        } else {
            None
        };
        let name = self.expect_identifier("Expected function name")?;
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
            let arg_name = self.expect_identifier("Expected argument name")?;
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
                self.advance(); // consomme "}"
                break;
            }
            statements.push(self.parse_statement()?);
        }
        Ok(statements)
    }

    fn parse_implementation(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_while_statement(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_print(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_variable_declaration(&mut self) -> Result<Statement, String> {
        
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        
    }
}

// --- Exemple d'utilisation --- //