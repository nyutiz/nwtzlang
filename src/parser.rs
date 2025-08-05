use std::collections::HashMap;
use std::fs;
use strum::IntoEnumIterator;
use crate::ast::{ArrayLiteral, AssignmentExpr, BinaryExpr, BooleanLiteral, CallExpr, ForStatement, FunctionDeclaration, IdentifierExpr, IfStatement, ImportAst, LiteralExpr, MemberExpr, NodeType, NullLiteral, ObjectLiteral, Program, Property, Stmt, StringVal, VariableDeclaration};
use crate::ast::NodeType::Identifier;
use crate::lexer::{tokenize, Token};
use crate::types::ValueType;
use crate::types::ValueType::{Boolean, Integer, Null, Object};

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
    imports: Option<HashMap<String, String>>,
    pub main: bool,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0, imports: None, main: false }
    }
    
    pub fn has_main(&mut self) -> bool {
        self.produce_ast();
        self.main
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
            Token::Identifier(_) if *self.peek() == Token::Semicolon => {
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

    pub fn provide_import(&mut self, imports: HashMap<String, String>) {
        self.imports = Some(imports);
    }
    pub fn get_import(&self, name: String) -> String {
        let imports = self
            .imports
            .as_ref()
            .expect(format!("No imports provided for {}", name).as_str());

        imports.get(&name).unwrap().clone()
    }

    fn parse_with_declaration(&mut self) -> Box<dyn Stmt> {
        self.eat();

        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected import name following with keyword");
        };

        if name.starts_with("_") {
            let mut new_name = name[1..].to_string();
            new_name.push_str(".nwtz");
            let import = fs::read_to_string(&new_name)
                .expect("Erreur lors de la lecture du fichier");
            //println!("{}", import);
            let tokens = tokenize(import);
            let mut external_parser = Parser::new(tokens);
            let external_ast = external_parser.produce_ast();
            self.expect(Token::Semicolon, "';' after import");
            Box::from(ImportAst {
                kind: NodeType::ImportAst,
                body: external_ast.body,
            })
        } else if name.starts_with("!") {
            let new_name = name[1..].to_string();

            let import = self.get_import(new_name);
            
            let tokens = tokenize(import);
            let mut external_parser = Parser::new(tokens);
            let external_ast = external_parser.produce_ast();
            self.expect(Token::Semicolon, "';' after import");
            Box::from(ImportAst {
                kind: NodeType::ImportAst,
                body: external_ast.body,
            })
        } else {
            unimplemented!("Chargement depuis une installation locale ou via le web non implémenté");
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

        let initializer: Option<Box<dyn Stmt>>  = if !matches!(self.at(), &Token::Semicolon) {
            Some(self.parse_var_declaration())
        } else {
            None
        };
        //self.expect(Semicolon, "`;` attendu après l'initialisation de i");

        let condition: Option<Box<dyn Stmt>>  = if !matches!(self.at(), &Token::Semicolon) {
            let left: Box<dyn Stmt> = match self.eat() {
                Token::Identifier(name) =>
                    Box::new(IdentifierExpr { kind: Identifier, name }),
                Token::Integer(n) =>
                    Box::new(LiteralExpr    { kind: NodeType::NumericLiteral, value: n as f64 }),
                Token::StringLiteral(s) =>
                    Box::new(StringVal{
                        r#type: None,
                        kind: NodeType::StringLiteral,
                        value: s,
                    }),
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
                Token::StringLiteral(s) =>
                    Box::new(StringVal{
                        r#type: None,
                        kind: NodeType::StringLiteral,
                        value: s,
                    }),
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
        self.expect(Token::Semicolon, "`;` attendu après la condition du for");

        let increment: Option<Box<dyn Stmt>> = if !matches!(self.at(), &Token::RParen) {
            Some(self.parse_var_declaration())
        } else {
            None
        };
        self.expect(Token::RParen, "`)` attendu après l’incrément du for");

        self.expect(Token::LBrace, "`{` attendu pour ouvrir le corps du for");

        let mut body = Vec::new();
        while self.not_eof() && *self.at() != Token::RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(Token::RBrace, "`}` attendu pour fermer le for");

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
            Token::StringLiteral(s) =>
                Box::new(StringVal {
                    r#type: None,
                    kind: NodeType::StringLiteral,
                    value: s[1..s.len()-1].to_string(),
                }),
            t => panic!("Variable, Chaine ou Entier attendu après `if` pas {:?}", t),
        };

        let operator = match self.eat() {
            Token::EqualEqual => "==",
            Token::Greater => ">",
            Token::Lower   => "<",
            _ => {
                panic!("Opérateur attendu (==, > ou <)")
            },
        }.to_string();

        //println!("{}", operator);

        let right_expr: Box<dyn Stmt> = match self.eat() {
            Token::Identifier(name) =>
                Box::new(IdentifierExpr {
                    kind: Identifier,
                    name
                }),
            Token::Integer(n) =>
                Box::new(LiteralExpr {
                    kind: NodeType::NumericLiteral,
                    value: n as f64
                }),
            Token::StringLiteral(s) =>
                Box::new(StringVal {
                    r#type: None,
                    kind: NodeType::StringLiteral,
                    value: s[1..s.len()-1].to_string(),
                }),
            e => panic!("Variable ou entier attendu après l’opérateur, {:?}", e),
        };

        let condition = Box::new(BinaryExpr {
            kind: NodeType::BinaryExpression,
            left:  left_expr,
            right: right_expr,
            operator,
        });

        self.expect(Token::LBrace, "‘{’ attendu après la condition du if");
        let mut then_branch = Vec::new();
        while self.not_eof() && *self.at() != Token::RBrace {
            then_branch.push(self.parse_stmt());
        }
        self.expect(Token::RBrace, "‘}’ attendu pour clôturer le bloc then");

        let else_branch = if self.eat_if(Token::Else) {
            self.expect(Token::LBrace, "‘{’ attendu après else");
            let mut else_vec = Vec::new();
            while self.not_eof() && *self.at() != Token::RBrace {
                else_vec.push(self.parse_stmt());
            }
            self.expect(Token::RBrace, "‘}’ attendu pour clôturer le bloc else");
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
            if name.eq("main") { self.main = true; }
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

        self.expect(Token::LBrace, "Expected function body following declaration");
        let mut body = Vec::new();
        while self.not_eof() && *self.at() != Token::RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(Token::RBrace, "Closing brace expected inside function declaration");

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
            if *self.at() == Token::Fn {
                //self.eat();

                let func = self.parse_func_declaration().downcast::<FunctionDeclaration>()
                    .expect("Expected FunctionDeclaration");

                properties.push(Property {
                    kind: NodeType::Property,
                    key: func.name.clone(),
                    value: Some(func),
                });
            }
            else if let Token::Identifier(key) = self.eat() {
                self.expect(Token::Colon, "Expected ':' after property key");
                let value = self.parse_expr();
                properties.push(Property {
                    kind: NodeType::Property,
                    key,
                    value: Some(value),
                });
            }
            else {
                panic!("Expected property key or 'fn' in object literal");
            }

            if *self.at() == Token::Comma {
                self.eat();
            }
        }

        self.expect(Token::RBrace, "Expected '}' to close object literal");
        self.expect(Token::Semicolon, "Expected ';' after object declaration");

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

        self.expect(Token::Semicolon, "Expected semicolon after variable declaration");

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
        if *self.at() != Token::LBrace {
            return self.parse_additive_expr()
        }

        self.eat();
        let mut properties: Vec<Property> = Vec::new();
        while self.not_eof() && *self.at() != Token::RBrace{
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

            if *self.at() != Token::RBrace{
                self.expect(Token::Comma, "Expected comma or Closing Bracket following Property");
            }

        }

        self.expect(Token::RBrace, "Object literal missing closing brace");
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

        if *self.at() == Token::Fn {
            args.push(self.parse_function_expr());
        } else {
            args.push(self.parse_assignment_expr());
        }

        while let Token::Comma = self.at() {
            self.eat();
            if *self.at() == Token::Fn {
                args.push(self.parse_function_expr());
            } else {
                args.push(self.parse_assignment_expr());
            }
        }

        args
    }

    fn parse_function_expr(&mut self) -> Box<dyn Stmt> {
        self.eat();

        let name = if let Token::Identifier(name) = self.eat() {
            name
        } else {
            panic!("Expected function name after 'fn' in function expression");
        };

        let args = self.parse_args();
        let mut params = Vec::new();
        for arg in args {
            if let Some(ident) = arg.as_any().downcast_ref::<IdentifierExpr>() {
                params.push(ident.name.clone());
            } else {
                panic!("Inside function expression expected parameters to be identifiers");
            }
        }

        self.expect(Token::LBrace, "Expected function body following declaration");
        let mut body = Vec::new();
        while self.not_eof() && *self.at() != Token::RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(Token::RBrace, "Closing brace expected inside function expression");

        Box::from(FunctionDeclaration {
            kind: NodeType::FunctionDeclaration,
            parameters: params,
            name,
            body,
        })
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
