extern crate core;

use std::fs;
use logos::Logos;
use crate::nwtz::{Parser, Token};

mod nwtz;

fn main() {
    let source = fs::read_to_string("main.nwtz").unwrap();
    let mut lexer = Token::lexer(&*source);
    let mut tokens: Vec<Token> = Vec::new();

    while let Some(token) = lexer.next() {
        // Vous pouvez adapter ici la collecte des tokens selon votre implémentation
        tokens.push(token.clone().unwrap());
        println!("{:?}", token);
    }

    let mut parser = Parser::new(tokens);
    while let Ok(stmt) = parser.parse_statement() {
        println!("{:?}", stmt);
    }
    

}
