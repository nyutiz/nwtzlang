extern crate core;

use std::fs;
use logos::Logos;
use crate::nwtz::{Token};

mod nwtz;

fn main() {
    
    let source = fs::read_to_string("main.nwtz").unwrap();
    let mut lexer = Token::lexer(&*source);

    while let Some(token) = lexer.next() {
        println!("{:?} {:?}", token, lexer.slice());
    }
}
