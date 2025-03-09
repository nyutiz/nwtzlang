#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]


use std::{fs, io};
use std::io::Write;
use crate::nwtz::{evaluate, mk_bool, mk_null, mk_number, tokenize, Environment, NumberVal, Parser, ValueType};

mod nwtz;

fn main() {
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() || input.contains("exit") {
            println!("Exiting");
            break;
        }

        let mut env = Environment::new(None);
        env.declare_var("x".to_string(), mk_number(100));
        env.declare_var("null".to_string(), mk_null());
        env.declare_var("true".to_string(), mk_bool(true));
        env.declare_var("false".to_string(), mk_bool(false));


        let tokens = tokenize(input.to_string());
        let mut parser = Parser::new(tokens);
        let program = parser.produce_ast();

        let res = evaluate(Box::new(program), &mut env);        
        println!("{:#?}", res);
    }
}
