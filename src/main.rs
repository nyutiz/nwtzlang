#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]


use std::{fs, io};
use std::io::Write;
use std::sync::Arc;
use std::time::SystemTime;
use crate::nwtz::{evaluate, make_global_env, mk_bool, mk_native_fn, mk_null, mk_number, tokenize, BooleanVal, Environment, FunctionCall, NullVal, NumberVal, Parser, RuntimeVal, StringVal, ValueType};

mod nwtz;

fn main() {
    let mut env = make_global_env();
    
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
            
            let result = evaluate(Box::new(ast), &mut env);
        }


        let tokens = tokenize(input.to_string());
        let mut parser = Parser::new(tokens);
        let program = parser.produce_ast().merge_imports();

        let res = evaluate(Box::new(program), &mut env);
        //println!("{:#?}", res);
    }

}

//Modify eval object expr to handle tables
//and to reassign value by index

//User Defined Functions & Closures - Programming Language From Scratch  27:16