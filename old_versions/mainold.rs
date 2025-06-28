
/*
use std::{fs, io};
use std::io::Write;
use std::sync::Arc;
use nwtzlang::{evaluate_runtime, make_global_env, };
use nwtzlang::nwtz::Program;
use crate::nwtz::{evaluate, mk_native_fn, mk_null, tokenize, ArrayVal, BooleanVal, NullVal, NumberVal, Parser, StringVal};

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
            let ast: nwtzlang::nwtz::Program = parser.produce_ast();
            //println!("{:#?}", ast);
            
            let _result = evaluate_runtime(ast, &mut env);
        }


        let tokens = tokenize(input.to_string());
        let mut parser = Parser::new(tokens);
        let program = parser.produce_ast().merge_imports();

        let _res = evaluate_runtime(program, &mut env);
        //println!("{:#?}", res);
    }

}
*/
//Modify eval object expr to handle tables
//and to reassign value by index

//User Defined Functions & Closures - Programming Language From Scratch  27:16