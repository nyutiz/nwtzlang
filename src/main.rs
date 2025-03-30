#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_imports)]


use std::{fs, io};
use std::io::Write;
use std::sync::Arc;
use std::time::SystemTime;
use crate::nwtz::{evaluate, mk_bool, mk_native_fn, mk_null, mk_number, tokenize, BooleanVal, Environment, FunctionCall, NullVal, NumberVal, Parser, RuntimeVal, StringLiteralExpr, ValueType};

mod nwtz;

fn main() {
    let mut env = Environment::new(None);

    // Optionally, declare built-in variables here:
    //env.declare_var("x".to_string(), mk_number(100));
    //env.declare_var("null".to_string(), mk_null());
    //env.declare_var("true".to_string(), mk_bool(true));
    //env.declare_var("false".to_string(), mk_bool(false));

    env.declare_var("null".to_string(), mk_null());
    env.declare_var("true".to_string(), mk_bool(true));
    env.declare_var("false".to_string(), mk_bool(false));

    env.declare_var(
        "log".to_string(),
        mk_native_fn(Arc::new(|args, _env| {
            for arg in args {
                if let Some(string_val) = arg.as_any().downcast_ref::<StringLiteralExpr>() {
                    println!("{}", string_val.value);
                }
                else if let Some(number_val) = arg.as_any().downcast_ref::<NumberVal>() {
                    println!("{}", number_val.value);
                }
                else if let Some(bool_val) = arg.as_any().downcast_ref::<BooleanVal>() {
                    println!("{}", bool_val.value);
                }
                else if let Some(bool_val) = arg.as_any().downcast_ref::<NullVal>() {
                    println!("null");
                }
                else {
                    println!("{:#?}", arg);
                }
            }
            mk_null()
        })),
    );


    env.declare_var(
        "time".to_string(),
        mk_native_fn(Arc::new(|args: Vec<Box<dyn RuntimeVal>>, _scope: &mut Environment| {
            mk_number(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as f64 / 1000.0)
        })),
    );

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
            println!("{:#?}", ast);


            //println!("{:#?}", ast);
            
            let result = evaluate(Box::new(ast), &mut env);
        }


        let tokens = tokenize(input.to_string());
        let mut parser = Parser::new(tokens);
        let program = parser.produce_ast();

        let res = evaluate(Box::new(program), &mut env);
        //println!("{:#?}", res);
    }

}


//User Defined Functions & Closures - Programming Language From Scratch  27:16