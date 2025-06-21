use std::{fs, io};
use std::io::Write;
use std::sync::Arc;
use nwtzlang::{evaluate_runtime, make_global_env};
use crate::nwtz::{evaluate, mk_native_fn, mk_null, tokenize, ArrayVal, BooleanVal, NullVal, NumberVal, Parser, StringVal};

mod nwtz;

fn main() {
    let mut env = make_global_env();

    env.declare_var(
        "log".to_string(),
        mk_native_fn(Arc::new(|args, _env| {
            for arg in args {
                if let Some(string_val) = arg.as_any().downcast_ref::<StringVal>() {
                    println!("{}", string_val.value);
                }
                else if let Some(number_val) = arg.as_any().downcast_ref::<NumberVal>() {
                    println!("{}", number_val.value);
                }
                else if let Some(bool_val) = arg.as_any().downcast_ref::<BooleanVal>() {
                    println!("{}", bool_val.value);
                }
                else if let Some(array_val) = arg.as_any().downcast_ref::<ArrayVal>() {
                    let mut out = String::new();
                    for element in array_val.elements.borrow().iter() {
                        let s = if let Some(string_val) = element.as_any().downcast_ref::<StringVal>() {
                            string_val.value.clone()
                        } else if let Some(num_val) = element.as_any().downcast_ref::<NumberVal>() {
                            num_val.value.to_string()
                        } else if let Some(bool_val) = element.as_any().downcast_ref::<BooleanVal>() {
                            bool_val.value.to_string()
                        } else if let Some(_array_val) = element.as_any().downcast_ref::<ArrayVal>() {
                            //let mut joined = String::new();
                            //for child in array_val.elements.borrow().iter() {
                            //    if !joined.is_empty() { joined.push(','); }
                            //    joined.push_str(&format_val(child));
                            //}
                            //format!("[{}]", joined)
                            "ARRAY INSIDE ARRAY NOT IMPLEMENTED".to_string()
                        } else if let Some(_) = element.as_any().downcast_ref::<NullVal>() {
                            "null".to_string()
                        } else {
                            String::new()
                        };

                        if !out.is_empty() {
                            out.push(',');
                        }
                        out.push_str(&s);
                    }
                    println!("{}", out);
                }
                else if let Some(_null_val) = arg.as_any().downcast_ref::<NullVal>() {
                    println!("null");
                }
                else {
                    println!("{:#?}", arg);
                }
            }
            mk_null()
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
            //println!("{:#?}", ast);
            
            let _result = evaluate_runtime(Box::new(ast), &mut env);
        }


        let tokens = tokenize(input.to_string());
        let mut parser = Parser::new(tokens);
        let program = parser.produce_ast().merge_imports();

        let _res = evaluate_runtime(Box::new(program), &mut env);
        //println!("{:#?}", res);
    }

}

//Modify eval object expr to handle tables
//and to reassign value by index

//User Defined Functions & Closures - Programming Language From Scratch  27:16