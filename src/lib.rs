use std::sync::{Arc, Mutex};
use crate::nwtz::{evaluate, make_global_env, mk_native_fn, mk_null, tokenize, ArrayVal, BooleanVal, Environment, NullVal, NumberVal, Parser, Program, RuntimeVal, StringVal, Token};
pub mod nwtz;

pub fn interpreter_to_vec_string(mut env: Environment,input: String) -> Vec<String> {
    let output = Arc::new(Mutex::new(Vec::<String>::new()));
    let output_for_native = output.clone();

    env.declare_var(
        "log".to_string(),
        mk_native_fn(Arc::new(move |args, _| {
            let mut guard = output_for_native.lock().unwrap();
            for arg in args {
                if let Some(s) = arg.as_any().downcast_ref::<StringVal>() {
                    guard.push(s.value.clone());
                } else if let Some(n) = arg.as_any().downcast_ref::<NumberVal>() {
                    guard.push(n.value.to_string());
                } else if let Some(b) = arg.as_any().downcast_ref::<BooleanVal>() {
                    guard.push(b.value.to_string());
                }else if let Some(array_val) = arg.as_any().downcast_ref::<ArrayVal>() {
                    let mut out = std::string::String::new();
                    for element in array_val.elements.borrow().iter() {
                        let s = if let Some(string_val) = element.as_any().downcast_ref::<StringVal>() {
                            string_val.value.clone()
                        } else if let Some(num_val) = element.as_any().downcast_ref::<NumberVal>() {
                            num_val.value.to_string()
                        } else if let Some(bool_val) = element.as_any().downcast_ref::<BooleanVal>() {
                            bool_val.value.to_string()
                        } else if let Some(_array_val) = element.as_any().downcast_ref::<ArrayVal>() {
                            "ARRAY INSIDE ARRAY NOT IMPLEMENTED YET".to_string()
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
                    guard.push(out);
                } else if arg.as_any().downcast_ref::<NullVal>().is_some() {
                    guard.push("null".to_string());
                } else {
                    guard.push(format!("{:?}", arg));
                }
            }
            mk_null()
        })),
    );

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    let _ = evaluate(Box::new(ast), &mut env);
    output.lock().unwrap().clone()
}

pub fn tokenize_to_vec_token(input: String) -> Vec<Token>{
    tokenize(input)
}

pub fn parse_tokens_to_parser(tokens: Vec<Token>) -> Parser{
    Parser::new(tokens)
}

pub fn produce_ast_from_parser(mut parser: Parser) -> Program{
    parser.produce_ast()
}

pub fn evaluate_runtime(ast: Program, env: &mut Environment) ->  Box<dyn RuntimeVal>{
    evaluate(Box::new(ast), env)
}

/*
pub fn interpreter_with_import_to_vec_string(input: String, ) -> Vec<String> {
    
}
 */