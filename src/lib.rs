pub mod types;
pub mod parser;
pub mod evaluator;
pub mod ast;
pub mod environment;
pub mod runtime;
pub mod lexer;

pub mod thread;

use crate::types::{ArrayVal, BooleanVal, FunctionVal, IntegerVal, NullVal, ObjectVal, ValueType};
use crate::types::ValueType::{Array, Boolean, Integer, Null, Object};
use crate::types::FunctionCall;
use crate::types::NativeFnValue;
use crate::types::ValueType::NativeFn;
use crate::types::ValueType::Function;

use std::collections::{HashMap};
use std::{env, fs, io};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use crate::ast::{NodeType, StringVal};
use crate::environment::Environment;
use crate::evaluator::evaluate;
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::runtime::RuntimeVal;
use crate::thread::ThreadManager;
//use crate::NodeType::Identifier;
//use crate::Token::{LBrace, RBrace, Semicolon};
//use crate::ValueType::{Boolean, NativeFn, Null, Integer, Object, Function, Array};


pub fn mk_number<T: Into<f64>>(number: T) -> Box<IntegerVal> {
    Box::from(IntegerVal {
        r#type: Option::from(Integer),
        value: number.into(),
    })
}


pub fn mk_null() -> Box<NullVal> {
    Box::from(NullVal{ r#type: Option::from(Null) })
}

pub fn mk_bool(b: bool) -> Box<BooleanVal> {
    Box::from(BooleanVal{ r#type: Option::from(Boolean), value: b })
}

pub fn mk_fn(call: FunctionCall) -> Box<NativeFnValue> {
    Box::from(NativeFnValue{
        r#type: Option::from(NativeFn),
        call
    })
}

pub fn mk_string(value: String) -> Box<StringVal> {
    Box::from(StringVal{
        r#type: Option::from(ValueType::String),
        kind: NodeType::StringLiteral,
        value,
    })
}

pub fn mk_array(elements: Vec<Box<dyn RuntimeVal + Send + Sync>>) -> Box<ArrayVal> {
    Box::from(ArrayVal {
        r#type: Option::from(Array),
        elements: Arc::new(Mutex::new(elements)),
    })
}

pub fn mk_object(properties: HashMap<String, Box<dyn RuntimeVal + Send + Sync>>) -> Box<ObjectVal> {
    Box::from(ObjectVal {
        r#type: Option::from(Object),
        properties: Arc::new(Mutex::new(properties)),
    })
}

pub fn native_fs_read(args: Vec<Box<dyn RuntimeVal + Send + Sync>>, _env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let path = args.into_iter().next()
        .and_then(|v| v.as_any().downcast_ref::<StringVal>().map(|s| s.value.clone()))
        .expect("_fs_read attend un string");
    let content = fs::read_to_string(&path)
        .expect("Erreur lecture fichier");
    Box::new(StringVal {
        r#type: Some(ValueType::String),
        kind: NodeType::StringLiteral,
        value: content,
    })
}



macro_rules! register_natives {
    ( $($name:expr => $func:path),* $(,)? ) => {
        pub fn native_registry() -> HashMap<&'static str, FunctionCall> {
            let mut m = HashMap::new();
            $(
                m.insert($name, Arc::new($func) as FunctionCall);
            )*
            m
        }
    }
}

register_natives!(
    "_fs_read" => native_fs_read,
);


pub fn interpreter_to_vec_string(env: &mut Environment, input: String) -> Vec<String> {
    let output = Arc::new(Mutex::new(Vec::<String>::new()));
    let output_for_native = output.clone();

    env.set_var(
        "log".to_string(),
        mk_fn(Arc::new(move |args, _| {
            let mut guard = output_for_native.lock().unwrap();
            for arg in args {
                guard.push(match_arg_to_string(&*arg));
            }
            mk_null()
        })),
        Option::from(NativeFn)
    );

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    let _ = evaluate(Box::new(ast), env);
    output.lock().unwrap().clone()
}

pub fn interpreter_to_stream(env: &mut Environment, input: String, ) -> UnboundedReceiver<String> {
    let (tx, rx): (UnboundedSender<String>, UnboundedReceiver<String>) = unbounded_channel();

    let tx_for_native = tx.clone();
    env.set_var(
        "log".to_string(),
        mk_fn(Arc::new(move |args, _| {
            for arg in args {
                let s = match_arg_to_string(&*arg);
                let _ = tx_for_native.send(s);
            }
            mk_null()
        })),
        Some(NativeFn),
    );

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();

    let mut env_for_task = env.clone();
    tokio::spawn(async move {
        let _ = evaluate(Box::new(ast), &mut env_for_task);
        drop(tx);
    });

    rx
}

pub fn match_arg_to_string(arg: &dyn RuntimeVal) -> String {
    if let Some(sv) = arg.as_any().downcast_ref::<StringVal>() {
        sv.value.clone()
    } else if let Some(iv) = arg.as_any().downcast_ref::<IntegerVal>() {
        iv.value.to_string()
    } else if let Some(bv) = arg.as_any().downcast_ref::<BooleanVal>() {
        bv.value.to_string()
    } else if let Some(av) = arg.as_any().downcast_ref::<ArrayVal>() {
        av.elements
            .lock()
            .unwrap()
            .iter()
            .map(|e| match_arg_to_string(e.as_ref()))
            .collect::<Vec<_>>()
            .join(",")
        
    } else if let Some(fu) = arg.as_any().downcast_ref::<FunctionVal>() {
        format!("{:?}", fu.body)
    } else if let Some(ob) = arg.as_any().downcast_ref::<ObjectVal>() {
        ob.properties
            .lock().unwrap()
            .iter()
            .map(|(key, val)| {
                let v = match_arg_to_string(val.as_ref());
                format!("{}: {{ {} }}", key, v)
            })
            .collect::<Vec<_>>()
            .join(", ")
    } else if arg.as_any().downcast_ref::<NullVal>().is_some() {
        "null".into()
    } else {
        format!("{:?}", arg)
    }
}

pub fn drive_stream(mut rx: UnboundedReceiver<String>) {
    std::thread::spawn(move || {
        while let Some(msg) = rx.blocking_recv() {
            println!("{}", msg);
        }
    });
}

pub fn  call_nwtz(name: &str, args: Option<Vec<String>>, scope: &mut Environment, ) -> Option<Box<dyn RuntimeVal + Send + Sync>> {

    let arg_vals: Vec<Box<dyn RuntimeVal + Send + Sync>> = args
        .unwrap_or_default()
        .into_iter()
        .map(|s| {
            let boxed: Box<dyn RuntimeVal + Send + Sync> = mk_string(s);
            boxed
        })
        .collect();

    let v = scope.lookup_var(name.to_string());

    match v.value_type().unwrap() {
        NativeFn => {
            let native = v
                .as_any()
                .downcast_ref::<NativeFnValue>()
                .expect("Expected a NativeFnValue");
            let res = (native.call)(arg_vals, scope);
            Some(res)
        }
        Function => {
            let f = v
                .as_any()
                .downcast_ref::<FunctionVal>()
                .expect("Expected a FunctionVal");

            let decl_env = f.declaration_env.lock().unwrap();
            let mut local_scope =
                Environment::new(Some(Box::new(decl_env.clone())));

            if arg_vals.len() != f.parameters.len() {
                panic!(
                    "Function `{}` expected {} args but got {}",
                    f.name,
                    f.parameters.len(),
                    arg_vals.len()
                );
            }

            for (param, arg_val) in f.parameters.iter().zip(arg_vals.into_iter()) {
                local_scope.set_var(param.clone(), arg_val, None);
            }

            let mut result: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in f.body.iter() {
                result = evaluate(stmt.clone(), &mut local_scope);
            }

            Some(result)
        }
        _ => None,
    }
}



pub fn make_global_env() -> Environment {
    let mut env = Environment::new(None);
    let thread_manager = ThreadManager::new();

    env.set_var("null".to_string(), mk_null(), Option::from(Null));
    env.set_var("true".to_string(), mk_bool(true), Option::from(Boolean));
    env.set_var("false".to_string(), mk_bool(false), Option::from(Boolean));

    env.set_var(
        "time".to_string(),
        mk_fn(Arc::new(|_args, _| {
            mk_number(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as f64 / 1000.0)
        })),
        Option::from(NativeFn),
    );

    env.set_var(
        "log".to_string(),
        mk_fn(Arc::new(move |args, _| {
            //let mut _guard = output_for_native.lock().unwrap();
            let mut out = String::new();
            for arg in args {
                out.push_str(&match_arg_to_string(&*arg));
            }
            println!("{}", out);
            mk_null()
        })),
        Option::from(NativeFn)
    );

    env.set_var(
        "sleep".to_string(),
        mk_fn(Arc::new(move |args, _| {
            let secs = args.first()
                .expect("sleep: un argument attendu")
                .as_any()
                .downcast_ref::<IntegerVal>()
                .expect("sleep: l'argument doit être un nombre")
                .value;

            std::thread::sleep(Duration::from_secs_f64(secs));
            mk_null()
        })),
        Option::from(NativeFn)
    );

    env.set_var(
        "input".to_string(),
        mk_fn(Arc::new(move |_args, _| {
            let mut out = String::new();
            io::stdin()
                .read_line(&mut out)
                .expect("failed to readline");
            out = out.trim_end().to_string();
            
            //println!("EXTRAIT : {}", out);

            mk_string(out)
        })),
        Option::from(NativeFn)
    );

    env.set_var(
        "thread".to_string(),
        mk_object({
            let mut props: HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();
            let thread_manager_clone = thread_manager.clone();

            props.insert(
                "start".to_string(),
                mk_fn(Arc::new(move |args, _scope| {

                    if args.is_empty() {
                        panic!("thread.start: fonction attendue comme argument");
                    }

                    let func_arg = &args[0];

                    match func_arg.value_type() {
                        Some(Function) => {
                            let func = func_arg
                                .as_any()
                                .downcast_ref::<FunctionVal>()
                                .expect("Expected FunctionVal");

                            let func_name = func.name.clone();
                            let func_body = func.body.clone();
                            let func_env = func.declaration_env.lock().unwrap().clone();

                            let handle = std::thread::spawn(move || {
                                let mut local_env = Environment::new(Some(Box::new(func_env)));

                                for stmt in func_body.iter() {
                                    let _ = evaluate(stmt.clone(), &mut local_env);
                                }
                            });

                            thread_manager.handles.lock().unwrap().push(handle);

                            mk_string(format!("{}", func_name))
                        }
                        _ => {
                            panic!("thread.start: argument doit être une fonction");
                        }
                    }
                })),
            );

            props.insert(
                "wait".to_string(),
                mk_fn(Arc::new({
                    move |_args, _scope| {
                        thread_manager_clone.wait_all();
                        mk_null()
                    }
                })),
            );
            
            props
            },
        ),
        Option::from(Object)
    );

    env.set_var(
        "system".to_string(),
        mk_object({
            let mut props: HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();

            props.insert(
                "a".to_string(),
                mk_string("A A A A A".to_string()),
            );

            props.insert(
                "config".to_string(),
                {

                    // pas la meilleure implementation, il faudrait sysinfo sauf que lib trop grandes

                    let mut a:Vec<Box<dyn RuntimeVal + Send + Sync>> = Vec::new();

                    for (key, value) in env::vars() {
                        a.push(mk_string(format!("{}: {:#?}", key, value).to_string()));
                    }

                    mk_array(a)
                }
            );

            props.insert(
                "type".to_string(),
                mk_fn(Arc::new(|args, _scope| {
                    mk_string(format!("{:?}", args[0].clone().value_type().unwrap()))
                })),
            );

            props.insert(
                "socket".to_string(),
                mk_object({
                    let mut props: HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();

                    props.insert("start".to_string(), mk_fn(Arc::new(move |args, scope| {
                        let addr = args.first()
                            .and_then(|arg| {
                                arg.as_any()
                                    .downcast_ref::<StringVal>()
                                    .map(|s| s.value.clone())
                            })
                            .unwrap_or_else(|| "127.0.0.1:8080".to_string());

                        let message = args
                            .get(1)
                            .and_then(|arg| {
                                arg.as_any()
                                    .downcast_ref::<StringVal>()
                                    .map(|s| s.value.clone())
                            })
                            .unwrap_or_else(|| "Hello from nwtz server!\n".to_string());

                        let listener = std::net::TcpListener::bind(&addr)
                            .expect("Impossible de binder l'adresse");

                        call_nwtz("log", Some(Vec::from([format!("Serveur démarré sur {}, en attente d'un client...", addr).to_string()])), scope);

                        let mut out:HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();

                        let (mut stream, peer_addr) = listener
                            .accept()
                            .expect("Échec de l'acceptation d'un client");

                        call_nwtz("log", Some(Vec::from([format!("Client connecté depuis : {}", peer_addr).to_string()])), scope);

                        let mut buffer = [0u8; 512];
                        let n = stream
                            .read(&mut buffer)
                            .expect("Échec de lecture sur le socket");

                        call_nwtz("log", Some(Vec::from([format!("Reçu ({} octets) : {}", n, String::from_utf8_lossy(&buffer[..n])).to_string()])), scope);

                        stream.write_all(message.as_bytes()).expect("Échec d'envoi de la réponse");
                        call_nwtz("log", Some(Vec::from(["Réponse envoyée, fermeture de la connexion.".to_string()])), scope);
                        //mk_null()

                        out.insert("output".to_string(), mk_string(String::from_utf8_lossy(&buffer[..n]).to_string()));

                        mk_object(out)
                    })));

                    props
                })
            );
            props
            },
        ),
        Option::from(Object)
    );

    let registry = native_registry();
    for (name, func) in registry {
        env.set_var(
            name.to_string(),
            mk_fn(func.clone()),
            Option::from(NativeFn)
        );
    }
    env
}



#[cfg(test)]
mod tests {

    //use crate::{evaluate, make_global_env, tokenize, Parser};
    #[warn(dead_code)]
    fn test() {
        println!("test");
    }
    
}

/*
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

                let _result = evaluate(Box::new(ast), &mut env);
            }


            let tokens = tokenize(input.to_string());
            let mut parser = Parser::new(tokens);
            let program = parser.produce_ast().merge_imports();

            let _res = evaluate(Box::new(program), &mut env);
            //println!("{:#?}", res);
        }
 */


