use nwtzlang::evaluator::evaluate;
use nwtzlang::lexer::tokenize;
use nwtzlang::make_global_env;
use nwtzlang::parser::Parser;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;
use nwtzlang::environment::Environment;
use nwtzlang::types::ValueType;

#[tokio::main]
async fn main() {
    let mut env = make_global_env();
    let mut input = String::new();
    let mut print_ast = false;
    let mut show_version = false;

    let args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        run_repl(&mut env).await;
        return;
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage_and_exit(0);
            }
            "-v" | "--version" => {
                show_version = true;
                i += 1;
            }
            "-e" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: no code provided after '-e'");
                    process::exit(1);
                }
                input = args[i].clone();
                i += 1;
            }
            "-r" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: no file path provided after '-f'");
                    process::exit(1);
                }
                input = fs::read_to_string(&args[i]).unwrap_or_else(|err| {
                    eprintln!("Unable to read file {}: {}", args[i], err);
                    process::exit(1);
                });
                i += 1;
            }
            "--print-ast" => {
                print_ast = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                print_usage_and_exit(1);
            }
        }
    }

    if show_version {
        println!("nwtzlang v{}", env!("CARGO_PKG_VERSION"));
        process::exit(0);
    }

    if input.is_empty() {
        eprintln!("No input provided.");
        print_usage_and_exit(1);
    }

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    if print_ast {
        println!("{:#?}\n", ast);
    }

    let result = evaluate(Box::new(ast), &mut env);

    if !matches!(result.value_type(), Some(ValueType::Null)) {
        println!("{:#?}", result);
    }
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage: nwtzlang [OPTIONS]\n\n\
         Options:\n\
           -h, --help           Display this help\n\
           -v, --version        Show version\n\
           -e <code>            Evaluate the string <code>\n\
           -r <file>            Execute the code in <file>\n\
           --print-ast          Print the AST before evaluation\n\
           (no option)          Launch the interactive REPL\n"
    );
    process::exit(code);
}

async fn run_repl(env: &mut Environment) {
    let stdin = io::stdin();
    loop {
        print!("» ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }

        let tokens = tokenize(line.clone());
        let mut parser = Parser::new(tokens);
        let ast = parser.produce_ast();

        let result = evaluate(Box::new(ast), env);
        if !matches!(result.value_type(), Some(ValueType::Null)) {
            println!("{:#?}", result);
        }
    }
}
