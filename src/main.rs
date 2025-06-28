use nwtzlang::evaluator::evaluate;
use nwtzlang::lexer::tokenize;
use nwtzlang::make_global_env;
use nwtzlang::parser::Parser;

#[tokio::main]
async fn main() {
    let mut env = make_global_env();
    //let output = Arc::new(Mutex::new(Vec::<String>::new()));
    //let output_for_native = output.clone();

    //let input = fs::read_to_string("code.nwtz").unwrap();

    let input = r#"

dzgefs = input();

if dzgefs == "a" {
    log("aA");
} else {
    log("AFAFA ! ", dzgefs);
}

s = system.socket("localhost:8080");


"#.to_string();

    // remplacer StringLitteral par Identifier // deja identifier donc pas ça le pb  StringVal { type: None, kind: StringLiteral, value: "\"a\""}
    // Peut etre stringlitteral .value

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    //println!("AST{:#?}\n\n", ast);
    let _ = evaluate(Box::new(ast), &mut env);
    //println!("EVALUATED {:#?}", output.lock().unwrap().clone())


}