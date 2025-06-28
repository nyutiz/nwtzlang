use nwtzlang::evaluator::evaluate;
use nwtzlang::lexer::tokenize;
use nwtzlang::make_global_env;
use nwtzlang::parser::Parser;

#[tokio::main]
async fn main() {
    let mut env = make_global_env();

    let input = r#"

log("Enter input : ");

dzgefs = input();

if dzgefs == "a" {
    log("aA");
} else {
    log("AFAFA ! ", dzgefs);
}

s = system.socket("localhost:8080");


"#.to_string();
    

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    //println!("AST{:#?}\n\n", ast);
    let _ = evaluate(Box::new(ast), &mut env);
    //println!("EVALUATED {:#?}", output.lock().unwrap().clone())


}