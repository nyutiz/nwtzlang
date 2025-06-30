use nwtzlang::evaluator::evaluate;
use nwtzlang::lexer::tokenize;
use nwtzlang::make_global_env;
use nwtzlang::parser::Parser;

#[tokio::main]
async fn main() {
    let mut env = make_global_env();

    let input = r#"
thread.start(fn worker() {
    log("Hello from thread!");
    sleep(5);
    log("Thread working...");
});

log("Main thread continues...");
sleep(5);
"#.to_string();
    

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    //println!("AST{:#?}\n\n", ast);
    let _ = evaluate(Box::new(ast), &mut env);
    //println!("EVALUATED {:#?}", output.lock().unwrap().clone())


}