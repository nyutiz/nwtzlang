use nwtzlang::evaluator::evaluate;
use nwtzlang::lexer::tokenize;
use nwtzlang::make_global_env;
use nwtzlang::parser::Parser;

#[tokio::main]
async fn main() {
    let mut env = make_global_env();

    let input = r#"
thread.start(fn start() {
    log("Hello from thread!");
    sleep(5);
    log("Thread working...");
});

s = thread.start(fn start() {
    for(i = 0; i < 100; i = i + 1;) {
        log(i);
        sleep(0.05);
        //log(time());
    }
});

log("Thread : ", s);

log("Main thread continues...");
thread.wait();
"#.to_string();
    

    let tokens = tokenize(input);
    let mut parser = Parser::new(tokens);
    let ast = parser.produce_ast();
    //println!("AST{:#?}\n\n", ast);
    let _ = evaluate(Box::new(ast), &mut env);
    //println!("EVALUATED {:#?}", output.lock().unwrap().clone())


}