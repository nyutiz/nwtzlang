use std::fs;
use crate::nwtz::Nwtz;

mod nwtz;


fn main() {
    let input = fs::read_to_string("main.nwtz");
    
    let mut tokenizer = Nwtz::new(input.unwrap());
    let tokens = tokenizer.tokenize();

    //for token in tokens.iter() {
    //    println!("{:?}", token);
    //}

    Nwtz::grammar(tokens);

}

// define a variable : set x 2
// stack aussi : push x || push 2
// stack : set x pop || pop (supp la derniere variable)