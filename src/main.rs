use std::fs;
use crate::nwtz::Nwtz;

mod nwtz;


fn main() {
    
    let input = fs::read_to_string("main.nwtz").unwrap();
    Nwtz::grammar(Nwtz::new(input).tokenize());
}
