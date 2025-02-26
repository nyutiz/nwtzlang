#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::HashMap;
use std::sync::Mutex;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::nwtz::IdentifierType::{Func, Generic, Pop, Text, Dis, End, Set, Push};
use crate::nwtz::PunctuationType::{FParenthesis, QuotationMark, SParenthesis};


pub(crate) const DEBUG: bool = false;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Identifier(IdentifierType),
    Number,
    Whitespace,
    Operator,
    Punctuation(PunctuationType),
    NewLine,
    EndOfFile,
    Function,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PunctuationType {
    FParenthesis, // First parentheses
    SParenthesis, // Second parentheses
    QuotationMark,
    Generic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IdentifierType {
    Generic,
    Stack,
    String,
    Text,
    End,
    Set,
    Push,
    Pop,
    Dis,
    Func,
}

#[derive(Debug, Clone)]
pub struct Token {
    token_type: TokenType,
    value: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    name: String,
    action: Vec<Vec<Token>>, // Liste de Liste De Token
}

impl Token {
    pub fn new(token_type: TokenType, value: String) -> Self {
        Token { token_type, value }
    }
}

pub struct Nwtz {
    input: String,
    regex: Vec<(Regex, Box<dyn Fn(&str) -> TokenType>)>,
}

pub struct Stack<T> {
    elements: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    pub fn push(&mut self, item: T) {
        self.elements.push(item);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.elements.pop()
    }

    // retourne le pointeur vers le dersnier element sans l'enlever
    pub fn peek(&self) -> Option<&T> {
        self.elements.last()
    }
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

static VARS: Lazy<Mutex<Vec<HashMap<String, String>>>> = Lazy::new(|| Mutex::new(Vec::new()));
static STACK: Lazy<Mutex<Stack<String>>> = Lazy::new(|| Mutex::new(Stack::new()));

static FUNCS: Lazy<Mutex<Vec<Function>>> = Lazy::new(|| Mutex::new(Vec::new()));

impl Nwtz {

    pub fn new(input: String) -> Self {
        let regex: Vec<(Regex, Box<dyn Fn(&str) -> TokenType>)> = vec![
            (
                Regex::new(r"^[ \t]+").unwrap(),
                Box::new(|_value: &str| TokenType::Whitespace) as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r#"^"([^"]*)""#).unwrap(),
                Box::new(|_value: &str| TokenType::Identifier(Text))
                    as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r"^%([^:\n]+):").unwrap(),
                Box::new(|_value: &str| TokenType::Function)
                    as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r"^[a-zA-Z][a-zA-Z0-9]*").unwrap(),
                Box::new(|value: &str| {
                    if value == "end%" {
                        TokenType::Identifier(End)
                    }
                    else if value == "set" {
                        TokenType::Identifier(Set)
                    }
                    else if value == "dis" {
                        TokenType::Identifier(Dis)
                    }
                    else if value == "stack" {
                        TokenType::Identifier(Dis)
                    }
                    else if value == "push" {
                        TokenType::Identifier(Push)
                    }
                    else if value == "pop" {
                        TokenType::Identifier(Pop)
                    }
                    else if value == "func" {
                        TokenType::Identifier(Func)
                    }
                    else if value == "end" {
                        TokenType::Identifier(End)
                    }
                    else if value.starts_with("$") {
                        TokenType::Identifier(IdentifierType::String)
                    }
                    else {
                        TokenType::Identifier(Generic)
                    }
                }) as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r"^\d+").unwrap(),
                Box::new(|_value: &str| TokenType::Number) as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r"^[\+\-\*/=<>!&|]+").unwrap(),
                Box::new(|_value: &str| TokenType::Operator) as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r#"^[.,;:!?(){}\[\]<>"']"#).unwrap(),
                Box::new(|value: &str| {
                    if value == "(" {
                        TokenType::Punctuation(FParenthesis)
                    }
                    else if value == ")" {
                        TokenType::Punctuation(SParenthesis)
                    }
                    else if value == "\"" {
                        TokenType::Punctuation(QuotationMark)
                    }
                    else {
                        TokenType::Punctuation(PunctuationType::Generic)
                    }
                }) as Box<dyn Fn(&str) -> TokenType>,
            ),
            (
                Regex::new(r"\n").unwrap(),
                Box::new(|_value: &str| TokenType::NewLine) as Box<dyn Fn(&str) -> TokenType>,
            ),
        ];
        Nwtz { input, regex }
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut remaining_input = self.input.as_str();

        while !remaining_input.is_empty() {
            let mut matched = false;

            for (regex, token_func) in &self.regex {
                if let Some(mat) = regex.find(remaining_input) {
                    let mut token_value = mat.as_str().to_string();
                    let token_type = token_func(token_value.as_str());

                    if let TokenType::Identifier(Text) = token_type {
                        token_value = token_value.trim_matches('"').to_string();
                    }

                    if token_type != TokenType::Whitespace {
                        tokens.push(Token::new(token_type, token_value.clone()));
                    }

                    remaining_input = &remaining_input[mat.end()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                let unknown_char = remaining_input.chars().next().unwrap().to_string();
                tokens.push(Token::new(TokenType::Unknown, unknown_char.clone()));
                remaining_input = &remaining_input[1..];
            }
        }

        tokens
    }


    pub fn grammar(tokens: Vec<Token>) {
        let mut active_line: Vec<Token> = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            let token = &tokens[i];

            if token.token_type != TokenType::NewLine {
                active_line.push(token.clone());
                i += 1;
            } else {
                if !active_line.is_empty() {
                    if active_line[0].token_type == TokenType::Function {
                        let mut name = active_line[0].value.clone();
                        name.pop();
                        if name.len() > 0 {
                            name.remove(0);
                        }
                        //println!("Function {}", name);

                        let mut function_body: Vec<Vec<Token>> = Vec::new();
                        let mut current_line: Vec<Token> = Vec::new();
                        let mut found_end = false;
                        i += 1;

                        while i < tokens.len() && !found_end {
                            let current_token = &tokens[i];

                            if current_token.token_type == TokenType::NewLine {
                                if !current_line.is_empty() {
                                    let has_end = current_line.iter().any(|t|
                                        t.token_type == TokenType::Identifier(End)
                                    );

                                    if has_end {
                                        found_end = true;
                                    } else {
                                        function_body.push(current_line.clone());
                                    }
                                    current_line.clear();
                                }
                            } else {
                                current_line.push(current_token.clone());
                            }

                            i += 1;
                        }

                        if !current_line.is_empty() {
                            let has_end = current_line.iter().any(|t|
                                t.token_type == TokenType::Identifier(IdentifierType::End)
                            );

                            if !has_end {
                                function_body.push(current_line);
                            }
                        }

                        //println!("Function body contains {} lines", function_body.len());

                        Self::add_func(name, function_body);
                        
                        active_line.clear();
                        continue;
                    }
                    else {
                        debug(format!("Active line {:?}", active_line));
                        Self::line_handler(active_line.clone());
                    }
                    active_line.clear();
                }
                i += 1;
            }
        }

        if !active_line.is_empty() {
            debug(format!("Active line {:?}", active_line));
            Self::line_handler(active_line.clone());
        }

        debug(format!("VARS {:?}", VARS.lock().unwrap()));
        debug(format!("FUNS {:?}", FUNCS.lock().unwrap()));

    }
    pub fn add_func(name: String, func: Vec<Vec<Token>>){
        FUNCS.lock().unwrap().push(Function {
            name,
            action: func,
        });
    }

    pub fn run_func(name: String){
        let guard = FUNCS.lock().unwrap();
        for scope in guard.iter().rev() {
            if scope.name == name {
                for action in scope.action.iter() {
                    Self::line_handler(action.clone());
                }
            }
        }
    }

    pub fn line_handler(line: Vec<Token>) {

        let args:Vec<Token> = Self::handle_args(line.clone());

        debug(format!("args {:?}", args));

        for (index, token) in args.iter().enumerate() {
            if token.token_type == TokenType::Identifier(Set) && args.len() == 3{
                let arg1 = args.get(index+1).unwrap().value.clone();
                let mut arg2 = args.get(index+2).unwrap().clone();
                if arg2.token_type == TokenType::Identifier(Pop){
                    arg2.value = STACK.lock().unwrap().pop().unwrap();
                }
                debug(format!("Arg: {}  Arg: {}", arg1, arg2.value));
                Self::define_var(arg1, arg2.value);
            }
            else if token.token_type == TokenType::Identifier(Dis) && args.len() == 2 {
                let mut arg1 = args.get(index+1).unwrap().clone();
                if arg1.value == "" || arg1.value.ends_with("\\n") {
                    arg1.value.pop();
                    arg1.value.pop();
                    println!("{}", arg1.value);
                }else{
                    if arg1.token_type == TokenType::Identifier(Pop){
                        print!("{}", STACK.lock().unwrap().pop().unwrap())
                    } else if arg1.token_type == TokenType::Identifier(Generic) {
                        print!("{}", Self::read_var(arg1.value.clone()));
                    } else if arg1.token_type == TokenType::Identifier(Text){
                        print!("{}", arg1.value)
                    }
                }

            }
            else if token.token_type == TokenType::Identifier(Push) && args.len() == 2 {
                let arg1 = args.get(index+1).unwrap().clone();

                if arg1.token_type == TokenType::Identifier(Generic) {
                    STACK.lock().unwrap().push(Self::read_var(arg1.value));
                } else {
                    STACK.lock().unwrap().push(arg1.value);
                }
            }
            else if token.token_type == TokenType::Identifier(Pop) && args.len() == 2 {
                let argm1 = args.get(index-1).unwrap().value.clone();
                if argm1 != "dis" {
                    let arg1 = args.get(index+1).unwrap().value.clone();
                    Self::define_var(arg1, STACK.lock().unwrap().pop().unwrap())
                }
            }
            else if token.token_type == TokenType::Identifier(Func) && args.len() == 2 {
                let arg1 = args.get(index+1).unwrap().value.clone();
                Self::run_func(arg1);
            }
        }
    }

    pub fn handle_args(args: Vec<Token>) -> Vec<Token> {
        let mut results: Vec<Token> = Vec::new();
        debug(format!("Args with 0 : {:?}", args));

        let mut i = 0;
        while i < args.len() {
            let token = &args[i];

            if token.token_type == TokenType::Punctuation(FParenthesis) {
                let mut depth = 1;
                let mut end_index = i + 1;

                while depth > 0 && end_index < args.len() {
                    if args[end_index].token_type == TokenType::Punctuation(FParenthesis) {
                        depth += 1;
                    } else if args[end_index].token_type == TokenType::Punctuation(SParenthesis) {
                        depth -= 1;
                    }
                    if depth > 0 {
                        end_index += 1;
                    }
                }

                if depth == 0 {
                    let expr = args[i+1..end_index].to_vec();

                    let mut nbr: Vec<i32> = Vec::new();
                    let mut op: Vec<String> = Vec::new();

                    for arg in &expr {
                        if arg.token_type == TokenType::Number {
                            nbr.push(arg.value.parse::<i32>().unwrap());
                        }
                        else if arg.token_type == TokenType::Identifier(Generic) {
                            nbr.push(Self::read_var(arg.value.clone()).parse::<i32>().unwrap());
                        }
                        else if arg.token_type == TokenType::Operator {
                            op.push(arg.value.clone());
                        }
                    }

                    let mut j = 0;
                    while j < op.len() {
                        if op[j] == "*" {
                            let result = nbr[j] * nbr[j + 1];
                            nbr[j] = result;
                            nbr.remove(j + 1);
                            op.remove(j);
                        }
                        else if op[j] == "/" {
                            let result = nbr[j] / nbr[j + 1];
                            nbr[j] = result;
                            nbr.remove(j + 1);
                            op.remove(j);
                        }
                        else {
                            j += 1;
                        }
                    }

                    let mut result = nbr[0];
                    for (j, operator) in op.iter().enumerate() {
                        match operator.as_str() {
                            "+" => result += nbr[j + 1],
                            "-" => result -= nbr[j + 1],
                            _   => (),
                        }
                    }

                    results.push(Token::new(TokenType::Number, result.to_string()));

                    i = end_index + 1;
                } else {
                    results.push(token.clone());
                    i += 1;
                }
            } else {
                results.push(token.clone());
                i += 1;
            }
        }

        debug(format!("Result : {:?}", results));
        results
    }
    pub fn define_var(name: String, value: String) {
        let mut vars = VARS.lock().unwrap();
        if let Some(current_scope) = vars.last_mut() {
            current_scope.insert(name, value);
        } else {
            let mut new_scope = HashMap::new();
            new_scope.insert(name, value);
            vars.push(new_scope);
        }
    }
    pub fn read_var(name: String) -> String {
        let guard = VARS.lock().unwrap();
        for scope in guard.iter().rev() {
            if let Some(value) = scope.get(&name) {
                return value.clone();
            }
        }
        "Null".to_string()
    }

}

pub fn debug(text: String) {
    if DEBUG {
        println!("{}", text);
    }
}
