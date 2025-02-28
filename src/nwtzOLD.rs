#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::io;
use std::io::Write;
use std::sync::Mutex;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::nwtzOLD::IdentifierTypeOld::{Func, Generic, Pop, Text, Dis, End, Set, Push, IfEqual, Input};
use crate::nwtzOLD::PunctuationTypeOLD::{FParenthesis, QuotationMark, SParenthesis};

pub(crate) const DEBUG: bool = false;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenTypeOld {
    Identifier(IdentifierTypeOld),
    Number,
    Whitespace,
    Operator,
    Punctuation(PunctuationTypeOLD),
    NewLine,
    EndOfFile,
    Function,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PunctuationTypeOLD {
    FParenthesis, // First parentheses
    SParenthesis, // Second parentheses
    QuotationMark,
    Generic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IdentifierTypeOld {
    Generic,
    Input,
    IfEqual,
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
pub struct TokenOLD {
    token_type: TokenTypeOld,
    value: String,
}

#[derive(Debug, Clone)]
pub struct FunctionOLd {
    name: String,
    action: Vec<Vec<TokenOLD>>,
}

#[derive(Debug, Clone)]
pub struct VariableOLD {
    name: String,
    value: String,
}


impl TokenOLD {
    pub fn new(token_type: TokenTypeOld, value: String) -> Self {
        TokenOLD { token_type, value }
    }
}

pub struct NwtzOLD {
    input: String,
    regex: Vec<(Regex, Box<dyn Fn(&str) -> TokenTypeOld>)>,
}

pub struct StackOLD<T> {
    elements: Vec<T>,
}

impl<T> StackOLD<T> {
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

static VARSOLD: Lazy<Mutex<Vec<VariableOLD>>> = Lazy::new(|| Mutex::new(Vec::new()));
static STACKOLD: Lazy<Mutex<StackOLD<String>>> = Lazy::new(|| Mutex::new(StackOLD::new()));

static FUNCSOLD: Lazy<Mutex<Vec<FunctionOLd>>> = Lazy::new(|| Mutex::new(Vec::new()));

impl NwtzOLD {

    pub fn new(input: String) -> Self {
        let regex: Vec<(Regex, Box<dyn Fn(&str) -> TokenTypeOld>)> = vec![
            (
                Regex::new(r"^[ \t]+").unwrap(),
                Box::new(|_value: &str| TokenTypeOld::Whitespace) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r#"^"([^"]*)""#).unwrap(),
                Box::new(|_value: &str| TokenTypeOld::Identifier(Text))
                    as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r"^%([^:\n]+):").unwrap(),
                Box::new(|_value: &str| TokenTypeOld::Function)
                    as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r"^[a-zA-Z][a-zA-Z0-9]*").unwrap(),
                Box::new(|value: &str| {
                    if value == "end%" {
                        TokenTypeOld::Identifier(End)
                    }
                    else if value == "set" {
                        TokenTypeOld::Identifier(Set)
                    }
                    else if value == "dis" {
                        TokenTypeOld::Identifier(Dis)
                    }
                    else if value == "stack" {
                        TokenTypeOld::Identifier(Dis)
                    }
                    else if value == "push" {
                        TokenTypeOld::Identifier(Push)
                    }
                    else if value == "pop" {
                        TokenTypeOld::Identifier(Pop)
                    }
                    else if value == "func" {
                        TokenTypeOld::Identifier(Func)
                    }
                    else if value == "end" {
                        TokenTypeOld::Identifier(End)
                    }
                    else if value == "ife" {
                        TokenTypeOld::Identifier(IfEqual)
                    }
                    else if value == "input" {
                        TokenTypeOld::Identifier(Input)
                    }
                    else if value.starts_with("$") {
                        TokenTypeOld::Identifier(IdentifierTypeOld::String)
                    }
                    else {
                        TokenTypeOld::Identifier(Generic)
                    }
                }) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r"^\d+").unwrap(),
                Box::new(|_value: &str| TokenTypeOld::Number) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r"^[\+\-\*/=<>!&|]+").unwrap(),
                Box::new(|_value: &str| TokenTypeOld::Operator) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r#"^[.,;:!?(){}\[\]<>"']"#).unwrap(),
                Box::new(|value: &str| {
                    if value == "(" {
                        TokenTypeOld::Punctuation(FParenthesis)
                    }
                    else if value == ")" {
                        TokenTypeOld::Punctuation(SParenthesis)
                    }
                    else if value == "\"" {
                        TokenTypeOld::Punctuation(QuotationMark)
                    }
                    else {
                        TokenTypeOld::Punctuation(PunctuationTypeOLD::Generic)
                    }
                }) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
            (
                Regex::new(r"\n").unwrap(),
                Box::new(|_value: &str| TokenTypeOld::NewLine) as Box<dyn Fn(&str) -> TokenTypeOld>,
            ),
        ];
        NwtzOLD { input, regex }
    }

    pub fn tokenize(&mut self) -> Vec<TokenOLD> {
        let mut tokens = Vec::new();
        let mut remaining_input = self.input.as_str();

        while !remaining_input.is_empty() {
            let mut matched = false;

            for (regex, token_func) in &self.regex {
                if let Some(mat) = regex.find(remaining_input) {
                    let mut token_value = mat.as_str().to_string();
                    let token_type = token_func(token_value.as_str());

                    if let TokenTypeOld::Identifier(Text) = token_type {
                        token_value = token_value.trim_matches('"').to_string();
                    }

                    if token_type != TokenTypeOld::Whitespace {
                        tokens.push(TokenOLD::new(token_type, token_value.clone()));
                    }

                    remaining_input = &remaining_input[mat.end()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                let unknown_char = remaining_input.chars().next().unwrap().to_string();
                tokens.push(TokenOLD::new(TokenTypeOld::Unknown, unknown_char.clone()));
                remaining_input = &remaining_input[1..];
            }
        }

        tokens
    }


    pub fn grammar(tokens: Vec<TokenOLD>) {
        let mut active_line: Vec<TokenOLD> = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            let token = &tokens[i];


            if token.token_type != TokenTypeOld::NewLine {
                active_line.push(token.clone());
                i += 1;
            } else {
                if !active_line.is_empty() {
                    if active_line[0].token_type == TokenTypeOld::Function {
                        let mut name = active_line[0].value.clone();
                        name.pop();
                        if !name.is_empty() {
                            name.remove(0);
                        }
                        let mut function_body: Vec<Vec<TokenOLD>> = Vec::new();
                        let mut current_line: Vec<TokenOLD> = Vec::new();
                        let mut found_end = false;
                        i += 1;
                        while i < tokens.len() && !found_end {
                            let current_token = &tokens[i];
                            if current_token.token_type == TokenTypeOld::NewLine {
                                if !current_line.is_empty() {
                                    let has_end = current_line.iter().any(|t|
                                        t.token_type == TokenTypeOld::Identifier(End)
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
                                t.token_type == TokenTypeOld::Identifier(End)
                            );
                            if !has_end {
                                function_body.push(current_line);
                            }
                        }
                        Self::add_func(name, function_body);
                    }
                    else if active_line[0].token_type == TokenTypeOld::Identifier(IfEqual) && active_line.len() == 3 {
                        let mut arg1 = active_line[1].clone();
                        let mut arg2 = active_line[2].clone();

                        if arg1.token_type == TokenTypeOld::Identifier(Generic) {
                            arg1.value = Self::read_var(active_line[1].clone().value);
                        }
                        else if  arg1.token_type == TokenTypeOld::Identifier(Text){
                            arg1.value = arg1.value.trim_matches('"').to_string();
                        } 
                        else if arg1.token_type == TokenTypeOld::Identifier(Pop){
                            arg1.value = STACKOLD.lock().unwrap().pop().unwrap();
                        }
                        

                        if arg2.token_type == TokenTypeOld::Identifier(Generic) {
                            arg2.value = Self::read_var(active_line[2].clone().value);
                        }
                        else if  active_line[2].clone().token_type == TokenTypeOld::Identifier(Text){
                            arg2.value = arg2.value.trim_matches('"').to_string();
                        }
                        else if arg2.token_type == TokenTypeOld::Identifier(Pop){
                            arg2.value = STACKOLD.lock().unwrap().pop().unwrap();
                        }
                        

                        if arg1.value == arg2.value {
                            i += 1;
                            let mut next_line: Vec<TokenOLD> = Vec::new();
                            while i < tokens.len() && tokens[i].token_type != TokenTypeOld::NewLine {
                                next_line.push(tokens[i].clone());
                                i += 1;
                            }
                            if !next_line.is_empty() {
                                Self::line_handler(next_line);
                            }
                        } else {
                            i += 1;
                            while i < tokens.len() && tokens[i].token_type != TokenTypeOld::NewLine {
                                i += 1;
                            }
                        }
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

        debug(format!("VARS {:?}", VARSOLD.lock().unwrap()));
        debug(format!("FUNS {:?}", FUNCSOLD.lock().unwrap()));
    }


    pub fn check_if_equal(a: &str, b: &str) -> bool {
        true
    }

    pub fn add_func(name: String, func: Vec<Vec<TokenOLD>>){
        FUNCSOLD.lock().unwrap().push(FunctionOLd {
            name,
            action: func,
        });
    }

    pub fn run_func(name: String) {
        let funcs = FUNCSOLD.lock().unwrap();
        if let Some(func) = funcs.iter().find(|f| f.name == name) {
            let actions = func.action.clone();
            drop(funcs);
            for action in actions {
                Self::line_handler(action);
            }
        }
    }

    pub fn line_handler(line: Vec<TokenOLD>) {

        let args:Vec<TokenOLD> = Self::handle_args(line.clone());

        debug(format!("args {:?}", args));

        for (index, token) in args.iter().enumerate() {
            if token.token_type == TokenTypeOld::Identifier(Set) && args.len() == 3{
                let arg1 = args.get(index+1).unwrap().value.clone();
                let mut arg2 = args.get(index+2).unwrap().clone();
                if arg2.token_type == TokenTypeOld::Identifier(Pop){
                    arg2.value = STACKOLD.lock().unwrap().pop().unwrap();
                }
                debug(format!("Arg: {}  Arg: {}", arg1, arg2.value));
                Self::define_var(arg1, arg2.value);
            }
            else if token.token_type == TokenTypeOld::Identifier(Dis) && args.len() == 2 {
                let mut arg1 = args.get(index+1).unwrap().clone();
                if arg1.value == "" || arg1.value.ends_with("\\n") {
                    arg1.value.pop();
                    arg1.value.pop();
                    println!("{}", arg1.value);
                }else{
                    if arg1.token_type == TokenTypeOld::Identifier(Pop){
                        print!("{}", STACKOLD.lock().unwrap().pop().unwrap())
                    } else if arg1.token_type == TokenTypeOld::Identifier(Generic) {
                        print!("{}", Self::read_var(arg1.value.clone()));
                    } else if arg1.token_type == TokenTypeOld::Identifier(Text){
                        print!("{}", arg1.value)
                    }
                }

            }
            else if token.token_type == TokenTypeOld::Identifier(Push) && args.len() == 2 {
                let arg1 = args.get(index+1).unwrap().clone();

                if arg1.token_type == TokenTypeOld::Identifier(Generic) {
                    STACKOLD.lock().unwrap().push(Self::read_var(arg1.value));
                } else {
                    STACKOLD.lock().unwrap().push(arg1.value);
                }
            }
            else if token.token_type == TokenTypeOld::Identifier(Pop) && args.len() == 2 {
                if index > 0 {
                    let argm1 = args.get(index - 1).unwrap().value.clone();
                    if argm1 != "dis" {
                        let arg1 = args.get(index + 1).unwrap().value.clone();
                        Self::define_var(arg1, STACKOLD.lock().unwrap().pop().unwrap())
                    }
                }
            }

            else if token.token_type == TokenTypeOld::Identifier(Func) && args.len() == 2 {
                let arg1 = args.get(index+1).unwrap().value.clone();
                Self::run_func(arg1);
            }
            else if token.token_type == TokenTypeOld::Identifier(Input)  {
                let mut arg1 = String::new();
                if args.len() == 1 {
                    io::stdout().flush().unwrap();
                    io::stdin()
                        .read_line(&mut arg1)
                        .expect("Erreur lors de la lecture de l'entrée");

                    let trimmed_input = arg1.trim().to_string();

                    Self::define_var("INPUT".to_string(), trimmed_input);
                } else if args.len() == 2 {
                    io::stdout().flush().unwrap();
                    io::stdin()
                        .read_line(&mut arg1)
                        .expect("Erreur lors de la lecture de l'entrée");
                    let arg2 = args.get(index+1).unwrap().value.clone();
                    let trimmed_input = arg1.trim().to_string();
                    Self::define_var(arg2, trimmed_input);
                }
            }
        }
    }

    pub fn handle_args(args: Vec<TokenOLD>) -> Vec<TokenOLD> {
        let mut results: Vec<TokenOLD> = Vec::new();
        debug(format!("Args with 0 : {:?}", args));

        let mut i = 0;
        while i < args.len() {
            let token = &args[i];

            if token.token_type == TokenTypeOld::Punctuation(FParenthesis) {
                let mut depth = 1;
                let mut end_index = i + 1;

                while depth > 0 && end_index < args.len() {
                    if args[end_index].token_type == TokenTypeOld::Punctuation(FParenthesis) {
                        depth += 1;
                    } else if args[end_index].token_type == TokenTypeOld::Punctuation(SParenthesis) {
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
                        if arg.token_type == TokenTypeOld::Number {
                            nbr.push(arg.value.parse::<i32>().unwrap());
                        }
                        else if arg.token_type == TokenTypeOld::Identifier(Generic) {
                            nbr.push(Self::read_var(arg.value.clone()).parse::<i32>().unwrap());
                        }
                        else if arg.token_type == TokenTypeOld::Operator {
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

                    results.push(TokenOLD::new(TokenTypeOld::Number, result.to_string()));

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
        let mut vars = VARSOLD.lock().unwrap();
        if let Some(existing) = vars.iter_mut().find(|var| var.name == name) {
            existing.value = value;
        } else {
            vars.push(VariableOLD { name, value });
        }
    }

    pub fn read_var(name: String) -> String {
        let guard = VARSOLD.lock().unwrap();
        for var in guard.iter().rev() {
            if var.name == name {
                return var.value.clone();
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
