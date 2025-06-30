use logos::Logos;

#[derive(Clone, Debug, PartialEq, Logos)]
pub enum Token {
    #[regex(r"[ \t\n\r\f]+", logos::skip)]
    #[regex(r"//[^\n\r]*", logos::skip)]
    #[token("with")]
    With,
    #[token("obj")]
    Obj,
    #[token("fn")]
    Fn,
    #[token("impl")]
    Impl,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    //#[token("and")]
    //And,
    //#[token("or")]
    //Or,
    //#[token("whl")]
    //While,
    //#[token("in")]
    //In,
    #[token("for")]
    For,
    #[token("const")]
    Const,
    #[token("self")]
    SelfKw,
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("%")]
    Percent,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token("/")]
    Slash,
    #[token(".")]
    Dot,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(">")]
    Greater,
    #[token("<")]
    Lower,
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice().to_string())]
    StringLiteral(String),
    #[regex(r"\d+\.\d+", |lex| lex.slice().parse::<f64>().unwrap())]
    Float(f64),
    #[regex(r"\d+", |lex| lex.slice().parse::<i32>().unwrap())]
    Integer(i32),
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null,
    EOF,
}

pub fn tokenize(source: String) -> Vec<Token> {
    let mut lexer = Token::lexer(&source);
    let mut tokens: Vec<Token> = Vec::new();

    while let Some(token_result) = lexer.next() {
        match token_result {
            Ok(token) => tokens.push(token),
            Err(_) => {
                eprintln!(
                    "Lexer Error: Unexpected token at position {}",
                    lexer.span().start
                );
                continue;
            }
        }
    }

    tokens.push(Token::EOF);
    tokens
}