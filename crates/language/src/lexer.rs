//! Tokenizer for `.orla` sources.

use crate::{ParseError, Span};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Ident(String),
    Int(i32),
    Float(f32),
    Punct(&'static str),
    Eof,
}

impl Token {
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Ident(name) => format!("`{name}`"),
            Self::Int(value) => format!("`{value}`"),
            Self::Float(value) => format!("`{value:?}`"),
            Self::Punct(punct) => format!("`{punct}`"),
            Self::Eof => "end of file".to_owned(),
        }
    }
}

const PUNCTS: [&str; 24] = [
    "->", "<=", ">=", "==", "!=", "&&", "||", "{", "}", "(", ")", ":", ";", ",", "=", ".", "@",
    "+", "-", "*", "/", "<", ">", "!",
];

pub(crate) fn tokenize(source: &str) -> Result<Vec<(Token, Span)>, ParseError> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut line = 1;
    let mut column = 1;
    while index < chars.len() {
        let c = chars[index];
        let span = Span { line, column };
        let mut advance = |count: usize, index: &mut usize| {
            for _ in 0..count {
                if chars[*index] == '\n' {
                    line += 1;
                    column = 1;
                } else {
                    column += 1;
                }
                *index += 1;
            }
        };
        if c.is_whitespace() {
            advance(1, &mut index);
            continue;
        }
        if c == '/' && chars.get(index + 1) == Some(&'/') {
            while index < chars.len() && chars[index] != '\n' {
                advance(1, &mut index);
            }
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = index;
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
            {
                advance(1, &mut index);
            }
            let text: String = chars[start..index].iter().collect();
            tokens.push((Token::Ident(text), span));
            continue;
        }
        if c.is_ascii_digit() {
            let start = index;
            let mut is_float = false;
            while index < chars.len() && chars[index].is_ascii_digit() {
                advance(1, &mut index);
            }
            if chars.get(index) == Some(&'.')
                && chars.get(index + 1).is_some_and(|c| c.is_ascii_digit())
            {
                is_float = true;
                advance(1, &mut index);
                while index < chars.len() && chars[index].is_ascii_digit() {
                    advance(1, &mut index);
                }
            }
            let text: String = chars[start..index].iter().collect();
            let token = if is_float {
                Token::Float(text.parse().map_err(|_| ParseError {
                    span,
                    message: format!("`{text}` is not a valid float literal"),
                })?)
            } else {
                Token::Int(text.parse().map_err(|_| ParseError {
                    span,
                    message: format!("`{text}` does not fit in a 32-bit integer"),
                })?)
            };
            tokens.push((token, span));
            continue;
        }
        let rest: String = chars[index..chars.len().min(index + 2)].iter().collect();
        match PUNCTS.iter().find(|punct| rest.starts_with(**punct)) {
            Some(punct) => {
                advance(punct.len(), &mut index);
                tokens.push((Token::Punct(punct), span));
            }
            None => {
                return Err(ParseError {
                    span,
                    message: format!("unexpected character `{c}`"),
                });
            }
        }
    }
    tokens.push((Token::Eof, Span { line, column }));
    Ok(tokens)
}
