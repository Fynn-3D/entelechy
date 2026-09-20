//! Tokenizer for Orla documents.

use crate::{ParseError, Span};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Ident(String),
    Int(i32),
    Float(f32),
    Str(String),
    Punct(&'static str),
    Eof,
}

impl Token {
    pub(crate) fn describe(&self) -> String {
        match self {
            Self::Ident(name) => format!("`{name}`"),
            Self::Int(value) => format!("`{value}`"),
            Self::Float(value) => format!("`{value:?}`"),
            Self::Str(value) => format!("`\"{value}\"`"),
            Self::Punct(punct) => format!("`{punct}`"),
            Self::Eof => "end of file".to_owned(),
        }
    }
}

/// A token with its position and its byte range in the source.
#[derive(Clone, Debug)]
pub(crate) struct Lexeme {
    pub token: Token,
    pub span: Span,
    pub start: usize,
    pub end: usize,
}

const PUNCTS: [&str; 33] = [
    "->", "<-", "<~", "<=", ">=", "==", "!=", "&&", "||", "=>", ">>>", "<<", ">>", "{", "}", "(",
    ")", ":", ";", ",", "=", ".", "@", "+", "-", "*", "/", "%", "<", ">", "&", "|", "^",
];
const SINGLE_EXTRA: [&str; 1] = ["!"];

pub(crate) fn tokenize(source: &str) -> Result<Vec<Lexeme>, ParseError> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut byte = 0;
    let mut line = 1;
    let mut column = 1;
    while index < chars.len() {
        let c = chars[index];
        let span = Span { line, column };
        let start = byte;
        let mut advance = |count: usize, index: &mut usize, byte: &mut usize| {
            for _ in 0..count {
                if chars[*index] == '\n' {
                    line += 1;
                    column = 1;
                } else {
                    column += 1;
                }
                *byte += chars[*index].len_utf8();
                *index += 1;
            }
        };
        if c.is_whitespace() {
            advance(1, &mut index, &mut byte);
            continue;
        }
        if c == '/' && chars.get(index + 1) == Some(&'/') {
            while index < chars.len() && chars[index] != '\n' {
                advance(1, &mut index, &mut byte);
            }
            continue;
        }
        if c == '"' {
            advance(1, &mut index, &mut byte);
            let text_start = index;
            while index < chars.len() && chars[index] != '"' && chars[index] != '\n' {
                advance(1, &mut index, &mut byte);
            }
            if chars.get(index) != Some(&'"') {
                return Err(ParseError {
                    span,
                    message: "string literal is not closed".into(),
                });
            }
            let text: String = chars[text_start..index].iter().collect();
            advance(1, &mut index, &mut byte);
            tokens.push(Lexeme {
                token: Token::Str(text),
                span,
                start,
                end: byte,
            });
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
            {
                advance(1, &mut index, &mut byte);
            }
            let text: String = source[start..byte].to_owned();
            tokens.push(Lexeme {
                token: Token::Ident(text),
                span,
                start,
                end: byte,
            });
            continue;
        }
        if c == '0' && chars.get(index + 1) == Some(&'x') {
            // A hexadecimal `Int`: its 32 bits as written, so `0x80000000`
            // is the sign bit.
            advance(2, &mut index, &mut byte);
            while index < chars.len() && chars[index].is_ascii_hexdigit() {
                advance(1, &mut index, &mut byte);
            }
            let text = &source[start..byte];
            let value = u32::from_str_radix(&text[2..], 16).map_err(|_| ParseError {
                span,
                message: format!("`{text}` is not a 32-bit hexadecimal literal"),
            })?;
            tokens.push(Lexeme {
                token: Token::Int(value as i32),
                span,
                start,
                end: byte,
            });
            continue;
        }
        if c.is_ascii_digit() {
            let mut is_float = false;
            while index < chars.len() && chars[index].is_ascii_digit() {
                advance(1, &mut index, &mut byte);
            }
            if chars.get(index) == Some(&'.')
                && chars.get(index + 1).is_some_and(|c| c.is_ascii_digit())
            {
                is_float = true;
                advance(1, &mut index, &mut byte);
                while index < chars.len() && chars[index].is_ascii_digit() {
                    advance(1, &mut index, &mut byte);
                }
            }
            let text = &source[start..byte];
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
            tokens.push(Lexeme {
                token,
                span,
                start,
                end: byte,
            });
            continue;
        }
        let rest: String = chars[index..chars.len().min(index + 3)].iter().collect();
        let punct = PUNCTS
            .iter()
            .chain(SINGLE_EXTRA.iter())
            .find(|punct| rest.starts_with(**punct));
        match punct {
            Some(punct) => {
                advance(punct.len(), &mut index, &mut byte);
                tokens.push(Lexeme {
                    token: Token::Punct(punct),
                    span,
                    start,
                    end: byte,
                });
            }
            None => {
                return Err(ParseError {
                    span,
                    message: format!("unexpected character `{c}`"),
                });
            }
        }
    }
    tokens.push(Lexeme {
        token: Token::Eof,
        span: Span { line, column },
        start: byte,
        end: byte,
    });
    Ok(tokens)
}
