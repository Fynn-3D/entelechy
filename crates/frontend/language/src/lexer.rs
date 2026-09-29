//! Tokenizer for Orla documents.

use crate::{ParseError, Span};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Token {
    Ident(String),
    /// An integer as written; the compiler picks its width.
    Int(i64),
    /// A float as written; the compiler picks its width.
    Float(f64),
    /// A float written with `d`: a `Double`.
    Double(f64),
    /// An integer written with `l`: a `Long`.
    Long(i64),
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
            Self::Double(value) => format!("`{value:?}d`"),
            Self::Long(value) => format!("`{value}l`"),
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

const PUNCTS: [&str; 37] = [
    "..=", "..", "->", "<-", "<~", "<=", ">=", "==", "!=", "&&", "||", "=>", ">>>", "<<", ">>",
    "{", "}", "(", ")", "[", "]", ":", ";", ",", "=", ".", "@", "+", "-", "*", "/", "%", "<", ">",
    "&", "|", "^",
];
const SINGLE_EXTRA: [&str; 1] = ["!"];

/// `text` written as a string a document reads back as `text`: in quotes,
/// a new line, a tab, a quote and a backslash written as `\n`, `\t`, `\"`
/// and `\\`.
pub fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

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
            let opening = index;
            advance(1, &mut index, &mut byte);
            // A string on one line; `\n`, `\t`, `\"` and `\\` write a new
            // line, a tab, a quote and a backslash in it.
            let mut text = String::new();
            while index < chars.len() && chars[index] != '"' && chars[index] != '\n' {
                if chars[index] == '\\' {
                    // One line: the escape is as far along it as from the quote.
                    let at = Span {
                        line: span.line,
                        column: span.column + (index - opening) as u32,
                    };
                    let written = match chars.get(index + 1) {
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('"') => '"',
                        Some('\\') => '\\',
                        other => {
                            let what = match other {
                                None | Some('\n' | '\r') => "the line's end".to_owned(),
                                Some(c) => format!("`\\{c}`"),
                            };
                            return Err(ParseError {
                                span: at,
                                message: format!(
                                    "a string writes a new line, a tab, a quote or a backslash as `\\n`, `\\t`, `\\\"` or `\\\\`, not {what}"
                                ),
                            });
                        }
                    };
                    text.push(written);
                    advance(2, &mut index, &mut byte);
                    continue;
                }
                text.push(chars[index]);
                advance(1, &mut index, &mut byte);
            }
            if chars.get(index) != Some(&'"') {
                return Err(ParseError {
                    span,
                    message: "string literal is not closed".into(),
                });
            }
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
        // A number's suffix, `d` (`Double`) or `l` (`Long`), standing alone.
        let suffix = |at: usize, letters: &[char]| {
            chars.get(at).copied().filter(|c| {
                letters.contains(c)
                    && !chars
                        .get(at + 1)
                        .is_some_and(|next| next.is_ascii_alphanumeric() || *next == '_')
            })
        };
        if c == '0' && chars.get(index + 1) == Some(&'x') {
            // Hexadecimal is bits as written (`0x80000000` is an `Int`'s
            // sign bit); over eight digits or an `l` make a `Long`.
            advance(2, &mut index, &mut byte);
            while index < chars.len() && chars[index].is_ascii_hexdigit() {
                advance(1, &mut index, &mut byte);
            }
            let text = &source[start..byte];
            let digits = &text[2..];
            let long = suffix(index, &['l']).is_some();
            if long {
                advance(1, &mut index, &mut byte);
            }
            let token = match u64::from_str_radix(digits, 16) {
                Ok(bits) if long || digits.len() > 8 => Token::Long(bits as i64),
                Ok(bits) if bits <= u64::from(u32::MAX) => {
                    Token::Int(i64::from(bits as u32 as i32))
                }
                _ => {
                    return Err(ParseError {
                        span,
                        message: format!("`{text}` is not a 32- or 64-bit hexadecimal literal"),
                    });
                }
            };
            tokens.push(Lexeme {
                token,
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
            // An exponent, `1e30` or `2.5E-3`, makes a float.
            let signed = matches!(chars.get(index + 1), Some('+' | '-'));
            if matches!(chars.get(index), Some('e' | 'E'))
                && chars
                    .get(index + 1 + usize::from(signed))
                    .is_some_and(|c| c.is_ascii_digit())
            {
                is_float = true;
                advance(1 + usize::from(signed), &mut index, &mut byte);
                while index < chars.len() && chars[index].is_ascii_digit() {
                    advance(1, &mut index, &mut byte);
                }
            }
            let text = &source[start..byte];
            let written = suffix(index, &['d', 'l']);
            if written.is_some() {
                advance(1, &mut index, &mut byte);
            }
            let float = |text: &str| {
                text.parse::<f64>().map_err(|_| ParseError {
                    span,
                    message: format!("`{text}` is not a valid float literal"),
                })
            };
            let integer = |text: &str| {
                text.parse::<i64>().map_err(|_| ParseError {
                    span,
                    message: format!("`{text}` does not fit in a 64-bit integer"),
                })
            };
            let token = match (written, is_float) {
                (Some('d'), _) => Token::Double(float(text)?),
                (Some(_), true) => {
                    return Err(ParseError {
                        span,
                        message: format!("`{text}l`: a `Long` is a whole number"),
                    });
                }
                (Some(_), false) => Token::Long(integer(text)?),
                (None, true) => Token::Float(float(text)?),
                (None, false) => Token::Int(integer(text)?),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_string_writes_a_new_line_a_tab_a_quote_and_a_backslash() {
        let written = "a\nb\tc\"d\\e";
        let read = tokenize(&quoted(written)).unwrap();
        assert_eq!(read[0].token, Token::Str(written.into()));
        let lexed = tokenize(r#""a\nb\tc\"d\\e""#).unwrap();
        assert_eq!(lexed[0].token, Token::Str("a\nb\tc\"d\\e".into()));
        let error = tokenize(r#"  "C:\data""#).unwrap_err();
        assert!(error.message.contains("not `\\d`"), "{error}");
        assert_eq!((error.span.line, error.span.column), (1, 6));
        let error = tokenize("\"ab\\\nc\"").unwrap_err();
        assert!(error.message.ends_with("not the line's end"), "{error}");
    }
}
