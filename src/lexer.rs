//! Expressions support most mathematical and string expressions see [here](https://github.com/rust-playground/ksql/LEXER.md) for details of the lexer support and rules.

use thiserror::Error;

#[derive(Debug, PartialEq)]
pub enum Token {
    Identifier(String),
    String(String),
    Number(f64),
    Boolean(bool),
    Null,
    Equals,
    Add,
    Subtract,
    Multiply,
    Divide,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
    Contains,
    In,
    StartsWith,
    EndsWith,
    OpenBracket,
    CloseBracket,
    Comma,
    OpenParen,
    CloseParen,
}

/// Try to lex a single token from the input stream.
pub fn tokenize_single_token(data: &[u8]) -> Result<(Token, usize)> {
    let b = match data.get(0) {
        Some(b) => b,
        None => panic!("invalid data passed"),
    };

    let (token, end) = match b {
        b'=' => (Token::Equals, 1),
        b'+' => (Token::Add, 1),
        b'-' => (Token::Subtract, 1),
        b'*' => (Token::Multiply, 1),
        b'/' => (Token::Divide, 1),
        b'>' if data.get(1) == Some(&b'=') => (Token::Gte, 2),
        b'>' => (Token::Gt, 1),
        b'<' if data.get(1) == Some(&b'=') => (Token::Lte, 2),
        b'<' => (Token::Lt, 1),
        b'(' => (Token::OpenParen, 1),
        b')' => (Token::CloseParen, 1),
        b'[' => (Token::OpenBracket, 1),
        b']' => (Token::CloseBracket, 1),
        b',' => (Token::Comma, 1),
        b'"' => tokenize_string_double_quote(data)?,
        b'\'' => tokenize_string_single_quote(data)?,
        b'.' => tokenize_identifier(data)?,
        b't' | b'f' => tokenize_bool(data)?,
        b'A' => tokenize_and(data)?,
        b'O' => tokenize_or(data)?,
        b'C' => tokenize_contains(data)?,
        b'I' => tokenize_in(data)?,
        b'S' => tokenize_starts_with(data)?,
        b'E' => tokenize_ends_with(data)?,
        b'N' => tokenize_null(data)?,
        c if c.is_ascii_digit() => tokenize_number(data)?,
        _ => return Err(Error::UnsupportedCharacter(*b)),
    };
    Ok((token, end))
}

#[inline]
fn tokenize_and(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "AND" if data.len() > 3 => Ok((Token::And, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_null(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c.is_ascii_alphabetic()) {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "NULL" => Ok((Token::Null, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_ends_with(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "ENDSWITH" if data.len() > 8 => Ok((Token::EndsWith, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_starts_with(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "STARTSWITH" if data.len() > 10 => Ok((Token::StartsWith, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_in(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "IN" if data.len() > 2 => Ok((Token::In, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_contains(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "CONTAINS" if data.len() > 8 => Ok((Token::Contains, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_or(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c != b' ') {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "OR" if data.len() > 2 => Ok((Token::Or, end)),
            _ => Err(Error::InvalidOperator(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidOperator(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_identifier(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(&data[1..], |c| c != b' ') {
        Some((_, mut end)) => {
            if data.len() > end {
                end += 1;
            }
            Ok((
                Token::Identifier(String::from_utf8_lossy(&data[1..end]).to_string()),
                end,
            ))
        }
        None => Err(Error::InvalidIdentifier(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_bool(data: &[u8]) -> Result<(Token, usize)> {
    match take_while(data, |c| c.is_ascii_alphabetic()) {
        Some((_, end)) => match String::from_utf8_lossy(&data[..end]).as_ref() {
            "true" => Ok((Token::Boolean(true), end)),
            "false" => Ok((Token::Boolean(false), end)),
            _ => Err(Error::InvalidBool(
                String::from_utf8_lossy(data).to_string(),
            )),
        },
        None => Err(Error::InvalidBool(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_number(data: &[u8]) -> Result<(Token, usize)> {
    let mut dot_seen = false;
    let mut bad_number = false;

    match take_while(data, |c| match c {
        b'.' => {
            if dot_seen {
                bad_number = true;
                false
            } else {
                dot_seen = true;
                true
            }
        }
        b'-' | b'+' => true,
        c if c.is_ascii_alphanumeric() => true,
        _ => false,
    }) {
        Some((_, end)) => {
            if bad_number {
                Err(Error::InvalidNumber(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                match String::from_utf8_lossy(&data[..end]).parse::<f64>() {
                    Ok(n) => Ok((Token::Number(n), end)),
                    _ => Err(Error::InvalidNumber(
                        String::from_utf8_lossy(&data[..end]).to_string(),
                    )),
                }
            }
        }
        None => Err(Error::InvalidNumber(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_string_single_quote(data: &[u8]) -> Result<(Token, usize)> {
    let mut last_backslash = false;
    let mut ended_with_terminator = false;

    match take_while(&data[1..], |c| match c {
        b'\\' => {
            last_backslash = true;
            true
        }
        b'\'' => {
            if last_backslash {
                last_backslash = false;
                true
            } else {
                ended_with_terminator = true;
                false
            }
        }
        _ => {
            last_backslash = false;
            true
        }
    }) {
        Some((_, end)) => {
            if !ended_with_terminator {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                Ok((
                    Token::String(String::from_utf8_lossy(&data[1..end + 1]).to_string()),
                    end + 2,
                ))
            }
        }
        None => {
            if !ended_with_terminator || data.len() < 2 {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                Ok((
                    Token::String(String::from_utf8_lossy(&data[..0]).to_string()),
                    1,
                ))
            }
        }
    }
}

#[inline]
fn tokenize_string_double_quote(data: &[u8]) -> Result<(Token, usize)> {
    let mut last_backslash = false;
    let mut ended_with_terminator = false;

    match take_while(&data[1..], |c| match c {
        b'\\' => {
            last_backslash = true;
            true
        }
        b'"' => {
            if last_backslash {
                last_backslash = false;
                true
            } else {
                ended_with_terminator = true;
                false
            }
        }
        _ => {
            last_backslash = false;
            true
        }
    }) {
        Some((_, end)) => {
            if !ended_with_terminator {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                Ok((
                    Token::String(String::from_utf8_lossy(&data[1..end + 1]).to_string()),
                    end + 2,
                ))
            }
        }
        None => {
            if !ended_with_terminator || data.len() < 2 {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                Ok((
                    Token::String(String::from_utf8_lossy(&data[..0]).to_string()),
                    1,
                ))
            }
        }
    }
}

#[inline]
fn skip_whitespace(data: &[u8]) -> usize {
    match take_while(data, |c| c.is_ascii_whitespace()) {
        Some((_, bytes_skipped)) => bytes_skipped,
        None => 0,
    }
}

#[inline]
/// Consumes bytes while a predicate evaluates to true.
fn take_while<F>(data: &[u8], mut pred: F) -> Option<(&[u8], usize)>
where
    F: FnMut(u8) -> bool,
{
    let mut current_index = 0;

    for b in data {
        if !pred(*b) {
            break;
        }
        current_index += 1;
    }

    if current_index == 0 {
        None
    } else {
        Some((&data[..current_index], current_index))
    }
}

pub struct Tokenizer<'a> {
    current_index: usize,
    remaining_bytes: &'a [u8],
}

impl<'a> Tokenizer<'a> {
    pub fn new(src: &[u8]) -> Tokenizer {
        Tokenizer {
            current_index: 0,
            remaining_bytes: src,
        }
    }

    pub fn next_token(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace();

        if self.remaining_bytes.is_empty() {
            Ok(None)
        } else {
            let tok = self._next_token()?;
            Ok(Some(tok))
        }
    }

    pub fn tokenize(src: &[u8]) -> Result<Vec<Token>> {
        let mut tokenizer = Tokenizer::new(src);
        let mut tokens = Vec::new();

        while let Some(tok) = tokenizer.next_token()? {
            tokens.push(tok);
        }
        Ok(tokens)
    }

    fn skip_whitespace(&mut self) {
        let skipped = skip_whitespace(self.remaining_bytes);
        self.chomp(skipped);
    }

    fn _next_token(&mut self) -> Result<Token> {
        let (tok, bytes_read) = tokenize_single_token(self.remaining_bytes)?;
        self.chomp(bytes_read);
        Ok(tok)
    }

    fn chomp(&mut self, num_bytes: usize) {
        self.remaining_bytes = &self.remaining_bytes[num_bytes..];
        self.current_index += num_bytes;
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug, PartialEq)]
pub enum Error {
    #[error("invalid identifier: {0}")]
    InvalidIdentifier(String),

    #[error("invalid number: {0}")]
    InvalidNumber(String),

    #[error("invalid boolean: {0}")]
    InvalidBool(String),

    #[error("invalid operator: {0}")]
    InvalidOperator(String),

    #[error("Unsupported Character `{0}`")]
    UnsupportedCharacter(u8),

    #[error("Unterminated string `{0}`")]
    UnterminatedString(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lex_test {
        (FAIL: $name:ident, $src:expr, $e:expr ) => {
            #[test]
            fn $name() -> Result<()> {
                let src: &str = $src;
                let mut tokenizer = Tokenizer::new(src.as_bytes());
                let err = tokenizer.next_token();
                assert_eq!(Err($e), err);
                Ok(())
            }
        };
        ($name:ident, $src:expr, $( $tok:expr ),* ) => {
            #[test]
            fn $name() -> Result<()> {
                let src: &str = $src;
                let mut tokenizer = Tokenizer::new(src.as_bytes());

                $(
                    let token = tokenizer.next_token()?.unwrap();
                    assert_eq!($tok, token);
                )*
                Ok(())
            }
        };
    }

    // singular
    lex_test!(parse_bool_true, "true", Token::Boolean(true));
    lex_test!(parse_bool_fase, "false", Token::Boolean(false));
    lex_test!(parse_number_float, "123.23", Token::Number(123.23));
    lex_test!(parse_number_exp, "1e-10", Token::Number(1e-10));
    lex_test!(parse_number_int, "123", Token::Number(123_f64));
    lex_test!(
        FAIL: parse_number_invalid,
        "123.23.23",
        Error::InvalidNumber("123.23.23".to_string())
    );
    lex_test!(
        parse_identifier,
        ".properties.first_name",
        Token::Identifier("properties.first_name".to_string())
    );
    lex_test!(
        FAIL: parse_identifier_blank,
        ".",
        Error::InvalidIdentifier(".".to_string())
    );
    lex_test!(
        parse_string,
        r#""quoted""#,
        Token::String("quoted".to_string())
    );
    lex_test!(parse_string_blank, r#""""#, Token::String("".to_string()));
    lex_test!(
        FAIL: parse_string_unterminated,
        r#""dfg"#,
        Error::UnterminatedString(r#""dfg"#.to_string())
    );
    lex_test!(
        FAIL: parse_string_unterminated2,
        r#"""#,
        Error::UnterminatedString(r#"""#.to_string())
    );
    lex_test!(parse_equals, "=", Token::Equals);
    lex_test!(parse_add, "+", Token::Add);
    lex_test!(parse_subtracts, "-", Token::Subtract);
    lex_test!(parse_multiple, "*", Token::Multiply);
    lex_test!(parse_divide, "/", Token::Divide);
    lex_test!(parse_gt, ">", Token::Gt);
    lex_test!(parse_gte, ">=", Token::Gte);
    lex_test!(parse_lt, "<", Token::Lt);
    lex_test!(parse_lte, "<=", Token::Lte);
    lex_test!(parse_open_paren, "(", Token::OpenParen);
    lex_test!(parse_close_paran, ")", Token::CloseParen);
    lex_test!(parse_open_bracket, "[", Token::OpenBracket);
    lex_test!(parse_close_bracket, "]", Token::CloseBracket);
    lex_test!(parse_comma, ",", Token::Comma);

    // more complex
    lex_test!(
        parse_add_ident,
        ".field1 + .field2",
        Token::Identifier("field1".to_string()),
        Token::Add,
        Token::Identifier("field2".to_string())
    );
    lex_test!(
        parse_sub_ident,
        ".field1 - .field2",
        Token::Identifier("field1".to_string()),
        Token::Subtract,
        Token::Identifier("field2".to_string())
    );
    lex_test!(
        parse_brackets,
        ".field1 - ( .field2 + .field3 )",
        Token::Identifier("field1".to_string()),
        Token::Subtract,
        Token::OpenParen,
        Token::Identifier("field2".to_string()),
        Token::Add,
        Token::Identifier("field3".to_string()),
        Token::CloseParen
    );

    lex_test!(parse_or, " OR ", Token::Or);
    lex_test!(
        FAIL: parse_bad_or,
        " OR",
        Error::InvalidOperator("OR".to_string())
    );

    lex_test!(parse_in, " IN ", Token::In);
    lex_test!(
        FAIL: parse_bad_in,
        " IN",
        Error::InvalidOperator("IN".to_string())
    );

    lex_test!(parse_contains, " CONTAINS ", Token::Contains);
    lex_test!(
        FAIL: contains,
        " CONTAINS",
        Error::InvalidOperator("CONTAINS".to_string())
    );

    lex_test!(parse_starts_with, " STARTSWITH ", Token::StartsWith);
    lex_test!(
        FAIL: parse_bad_starts_with,
        " STARTSWITH",
        Error::InvalidOperator("STARTSWITH".to_string())
    );

    lex_test!(parse_ends_with, " ENDSWITH ", Token::EndsWith);
    lex_test!(
        FAIL: parse_bad_ends_with,
        " ENDSWITH",
        Error::InvalidOperator("ENDSWITH".to_string())
    );

    lex_test!(parse_null, "NULL", Token::Null);
    lex_test!(
        FAIL: parse_bad_null,
        "NULLL",
        Error::InvalidOperator("NULLL".to_string())
    );
    lex_test!(parse_and, " AND ", Token::And);
    lex_test!(
        FAIL: parse_bad_and,
        " AND",
        Error::InvalidOperator("AND".to_string())
    );
}
