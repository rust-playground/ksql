//! #### Syntax & Rules
//!
//! | Token          | Example                  | Syntax Rules                                                                                                                                                                              |
//! |----------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! | `Equals`       | `==`                     | supports both `==` and `=`.                                                                                                                                                               |
//! | `Add`          | `+`                      | N/A                                                                                                                                                                                       |
//! | `Subtract`     | `-`                      | N/A                                                                                                                                                                                       |
//! | `Multiply`     | `*`                      | N/A                                                                                                                                                                                       |
//! | `Divide`       | `/`                      | N/A                                                                                                                                                                                       |
//! | `Gt`           | `>`                      | N/A                                                                                                                                                                                       |
//! | `Gte`          | `>=`                     | N/A                                                                                                                                                                                       |
//! | `Lt`           | `<`                      | N/A                                                                                                                                                                                       |
//! | `Lte`          | `<=`                     | N/A                                                                                                                                                                                       |
//! | `OpenParen`    | `(`                      | N/A                                                                                                                                                                                       |
//! | `CloseParen`   | `)`                      | N/A                                                                                                                                                                                       |
//! | `OpenBracket`  | `[`                      | N/A                                                                                                                                                                                       |
//! | `CloseBracket` | `]`                      | N/A                                                                                                                                                                                       |
//! | `Comma`        | `,`                      | N/A                                                                                                                                                                                       |
//! | `QuotedString` | `"sample text"`          | Must start and end with an unescaped `"` character                                                                                                                                        |
//! | `Number`       | `123.45`                 | Must start and end with a valid `0-9` digit.                                                                                                                                              |
//! | `BoolenTrue`   | `true`                   | Accepts `true` as a boolean only.                                                                                                                                                         |
//! | `BoolenFalse`  | `false`                  | Accepts `false` as a boolean only.                                                                                                                                                        |
//! | `Identifier`   | `.identifier`            | Starts with a `.` and ends with whitespace blank space. This crate currently uses [gjson](https://github.com/tidwall/gjson.rs) and so the full gjson syntax for identifiers is supported. |
//! | `And`          | `&&`                     | N/A                                                                                                                                                                                       |
//! | `Not`          | `!`                      | Must be before Boolean identifier or expression or be followed by an operation                                                                                                            |
//! | `Or`           | <code>&vert;&vert;<code> | N/A                                                                                                                                                                                       |
//! | `Contains`     | `CONTAINS `              | Ends with whitespace blank space.                                                                                                                                                         |
//! | `In`           | `IN `                    | Ends with whitespace blank space.                                                                                                                                                         |
//! | `StartsWith`   | `STARTSWITH `            | Ends with whitespace blank space.                                                                                                                                                         |
//! | `EndsWith`     | `ENDSWITH `              | Ends with whitespace blank space.                                                                                                                                                         |
//! | `NULL`         | `NULL`                   | N/A                                                                                                                                                                                       |

use thiserror::Error;

/// The lexed token.
#[derive(Debug, PartialEq, Eq)]
pub struct Token {
    pub start: u32,
    pub len: u16,
    pub kind: TokenKind,
}

/// The kind of `Token`.
#[derive(Debug, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    QuotedString,
    Number,
    BooleanTrue,
    BooleanFalse,
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
    Not,
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

/// A lexer for the KSQL expression syntax.
pub struct Tokenizer<'a> {
    pos: u32,
    remaining: &'a [u8],
}

impl<'a> Tokenizer<'a> {
    /// Creates a new `Tokenizer` to iterate over tokens
    #[inline]
    #[must_use]
    pub fn new(src: &'a str) -> Self {
        Self::new_bytes(src.as_bytes())
    }

    /// Creates a new `Tokenizer` to iterate over tokens using bytes as the source.
    #[must_use]
    pub fn new_bytes(src: &'a [u8]) -> Self {
        Self {
            pos: 0,
            remaining: src,
        }
    }

    fn next_token(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace();

        if self.remaining.is_empty() {
            Ok(None)
        } else {
            let (kind, bytes_read) = tokenize_single_token(self.remaining)?;
            let token = Token {
                kind,
                start: self.pos,
                len: bytes_read,
            };
            self.chomp(bytes_read);
            Ok(Some(token))
        }
    }

    fn skip_whitespace(&mut self) {
        let skipped = skip_whitespace(self.remaining);
        self.chomp(skipped);
    }

    fn chomp(&mut self, len: u16) {
        self.remaining = &self.remaining[len as usize..];
        self.pos += u32::from(len);
    }
}

impl Iterator for Tokenizer<'_> {
    type Item = Result<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token().transpose()
    }
}

#[inline]
fn skip_whitespace(data: &[u8]) -> u16 {
    take_while(data, |c| c.is_ascii_whitespace()).unwrap_or(0)
}

#[inline]
/// Consumes bytes while a predicate evaluates to true.
fn take_while<F>(data: &[u8], mut pred: F) -> Option<u16>
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
        Some(current_index)
    }
}

/// Result of a single tokenization attempt.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for the lexer.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("invalid identifier: {0}")]
    InvalidIdentifier(String),

    #[error("invalid number: {0}")]
    InvalidNumber(String),

    #[error("invalid boolean: {0}")]
    InvalidBool(String),

    #[error("invalid keyword: {0}")]
    InvalidKeyword(String),

    #[error("Unsupported Character `{0}`")]
    UnsupportedCharacter(u8),

    #[error("Unterminated string `{0}`")]
    UnterminatedString(String),
}

/// Try to lex a single token from the input stream.
fn tokenize_single_token(data: &[u8]) -> Result<(TokenKind, u16)> {
    let b = match data.first() {
        Some(b) => b,
        None => panic!("invalid data passed"),
    };

    let (token, end) = match b {
        b'=' if data.get(1) == Some(&b'=') => (TokenKind::Equals, 2),
        b'=' => (TokenKind::Equals, 1),
        b'+' => (TokenKind::Add, 1),
        b'-' => (TokenKind::Subtract, 1),
        b'*' => (TokenKind::Multiply, 1),
        b'/' => (TokenKind::Divide, 1),
        b'>' if data.get(1) == Some(&b'=') => (TokenKind::Gte, 2),
        b'>' => (TokenKind::Gt, 1),
        b'<' if data.get(1) == Some(&b'=') => (TokenKind::Lte, 2),
        b'<' => (TokenKind::Lt, 1),
        b'(' => (TokenKind::OpenParen, 1),
        b')' => (TokenKind::CloseParen, 1),
        b'[' => (TokenKind::OpenBracket, 1),
        b']' => (TokenKind::CloseBracket, 1),
        b',' => (TokenKind::Comma, 1),
        b'!' => (TokenKind::Not, 1),
        b'"' | b'\'' => tokenize_string(data, *b)?,
        b'.' => tokenize_identifier(data)?,
        b't' | b'f' => tokenize_bool(data)?,
        b'&' if data.get(1) == Some(&b'&') => (TokenKind::And, 2),
        b'|' if data.get(1) == Some(&b'|') => (TokenKind::Or, 2),
        b'O' => tokenize_keyword(data, "OR".as_bytes(), TokenKind::Or)?,
        b'C' => tokenize_keyword(data, "CONTAINS".as_bytes(), TokenKind::Contains)?,
        b'I' => tokenize_keyword(data, "IN".as_bytes(), TokenKind::In)?,
        b'S' => tokenize_keyword(data, "STARTSWITH".as_bytes(), TokenKind::StartsWith)?,
        b'E' => tokenize_keyword(data, "ENDSWITH".as_bytes(), TokenKind::EndsWith)?,
        b'N' => tokenize_null(data)?,
        c if c.is_ascii_digit() => tokenize_number(data)?,
        _ => return Err(Error::UnsupportedCharacter(*b)),
    };
    Ok((token, end))
}

#[inline]
fn tokenize_string(data: &[u8], quote: u8) -> Result<(TokenKind, u16)> {
    let mut last_backslash = false;
    let mut ended_with_terminator = false;

    match take_while(&data[1..], |c| match c {
        b'\\' => {
            last_backslash = true;
            true
        }
        _ if c == quote => {
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
        Some(end) => {
            if ended_with_terminator {
                Ok((TokenKind::QuotedString, end + 2))
            } else {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            }
        }
        None => {
            if !ended_with_terminator || data.len() < 2 {
                Err(Error::UnterminatedString(
                    String::from_utf8_lossy(data).to_string(),
                ))
            } else {
                Ok((TokenKind::QuotedString, 2))
            }
        }
    }
}

#[inline]
fn tokenize_identifier(data: &[u8]) -> Result<(TokenKind, u16)> {
    match take_while(&data[1..], |c| {
        !c.is_ascii_whitespace() && c != b')' && c != b']'
    }) {
        Some(end) => Ok((TokenKind::Identifier, end + 1)),
        None => Err(Error::InvalidIdentifier(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_bool(data: &[u8]) -> Result<(TokenKind, u16)> {
    match take_while(data, |c| c.is_ascii_alphabetic()) {
        Some(end) => match data[..end as usize] {
            [b't', b'r', b'u', b'e'] => Ok((TokenKind::BooleanTrue, end)),
            [b'f', b'a', b'l', b's', b'e'] => Ok((TokenKind::BooleanFalse, end)),
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
fn tokenize_keyword(data: &[u8], keyword: &[u8], kind: TokenKind) -> Result<(TokenKind, u16)> {
    match take_while(data, |c| !c.is_ascii_whitespace()) {
        Some(end) if &data[..end as usize] == keyword && data.len() > keyword.len() => {
            Ok((kind, end))
        }
        _ => Err(Error::InvalidKeyword(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_null(data: &[u8]) -> Result<(TokenKind, u16)> {
    match take_while(data, |c| c.is_ascii_alphabetic()) {
        Some(end) if data[..end as usize] == [b'N', b'U', b'L', b'L'] => Ok((TokenKind::Null, end)),
        _ => Err(Error::InvalidKeyword(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[inline]
fn tokenize_number(data: &[u8]) -> Result<(TokenKind, u16)> {
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
        _ => c.is_ascii_alphanumeric(),
    }) {
        Some(end) if !bad_number => Ok((TokenKind::Number, end)),
        _ => Err(Error::InvalidNumber(
            String::from_utf8_lossy(data).to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lex_test {
        (FAIL: $name:ident, $src:expr, $e:expr ) => {
            #[test]
            fn $name() -> Result<()> {
                let src: &str = $src;
                let mut tokenizer = Tokenizer::new_bytes(src.as_bytes());
                let err = tokenizer.next_token();
                assert_eq!(Err($e), err);
                Ok(())
            }
        };
        ($name:ident, $src:expr, $( $tok:expr ),* ) => {
            #[test]
            fn $name() -> Result<()> {
                let src: &str = $src;
                let mut tokenizer = Tokenizer::new_bytes(src.as_bytes());

                $(
                    let token = tokenizer.next_token()?.unwrap();
                    assert_eq!($tok, token);
                )*
                Ok(())
            }
        };
    }

    // singular
    lex_test!(
        parse_bool_true,
        "true",
        Token {
            kind: TokenKind::BooleanTrue,
            start: 0,
            len: 4
        }
    );
    lex_test!(
        parse_bool_fase,
        "false",
        Token {
            kind: TokenKind::BooleanFalse,
            start: 0,
            len: 5
        }
    );
    lex_test!(
        parse_number_float,
        "123.23",
        Token {
            kind: TokenKind::Number,
            start: 0,
            len: 6
        }
    );
    lex_test!(
        parse_number_exp,
        "1e-10",
        Token {
            kind: TokenKind::Number,
            start: 0,
            len: 5
        }
    );
    lex_test!(
        parse_number_int,
        "123",
        Token {
            kind: TokenKind::Number,
            start: 0,
            len: 3
        }
    );
    lex_test!(
        FAIL: parse_number_invalid,
        "123.23.23",
        Error::InvalidNumber("123.23.23".to_string())
    );
    lex_test!(
        parse_identifier,
        ".properties.first_name",
        Token {
            kind: TokenKind::Identifier,
            start: 0,
            len: 22
        }
    );
    lex_test!(
        FAIL: parse_identifier_blank,
        ".",
        Error::InvalidIdentifier(".".to_string())
    );
    lex_test!(
        parse_string,
        r#""quoted""#,
        Token {
            kind: TokenKind::QuotedString,
            start: 0,
            len: 8
        }
    );
    lex_test!(
        parse_string_blank,
        r#""""#,
        Token {
            kind: TokenKind::QuotedString,
            start: 0,
            len: 2
        }
    );
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
    lex_test!(
        parse_equals_single,
        "=",
        Token {
            kind: TokenKind::Equals,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_equals,
        "==",
        Token {
            kind: TokenKind::Equals,
            start: 0,
            len: 2
        }
    );
    lex_test!(
        parse_add,
        "+",
        Token {
            kind: TokenKind::Add,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_subtracts,
        "-",
        Token {
            kind: TokenKind::Subtract,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_multiple,
        "*",
        Token {
            kind: TokenKind::Multiply,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_divide,
        "/",
        Token {
            kind: TokenKind::Divide,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_gt,
        ">",
        Token {
            kind: TokenKind::Gt,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_gte,
        ">=",
        Token {
            kind: TokenKind::Gte,
            start: 0,
            len: 2
        }
    );
    lex_test!(
        parse_lt,
        "<",
        Token {
            kind: TokenKind::Lt,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_lte,
        "<=",
        Token {
            kind: TokenKind::Lte,
            start: 0,
            len: 2
        }
    );
    lex_test!(
        parse_open_paren,
        "(",
        Token {
            kind: TokenKind::OpenParen,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_close_paran,
        ")",
        Token {
            kind: TokenKind::CloseParen,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_open_bracket,
        "[",
        Token {
            kind: TokenKind::OpenBracket,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_close_bracket,
        "]",
        Token {
            kind: TokenKind::CloseBracket,
            start: 0,
            len: 1
        }
    );
    lex_test!(
        parse_comma,
        ",",
        Token {
            kind: TokenKind::Comma,
            start: 0,
            len: 1
        }
    );

    // more complex
    lex_test!(
        parse_add_ident,
        ".field1 + .field2",
        Token {
            kind: TokenKind::Identifier,
            start: 0,
            len: 7
        },
        Token {
            kind: TokenKind::Add,
            start: 8,
            len: 1
        },
        Token {
            kind: TokenKind::Identifier,
            start: 10,
            len: 7
        }
    );
    lex_test!(
        parse_sub_ident,
        ".field1 - .field2",
        Token {
            kind: TokenKind::Identifier,
            start: 0,
            len: 7
        },
        Token {
            kind: TokenKind::Subtract,
            start: 8,
            len: 1
        },
        Token {
            kind: TokenKind::Identifier,
            start: 10,
            len: 7
        }
    );
    lex_test!(
        parse_brackets,
        ".field1 - ( .field2 + .field3 )",
        Token {
            kind: TokenKind::Identifier,
            start: 0,
            len: 7
        },
        Token {
            kind: TokenKind::Subtract,
            start: 8,
            len: 1
        },
        Token {
            kind: TokenKind::OpenParen,
            start: 10,
            len: 1
        },
        Token {
            kind: TokenKind::Identifier,
            start: 12,
            len: 7
        },
        Token {
            kind: TokenKind::Add,
            start: 20,
            len: 1
        },
        Token {
            kind: TokenKind::Identifier,
            start: 22,
            len: 7
        },
        Token {
            kind: TokenKind::CloseParen,
            start: 30,
            len: 1
        }
    );

    lex_test!(
        parse_or,
        "||",
        Token {
            kind: TokenKind::Or,
            start: 0,
            len: 2
        }
    );
    lex_test!(FAIL: parse_bad_or, "|", Error::UnsupportedCharacter(b'|'));

    lex_test!(
        parse_in,
        " IN ",
        Token {
            kind: TokenKind::In,
            start: 1,
            len: 2
        }
    );
    lex_test!(
        FAIL: parse_bad_in,
        " IN",
        Error::InvalidKeyword("IN".to_string())
    );

    lex_test!(
        parse_contains,
        " CONTAINS ",
        Token {
            kind: TokenKind::Contains,
            start: 1,
            len: 8
        }
    );
    lex_test!(
        FAIL: contains,
        " CONTAINS",
        Error::InvalidKeyword("CONTAINS".to_string())
    );

    lex_test!(
        parse_starts_with,
        " STARTSWITH ",
        Token {
            kind: TokenKind::StartsWith,
            start: 1,
            len: 10
        }
    );
    lex_test!(
        FAIL: parse_bad_starts_with,
        " STARTSWITH",
        Error::InvalidKeyword("STARTSWITH".to_string())
    );

    lex_test!(
        parse_ends_with,
        " ENDSWITH ",
        Token {
            kind: TokenKind::EndsWith,
            start: 1,
            len: 8
        }
    );
    lex_test!(
        FAIL: parse_bad_ends_with,
        " ENDSWITH",
        Error::InvalidKeyword("ENDSWITH".to_string())
    );

    lex_test!(
        parse_null,
        "NULL",
        Token {
            kind: TokenKind::Null,
            start: 0,
            len: 4
        }
    );
    lex_test!(
        FAIL: parse_bad_null,
        "NULLL",
        Error::InvalidKeyword("NULLL".to_string())
    );
    lex_test!(
        parse_and,
        "&&",
        Token {
            kind: TokenKind::And,
            start: 0,
            len: 2
        }
    );
    lex_test!(FAIL: parse_bad_and, "&", Error::UnsupportedCharacter(b'&'));
    lex_test!(
        parse_not,
        "!",
        Token {
            kind: TokenKind::Not,
            start: 0,
            len: 1
        }
    );
}
