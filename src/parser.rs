//! Parser is used to parse an expression for use against JSON data.
//!
//! ```rust
//! use ksql::parser::{Parser, Value};
//! use std::error::Error;
//!
//! fn main() -> Result<(), Box<dyn Error>>{
//!     let src = r#"{"name":"MyCompany", "properties":{"employees": 50}"#.as_bytes();
//!     let ex = Parser::parse(".properties.employees > 20")?;
//!     let result = ex.calculate(src)?;
//!     assert_eq!(Value::Bool(true), result);
//!     Ok(())
//! }
//! ```
//!

use crate::lexer::{Token, TokenKind, Tokenizer};
use anyhow::anyhow;
use gjson::Kind;
use std::collections::BTreeMap;
use std::fmt::{Debug, Display, Formatter};
use thiserror::Error;

/// Represents the calculated Expression result.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Null,
    String(String),
    Number(f64),
    Bool(bool),
    Object(BTreeMap<String, Value>),
    Array(Vec<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::String(s) => write!(f, r#""{}""#, s),
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Object(o) => write!(f, "{}", {
                let mut s = String::new();
                s.push('{');
                for (k, v) in o.iter() {
                    s.push_str(&format!(r#""{}":{}"#, k, v));
                }
                s.push('}');
                s
            }),
            Value::Array(a) => write!(f, "{}", {
                let mut s = String::new();
                s.push('[');
                for v in a.iter() {
                    s.push_str(&format!("{},", v));
                }
                s = s.trim_end_matches(',').to_string();
                s.push(']');
                s
            }),
        }
    }
}

impl<'a> From<gjson::Value<'a>> for Value {
    fn from(v: gjson::Value) -> Self {
        match v.kind() {
            Kind::Null => Value::Null,
            Kind::String => Value::String(v.str().to_string()),
            Kind::Number => Value::Number(v.f64()),
            Kind::False => Value::Bool(false),
            Kind::True => Value::Bool(true),
            Kind::Array => {
                let arr = v.array().into_iter().map(Into::into).collect();
                Value::Array(arr)
            }
            Kind::Object => {
                let mut m = BTreeMap::new();
                v.each(|k, v| {
                    m.insert(k.str().to_string(), v.into());
                    true
                });
                Value::Object(m)
            }
        }
    }
}

/// Represents a stateless parsed expression that can be applied to JSON data.
pub trait Expression: Debug {
    /// Will execute the parsed expression and apply it against the supplied json data.
    ///
    /// # Warnings
    ///
    /// This function assumes that the supplied JSON data is valid.
    ///
    /// # Errors
    ///
    /// Will return `Err` if the expression cannot be applied to the supplied data due to invalid
    /// data type comparisons.
    fn calculate(&self, json: &[u8]) -> Result<Value>;
}

/// Is an alias for a Box<dyn Expression>
type BoxedExpression = Box<dyn Expression>;

/// Parses a supplied expression and returns a `BoxedExpression`.
pub struct Parser<'a> {
    exp: &'a [u8],
}

impl<'a> Parser<'a> {
    fn new(exp: &'a [u8]) -> Self {
        Parser { exp }
    }

    /// parses the provided expression and turning it into a computation that can be applied to some
    /// source data.
    ///
    /// # Errors
    ///
    /// Will return `Err` the expression is invalid.
    #[inline]
    pub fn parse(expression: &str) -> anyhow::Result<BoxedExpression> {
        Parser::parse_bytes(expression.as_bytes())
    }

    /// parses the provided expression as bytes and turning it into a computation that can be applied to some
    /// source data.
    ///
    /// # Errors
    ///
    /// Will return `Err` the expression is invalid.
    pub fn parse_bytes(expression: &[u8]) -> anyhow::Result<BoxedExpression> {
        let tokens = Tokenizer::tokenize_bytes(expression)?;
        let mut pos = 0;
        let parser = Parser::new(expression);
        let result = parser.parse_value(&tokens, &mut pos)?;

        if let Some(result) = result {
            Ok(result)
        } else {
            Err(anyhow!("no expression results found"))
        }
    }

    fn parse_value(
        &self,
        tokens: &[Token],
        pos: &mut usize,
    ) -> anyhow::Result<Option<BoxedExpression>> {
        if let Some(tok) = tokens.get(*pos) {
            *pos += 1;
            match tok.kind {
                TokenKind::Identifier => self.parse_op(
                    Box::new(Ident {
                        ident: String::from_utf8_lossy(
                            &self.exp[(tok.start + 1) as usize..tok.end as usize],
                        )
                        .into_owned(),
                    }),
                    tokens,
                    pos,
                ),
                TokenKind::QuotedString => self.parse_op(
                    Box::new(Str {
                        s: String::from_utf8_lossy(
                            &self.exp[(tok.start + 1) as usize..(tok.end - 1) as usize],
                        )
                        .into_owned(),
                    }),
                    tokens,
                    pos,
                ),
                TokenKind::Number => self.parse_op(
                    Box::new(Num {
                        n: String::from_utf8_lossy(&self.exp[tok.start as usize..tok.end as usize])
                            .parse()?,
                    }),
                    tokens,
                    pos,
                ),
                TokenKind::BooleanTrue => self.parse_op(Box::new(Bool { b: true }), tokens, pos),
                TokenKind::BooleanFalse => self.parse_op(Box::new(Bool { b: false }), tokens, pos),
                TokenKind::Null => self.parse_op(Box::new(Null {}), tokens, pos),
                TokenKind::Not => {
                    let v = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no identifier after !")), Ok)?;
                    self.parse_op(Box::new(Not { value: v }), tokens, pos)
                }
                TokenKind::OpenBracket => {
                    let mut arr = Vec::new();

                    while let Some(v) = self.parse_value(tokens, pos)? {
                        arr.push(v);
                    }
                    let arr = Arr { arr };
                    self.parse_op(Box::new(arr), tokens, pos)
                }
                TokenKind::Comma => match self.parse_value(tokens, pos)? {
                    Some(v) => Ok(Some(v)),
                    None => Err(anyhow!("value required after comma: {:?}", tok)),
                },
                TokenKind::OpenParen => {
                    let op = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value between ()")), Ok)?;
                    self.parse_op(op, tokens, pos)
                }
                TokenKind::CloseParen => Err(anyhow!("no value between ()")),
                TokenKind::CloseBracket => Ok(None),
                _ => Err(anyhow!("invalid value: {:?}", tok)),
            }
        } else {
            Ok(None)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_op(
        &self,
        value: BoxedExpression,
        tokens: &[Token],
        pos: &mut usize,
    ) -> anyhow::Result<Option<BoxedExpression>> {
        if let Some(tok) = tokens.get(*pos) {
            *pos += 1;
            match tok.kind {
                TokenKind::In => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after IN")), Ok)?;
                    Ok(Some(Box::new(In { left: value, right })))
                }
                TokenKind::Contains => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after CONTAINS")), Ok)?;
                    Ok(Some(Box::new(Contains { left: value, right })))
                }
                TokenKind::StartsWith => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after STARTSWITH")), Ok)?;
                    Ok(Some(Box::new(StartsWith { left: value, right })))
                }
                TokenKind::EndsWith => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after ENDSWITH")), Ok)?;
                    Ok(Some(Box::new(EndsWith { left: value, right })))
                }
                TokenKind::And => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after AND")), Ok)?;
                    Ok(Some(Box::new(And { left: value, right })))
                }
                TokenKind::Or => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after OR")), Ok)?;
                    Ok(Some(Box::new(Or { left: value, right })))
                }
                TokenKind::Gt => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after >")), Ok)?;
                    Ok(Some(Box::new(Gt { left: value, right })))
                }
                TokenKind::Gte => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after >=")), Ok)?;
                    Ok(Some(Box::new(Gte { left: value, right })))
                }
                TokenKind::Lt => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after <")), Ok)?;
                    Ok(Some(Box::new(Lt { left: value, right })))
                }
                TokenKind::Lte => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after <=")), Ok)?;
                    Ok(Some(Box::new(Lte { left: value, right })))
                }
                TokenKind::Equals => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after ==")), Ok)?;
                    Ok(Some(Box::new(Eq { left: value, right })))
                }
                TokenKind::Add => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after +")), Ok)?;
                    Ok(Some(Box::new(Add { left: value, right })))
                }
                TokenKind::Subtract => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after -")), Ok)?;
                    Ok(Some(Box::new(Sub { left: value, right })))
                }
                TokenKind::Multiply => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after *")), Ok)?;
                    Ok(Some(Box::new(Mult { left: value, right })))
                }
                TokenKind::Divide => {
                    let right = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value after /")), Ok)?;
                    Ok(Some(Box::new(Div { left: value, right })))
                }
                TokenKind::Not => {
                    let op = self
                        .parse_op(value, tokens, pos)
                        .map_or_else(|_| Err(anyhow!("invalid operation after !")), Ok)?;
                    if let Some(value) = op {
                        let n = Not { value };
                        Ok(Some(Box::new(n)))
                    } else {
                        Err(anyhow!("no operator after !"))
                    }
                }
                TokenKind::OpenParen => {
                    let op = self
                        .parse_value(tokens, pos)?
                        .map_or_else(|| Err(anyhow!("no value between ()")), Ok)?;
                    self.parse_op(op, tokens, pos)
                }
                TokenKind::CloseBracket | TokenKind::CloseParen => Ok(Some(value)),
                _ => Err(anyhow!(
                    "invalid token after ident '{:?}'",
                    String::from_utf8_lossy(&self.exp[tok.start as usize..=tok.end as usize])
                )),
            }
        } else {
            Ok(Some(value))
        }
    }
}

#[derive(Debug)]
struct Add {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Add {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(ref s2)) => Ok(Value::String(s1 + s2)),
            (Value::String(s1), Value::Null) => Ok(Value::String(s1)),
            (Value::Null, Value::String(s2)) => Ok(Value::String(s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
            (Value::Number(n1), Value::Null) => Ok(Value::Number(n1)),
            (Value::Null, Value::Number(n2)) => Ok(Value::Number(n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} + {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Sub {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Sub {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 - n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} - {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Mult {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Mult {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 * n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} * {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Div {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Div {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 / n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} / {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Eq {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Eq {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;
        Ok(Value::Bool(left == right))
    }
}

#[derive(Debug)]
struct Gt {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Gt {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 > s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 > n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} > {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Gte {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Gte {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 >= s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 >= n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} >= {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Lt {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Lt {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 < s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 < n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} < {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Lte {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Lte {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 <= s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 <= n2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} <= {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Not {
    value: BoxedExpression,
}

impl Expression for Not {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let v = self.value.calculate(src)?;
        match v {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            v => Err(Error::UnsupportedTypeComparison(format!("{:?} for !", v))),
        }
    }
}

#[derive(Debug)]
struct Ident {
    ident: String,
}

impl Expression for Ident {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        Ok(unsafe { gjson::get_bytes(src, &self.ident).into() })
    }
}

#[derive(Debug)]
struct Str {
    s: String,
}

impl Expression for Str {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::String(self.s.clone()))
    }
}

#[derive(Debug)]
struct Num {
    n: f64,
}

impl Expression for Num {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Number(self.n))
    }
}

#[derive(Debug)]
struct Bool {
    b: bool,
}

impl Expression for Bool {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Bool(self.b))
    }
}

#[derive(Debug)]
struct Null;

impl Expression for Null {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Null)
    }
}

#[derive(Debug)]
struct Or {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Or {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 || b2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} || {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct And {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for And {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 && b2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} && {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Contains {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Contains {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;
        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1.contains(&s2))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} CONTAINS {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct StartsWith {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for StartsWith {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1.starts_with(&s2))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} + {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct EndsWith {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for EndsWith {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1.ends_with(&s2))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} + {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct In {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for In {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (v, Value::Array(a)) => Ok(Value::Bool(a.contains(&v))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} + {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct Arr {
    arr: Vec<BoxedExpression>,
}

impl Expression for Arr {
    fn calculate(&self, src: &[u8]) -> Result<Value> {
        let mut arr = Vec::new();
        for e in &self.arr {
            arr.push(e.calculate(src)?);
        }
        Ok(Value::Array(arr))
    }
}

/// Result type for the `parse` function.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for the expression parser.
#[derive(Error, Debug, PartialEq)]
pub enum Error {
    #[error("unsupported type comparison: {0}")]
    UnsupportedTypeComparison(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ident_add_ident_str() -> anyhow::Result<()> {
        let src = r#"{"field1":"Dean","field2":"Karn"}"#;
        let expression = r#".field1 + " " + .field2"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::String("Dean Karn".to_string()), result);
        Ok(())
    }

    #[test]
    fn ident_add_ident_num() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field1 + .field2"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.33), result);
        Ok(())
    }

    #[test]
    fn ident_sub_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field2 - .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(13.13), result);
        Ok(())
    }

    #[test]
    fn ident_mult_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":3}"#;
        let expression = r#".field2 * .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.3), result);
        Ok(())
    }

    #[test]
    fn ident_div_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#".field2 / .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(11.1), result);
        Ok(())
    }

    #[test]
    fn num_add_num() -> anyhow::Result<()> {
        let src = "";
        let expression = r#"11.1 + 22.2"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.3), result);
        Ok(())
    }

    #[test]
    fn ident_add_num() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 + .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(14.1), result);
        Ok(())
    }

    #[test]
    fn ident_eq_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 == .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn ident_eq_num_true() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 == .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn ident_gt_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 > .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn ident_gte_num_true() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 >= .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn bool_true() -> anyhow::Result<()> {
        let ex = Parser::parse("true == true")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn bool_false() -> anyhow::Result<()> {
        let ex = Parser::parse("false == true")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn null_eq() -> anyhow::Result<()> {
        let ex = Parser::parse("NULL == NULL")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn or() -> anyhow::Result<()> {
        let ex = Parser::parse("true || false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false || true")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false || false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn and() -> anyhow::Result<()> {
        let ex = Parser::parse("true && true")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false && false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse("true && false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse("false && true")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn contains() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" CONTAINS "i""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" CONTAINS "ea""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn starts_with() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" STARTSWITH "i""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" STARTSWITH "te""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn ends_with() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" ENDSWITH "i""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" ENDSWITH "am""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn inn() -> anyhow::Result<()> {
        let src = r#"{"field1":["test"]}"#.as_bytes();
        let expression = r#""test" IN .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"field1":["test"]}"#.as_bytes();
        let expression = r#""me" IN .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn inn_arr() -> anyhow::Result<()> {
        let expression = r#""me" IN ["me"]"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#""me" IN ["z"]"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#""me" IN []"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#"[] == []"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"[] == ["test"]"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }

    #[test]
    fn ampersand() -> anyhow::Result<()> {
        let expression = "(1 + 1) / 2";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Number(1.0), result);

        let expression = "1 + 1 / 2";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Number(1.5), result);
        Ok(())
    }

    #[test]
    fn company_employees() -> anyhow::Result<()> {
        let src = r#"{"name":"Company","properties":{"employees":50}}"#.as_bytes();
        let expression = ".properties.employees > 20";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = ".properties.employees > 50";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn company_not_employees() -> anyhow::Result<()> {
        let src = r#"{"name":"Company","properties":{"employees":50}}"#.as_bytes();
        let expression = ".properties.employees !> 20";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let expression = ".properties.employees !> 50";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = ".properties.employees != 50";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn company_not() -> anyhow::Result<()> {
        let src = r#"{"f1":true,"f2":false}"#.as_bytes();
        let expression = "!.f1";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src = r#"{"f1":true,"f2":false}"#.as_bytes();
        let expression = "!.f2";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"f1":true,"f2":false}"#.as_bytes();
        let expression = "!(.f1 && .f2)";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"f1":true,"f2":false}"#.as_bytes();
        let expression = "!(.f1 != .f2)";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src = r#"{"f1":true,"f2":false}"#.as_bytes();
        let expression = "!(.f1 != .f2) && !.f2";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }
}
