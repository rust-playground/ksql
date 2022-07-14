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

use crate::lexer::{TokenKind, Tokenizer};
use anyhow::anyhow;
use chrono::{DateTime, FixedOffset, SecondsFormat};
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
    DateTime(DateTime<FixedOffset>), // What to put here arg! do we preserve the original zone etc..?
    Object(BTreeMap<String, Value>),
    Array(Vec<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => f.write_str("null"),
            Value::String(s) => {
                f.write_str(r#"""#)?;
                f.write_str(s)?;
                f.write_str(r#"""#)
            }
            Value::DateTime(dt) => {
                f.write_str(r#"""#)?;
                f.write_str(&dt.to_rfc3339_opts(SecondsFormat::Nanos, true))?;
                f.write_str(r#"""#)
            }
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => {
                if *b {
                    f.write_str("true")
                } else {
                    f.write_str("false")
                }
            }
            Value::Object(o) => {
                f.write_str("{")?;

                let len = o.len() - 1;
                for (i, (k, v)) in o.iter().enumerate() {
                    f.write_str(r#"""#)?;
                    f.write_str(k)?;
                    f.write_str(r#"":"#)?;
                    write!(f, "{}", v)?;
                    if i < len {
                        f.write_str(",")?;
                    }
                }
                f.write_str("}")
            }
            Value::Array(a) => {
                f.write_str("[")?;

                let len = a.len() - 1;
                for (i, v) in a.iter().enumerate() {
                    write!(f, "{}", v)?;
                    if i < len {
                        f.write_str(",")?;
                    }
                }
                f.write_str("]")
            }
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
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    fn new(exp: &'a [u8], tokenizer: Tokenizer<'a>) -> Self {
        Parser { exp, tokenizer }
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
        let tokenizer = Tokenizer::new_bytes(expression);
        let mut parser = Parser::new(expression, tokenizer);
        let result = parser.parse_value(false)?;

        if let Some(result) = result {
            Ok(result)
        } else {
            Err(anyhow!("no expression results found"))
        }
    }

    fn parse_value(&mut self, return_value_now: bool) -> anyhow::Result<Option<BoxedExpression>> {
        if let Some(tok) = self.tokenizer.next() {
            let tok = tok?;
            dbg!(&tok.kind);
            match tok.kind {
                TokenKind::SelectorPath => {
                    let start = tok.start as usize;
                    self.parse_op(Box::new(SelectorPath {
                        ident: String::from_utf8_lossy(
                            &self.exp[start + 1..(start + tok.len as usize)],
                        )
                        .into_owned(),
                    }))
                }
                TokenKind::QuotedString => {
                    let start = tok.start as usize;
                    // TODO: make tokenizer peekable, peek here to see if next is a cast
                    // to build a constant Value instead fo casting each time
                    self.parse_op(Box::new(Str {
                        s: String::from_utf8_lossy(
                            &self.exp[start + 1..(start + tok.len as usize - 1)],
                        )
                        .into_owned(),
                    }))
                }
                TokenKind::Number => {
                    let start = tok.start as usize;
                    self.parse_op(Box::new(Num {
                        n: String::from_utf8_lossy(&self.exp[start..start + tok.len as usize])
                            .parse()?,
                    }))
                }
                TokenKind::BooleanTrue => self.parse_op(Box::new(Bool { b: true })),
                TokenKind::BooleanFalse => self.parse_op(Box::new(Bool { b: false })),
                TokenKind::Null => self.parse_op(Box::new(Null {})),
                TokenKind::Not => {
                    let v = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no identifier after !")), Ok)?;
                    self.parse_op(Box::new(Not { value: v }))
                }
                TokenKind::OpenBracket => {
                    let mut arr = Vec::new();

                    while let Some(v) = self.parse_value()? {
                        arr.push(v);
                    }
                    let arr = Arr { arr };
                    self.parse_op(Box::new(arr))
                }
                TokenKind::Comma => match self.parse_value()? {
                    Some(v) => Ok(Some(v)),
                    None => Err(anyhow!("value required after comma: {:?}", tok)),
                },
                TokenKind::OpenParen => {
                    let op = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value between ()")), Ok)?;
                    self.parse_op(op)
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
        &mut self,
        return_value_now: bool,
        value: BoxedExpression,
    ) -> anyhow::Result<Option<BoxedExpression>> {
        if let Some(tok) = self.tokenizer.next() {
            let tok = tok?;
            dbg!(&tok.kind);
            match tok.kind {
                TokenKind::In => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after IN")), Ok)?;
                    Ok(Some(Box::new(In { left: value, right })))
                }
                TokenKind::Contains => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after CONTAINS")), Ok)?;
                    Ok(Some(Box::new(Contains { left: value, right })))
                }
                TokenKind::StartsWith => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after STARTSWITH")), Ok)?;
                    Ok(Some(Box::new(StartsWith { left: value, right })))
                }
                TokenKind::EndsWith => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after ENDSWITH")), Ok)?;
                    Ok(Some(Box::new(EndsWith { left: value, right })))
                }
                TokenKind::And => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after AND")), Ok)?;
                    Ok(Some(Box::new(And { left: value, right })))
                }
                TokenKind::Or => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after OR")), Ok)?;
                    Ok(Some(Box::new(Or { left: value, right })))
                }
                TokenKind::Gt => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after >")), Ok)?;
                    Ok(Some(Box::new(Gt { left: value, right })))
                }
                TokenKind::Gte => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after >=")), Ok)?;
                    Ok(Some(Box::new(Gte { left: value, right })))
                }
                TokenKind::Lt => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after <")), Ok)?;
                    Ok(Some(Box::new(Lt { left: value, right })))
                }
                TokenKind::Lte => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after <=")), Ok)?;
                    Ok(Some(Box::new(Lte { left: value, right })))
                }
                TokenKind::Equals => {
                    // yes need to parse RHS of equals
                    // once a VALUE is parsed though, without any additional nesting, the RHS
                    // parsing must stop! and then after creating Eq call parse_op again.
                    //
                    // We need to pass down a bool, or something, to know when it should return a
                    // parse value, and when it's ok to continue parsing down the nested chain.
                    //
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after ==")), Ok)?;
                    Ok(Some(Box::new(Eq { left: value, right })))
                }
                TokenKind::Add => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after +")), Ok)?;
                    Ok(Some(Box::new(Add { left: value, right })))
                }
                TokenKind::Subtract => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after -")), Ok)?;
                    Ok(Some(Box::new(Sub { left: value, right })))
                }
                TokenKind::Multiply => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after *")), Ok)?;
                    Ok(Some(Box::new(Mult { left: value, right })))
                }
                TokenKind::Divide => {
                    let right = self
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value after /")), Ok)?;
                    Ok(Some(Box::new(Div { left: value, right })))
                }
                TokenKind::Not => {
                    let op = self
                        .parse_op(value)
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
                        .parse_value()?
                        .map_or_else(|| Err(anyhow!("no value between ()")), Ok)?;
                    self.parse_op(op)
                }
                TokenKind::Cast => {
                    // special case, CAST MUST be followed by an Identifier that matches a static
                    // pre-defined list of supported cast types.

                    // TODO: call parse_op with CastDateTime instead of returning?
                    let op = match self.tokenizer.next() {
                        Some(Ok(tok)) if tok.kind == TokenKind::Identifier => {
                            let start = tok.start as usize;
                            let ident =
                                String::from_utf8_lossy(&self.exp[start..start + tok.len as usize]);
                            match ident.as_ref() {
                                "datetime" => Box::new(CastDateTime { value }),
                                _ => return Err(anyhow!("invalid CAST data type '{:?}'", &ident)),
                            }
                        }
                        _ => {
                            return Err(anyhow!(
                                "invalid token type after CAST '{:?}'",
                                String::from_utf8_lossy(&self.exp[tok.start as usize..])
                            ))
                        }
                    };
                    self.parse_op(op)
                }
                TokenKind::CloseBracket | TokenKind::CloseParen | TokenKind::Comma => {
                    Ok(Some(value))
                }
                _ => {
                    let start = tok.start as usize;
                    Err(anyhow!(
                        "invalid token after selector path '{:?}'",
                        String::from_utf8_lossy(&self.exp[start..=start + tok.len as usize])
                    ))
                }
            }
        } else {
            Ok(Some(value))
        }
    }
}

#[derive(Debug)]
struct CastDateTime {
    value: BoxedExpression,
}

impl Expression for CastDateTime {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let value = self.value.calculate(json)?;

        match value {
            // TODO: Add more variants
            Value::String(ref s) => match anydate::parse(s) {
                Err(_) => Ok(Value::Null),
                Ok(dt) => Ok(Value::DateTime(dt)),
            },
            value => Err(Error::UnsupportedCast(format!("{:?} CAST datetime", value))),
        }
    }
}

#[derive(Debug)]
struct Add {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Add {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        Ok(Value::Bool(left == right))
    }
}

#[derive(Debug)]
struct Gt {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Gt {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let v = self.value.calculate(json)?;
        match v {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            v => Err(Error::UnsupportedTypeComparison(format!("{:?} for !", v))),
        }
    }
}

#[derive(Debug)]
struct SelectorPath {
    ident: String,
}

impl Expression for SelectorPath {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        Ok(unsafe { gjson::get_bytes(json, &self.ident).into() })
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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

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
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let mut arr = Vec::new();
        for e in &self.arr {
            arr.push(e.calculate(json)?);
        }
        Ok(Value::Array(arr))
    }
}

/// Result type for the `parse` function.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for the expression parser.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("unsupported type comparison: {0}")]
    UnsupportedTypeComparison(String),

    #[error("unsupported cast: {0}")]
    UnsupportedCast(String),
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

        let ex = Parser::parse("false || false || false == false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false || false || false != false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }

    #[test]
    fn and() -> anyhow::Result<()> {
        let ex = Parser::parse("true == true && false == false")?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

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
        let expression = r#""me" IN ["you","me"]"#;
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

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::Null), "null");
        assert_eq!(
            format!("{}", Value::String("string".to_string())),
            r#""string""#
        );
        assert_eq!(format!("{}", Value::Number(64.1)), "64.1");
        assert_eq!(format!("{}", Value::Bool(true)), "true");

        let mut m = BTreeMap::new();
        m.insert("key".to_string(), Value::String("value".to_string()));
        m.insert("key2".to_string(), Value::String("value2".to_string()));

        assert_eq!(
            format!("{}", Value::Object(m)),
            r#"{"key":"value","key2":"value2"}"#
        );
        assert_eq!(
            format!(
                "{}",
                Value::Array(vec![
                    Value::String("string".to_string()),
                    Value::Number(1.1)
                ])
            ),
            r#"["string",1.1]"#
        );
    }

    #[test]
    fn cast_datetime() -> anyhow::Result<()> {
        let src = r#"{"name":"2022-01-02"}"#.as_bytes();
        let expression = ".name CAST datetime";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(r#""2022-01-02T00:00:00.000000000Z""#, format!("{}", result));

        let src = r#"{"dt1":"2022-01-02","dt2":"2022-01-02"}"#.as_bytes();
        let expression = ".dt1 CAST datetime == .dt2 CAST datetime";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src =
            r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
                .as_bytes();
        let expression = "(.dt1 CAST datetime == .dt2 CAST datetime) && true == true";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        // let src =
        //     r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
        //         .as_bytes();
        // let expression = ".dt1 CAST datetime == .dt2 CAST datetime && true == true";
        // let ex = Parser::parse(expression)?;
        // let result = ex.calculate(src)?;
        // assert_eq!(Value::Bool(false), result);
        Ok(())
    }
}
