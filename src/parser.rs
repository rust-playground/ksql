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
use chrono::{DateTime, Utc};
use gjson::Kind;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::{Debug, Display, Formatter};
use thiserror::Error;

/// Represents the calculated Expression result.
#[derive(Debug, PartialEq, Clone, Serialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    String(String),
    Number(f64),
    Bool(bool),
    DateTime(DateTime<Utc>), // What to put here arg! do we preserve the original zone etc..?
    Object(BTreeMap<String, Value>),
    Array(Vec<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use serde::ser::Error;

        match serde_json::to_string(self) {
            Ok(s) => {
                f.write_str(&s)?;
                Ok(())
            }
            Err(e) => Err(std::fmt::Error::custom(e)),
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
        let result = parser.parse_expression()?;

        if let Some(result) = result {
            Ok(result)
        } else {
            Err(anyhow!("no expression results found"))
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_expression(&mut self) -> anyhow::Result<Option<BoxedExpression>> {
        let mut current: Option<BoxedExpression> = None;

        loop {
            if let Some(token) = self.tokenizer.next() {
                let token = token?;
                if let Some(expression) = current {
                    // CloseParen is the end of an expression block, return parsed expression.
                    if token.kind == TokenKind::CloseParen {
                        return Ok(Some(expression));
                    }
                    // look for next operation
                    current = self.parse_operation(token, expression)?;
                } else {
                    // look for next value
                    current = Some(self.parse_value(token)?);
                }
            } else {
                return Ok(current);
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_value(&mut self, token: Token) -> anyhow::Result<BoxedExpression> {
        match token.kind {
            TokenKind::OpenBracket => {
                let mut arr = Vec::new();

                loop {
                    if let Some(token) = self.tokenizer.next() {
                        let token = token?;

                        match token.kind {
                            TokenKind::CloseBracket => {
                                break;
                            }
                            TokenKind::Comma => continue, // optional for defining arrays
                            _ => {
                                arr.push(self.parse_value(token)?);
                            }
                        };
                    } else {
                        return Err(anyhow!("unclosed Array '['"));
                    }
                }
                Ok(Box::new(Arr { arr }))
            }
            TokenKind::OpenParen => {
                if let Some(expression) = self.parse_expression()? {
                    Ok(expression)
                } else {
                    Err(anyhow!(
                        "expression after open parenthesis '(' ends unexpectedly."
                    ))
                }
            }
            TokenKind::SelectorPath => {
                let start = token.start as usize;
                Ok(Box::new(SelectorPath {
                    ident: String::from_utf8_lossy(
                        &self.exp[start + 1..(start + token.len as usize)],
                    )
                    .into_owned(),
                }))
            }
            TokenKind::QuotedString => {
                let start = token.start as usize;
                Ok(Box::new(Str {
                    s: String::from_utf8_lossy(
                        &self.exp[start + 1..(start + token.len as usize - 1)],
                    )
                    .into_owned(),
                }))
            }
            TokenKind::Number => {
                let start = token.start as usize;
                Ok(Box::new(Num {
                    n: String::from_utf8_lossy(&self.exp[start..start + token.len as usize])
                        .parse()?,
                }))
            }
            TokenKind::BooleanTrue => Ok(Box::new(Bool { b: true })),
            TokenKind::BooleanFalse => Ok(Box::new(Bool { b: false })),
            TokenKind::Null => Ok(Box::new(Null {})),
            TokenKind::Coerce => {
                // COERCE <expression> _<datatype>_
                let next_token = self.next_operator_token(token)?;
                let const_eligible = matches!(
                    next_token.kind,
                    TokenKind::QuotedString
                        | TokenKind::Number
                        | TokenKind::BooleanFalse
                        | TokenKind::BooleanTrue
                        | TokenKind::Null
                );
                let value = self.parse_value(next_token)?;
                if let Some(token) = self.tokenizer.next() {
                    let token = token?;
                    let start = token.start as usize;

                    if token.kind == TokenKind::Identifier {
                        let ident =
                            String::from_utf8_lossy(&self.exp[start..start + token.len as usize]);
                        match ident.as_ref() {
                            "_datetime_" => {
                                let expression = COERCEDateTime { value };
                                if const_eligible {
                                    Ok(Box::new(CoercedConst {
                                        value: expression.calculate(&[])?,
                                    }))
                                } else {
                                    Ok(Box::new(expression))
                                }
                            }
                            _ => Err(anyhow!("invalid COERCE data type '{:?}'", &ident)),
                        }
                    } else {
                        Err(anyhow!(
                            "COERCE missing data type identifier, found instead: {:?}",
                            &self.exp[start..(start + token.len as usize)]
                        ))
                    }
                } else {
                    Err(anyhow!("no identifier after value for: COERCE"))
                }
            }
            TokenKind::Not => {
                let next_token = self.next_operator_token(token)?;
                let value = self.parse_value(next_token)?;
                Ok(Box::new(Not { value }))
            }
            _ => Err(anyhow!("token is not a valid value: {:?}", token)),
        }
    }

    #[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
    fn next_operator_token(&mut self, operation_token: Token) -> anyhow::Result<Token> {
        if let Some(token) = self.tokenizer.next() {
            Ok(token?)
        } else {
            let start = operation_token.start as usize;
            Err(anyhow!(
                "no value found after operation: {:?}",
                &self.exp[start..(start + operation_token.len as usize)]
            ))
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_operation(
        &mut self,
        token: Token,
        current: BoxedExpression,
    ) -> anyhow::Result<Option<BoxedExpression>> {
        match token.kind {
            TokenKind::Add => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Add {
                    left: current,
                    right,
                })))
            }
            TokenKind::Subtract => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Sub {
                    left: current,
                    right,
                })))
            }
            TokenKind::Multiply => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Mult {
                    left: current,
                    right,
                })))
            }
            TokenKind::Divide => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Div {
                    left: current,
                    right,
                })))
            }
            TokenKind::Equals => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Eq {
                    left: current,
                    right,
                })))
            }
            TokenKind::Gt => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Gt {
                    left: current,
                    right,
                })))
            }
            TokenKind::Gte => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Gte {
                    left: current,
                    right,
                })))
            }
            TokenKind::Lt => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Lt {
                    left: current,
                    right,
                })))
            }
            TokenKind::Lte => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Lte {
                    left: current,
                    right,
                })))
            }
            TokenKind::Or => {
                let right = self
                    .parse_expression()?
                    .map_or_else(|| Err(anyhow!("invalid operation after ||")), Ok)?;
                Ok(Some(Box::new(Or {
                    left: current,
                    right,
                })))
            }
            TokenKind::And => {
                let right = self
                    .parse_expression()?
                    .map_or_else(|| Err(anyhow!("invalid operation after &&")), Ok)?;
                Ok(Some(Box::new(And {
                    left: current,
                    right,
                })))
            }
            TokenKind::StartsWith => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(StartsWith {
                    left: current,
                    right,
                })))
            }
            TokenKind::EndsWith => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(EndsWith {
                    left: current,
                    right,
                })))
            }
            TokenKind::In => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(In {
                    left: current,
                    right,
                })))
            }
            TokenKind::Contains => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(Contains {
                    left: current,
                    right,
                })))
            }
            TokenKind::ContainsAny => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(ContainsAny {
                    left: current,
                    right,
                })))
            }
            TokenKind::ContainsAll => {
                let next_token = self.next_operator_token(token)?;
                let right = self.parse_value(next_token)?;
                Ok(Some(Box::new(ContainsAll {
                    left: current,
                    right,
                })))
            }
            TokenKind::Between => {
                let lhs_token = self.next_operator_token(token.clone())?;
                let left = self.parse_value(lhs_token)?;
                let rhs_token = self.next_operator_token(token)?;
                let right = self.parse_value(rhs_token)?;
                Ok(Some(Box::new(Between {
                    left,
                    right,
                    value: current,
                })))
            }
            TokenKind::Not => {
                let next_token = self.next_operator_token(token)?;
                let value = self
                    .parse_operation(next_token, current)?
                    .map_or_else(|| Err(anyhow!("invalid operation after !")), Ok)?;
                Ok(Some(Box::new(Not { value })))
            }
            TokenKind::CloseBracket => Ok(Some(current)),
            _ => Err(anyhow!("invalid operation: {:?}", token)),
        }
    }
}

#[derive(Debug)]
struct Between {
    left: BoxedExpression,
    right: BoxedExpression,
    value: BoxedExpression,
}

impl Expression for Between {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        let value = self.value.calculate(json)?;

        match (value, left, right) {
            (Value::String(v), Value::String(lhs), Value::String(rhs)) => {
                Ok(Value::Bool(v > lhs && v < rhs))
            }
            (Value::Number(v), Value::Number(lhs), Value::Number(rhs)) => {
                Ok(Value::Bool(v > lhs && v < rhs))
            }
            (Value::DateTime(v), Value::DateTime(lhs), Value::DateTime(rhs)) => {
                Ok(Value::Bool(v > lhs && v < rhs))
            }
            (Value::Null, _, _) | (_, Value::Null, _) | (_, _, Value::Null) => {
                Ok(Value::Bool(false))
            }
            (v, lhs, rhs) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} BETWEEN {:?} {:?}",
                v, lhs, rhs
            ))),
        }
    }
}

#[derive(Debug)]
struct COERCEDateTime {
    value: BoxedExpression,
}

impl Expression for COERCEDateTime {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let value = self.value.calculate(json)?;

        match value {
            Value::String(ref s) => match anydate::parse_utc(s) {
                Err(_) => Ok(Value::Null),
                Ok(dt) => Ok(Value::DateTime(dt)),
            },
            Value::Null => Ok(value),
            value => Err(Error::UnsupportedCOERCE(format!(
                "{:?} COERCE datetime",
                value
            ))),
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
            (Value::DateTime(dt1), Value::DateTime(dt2)) => Ok(Value::Bool(dt1 > dt2)),
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
            (Value::DateTime(dt1), Value::DateTime(dt2)) => Ok(Value::Bool(dt1 >= dt2)),
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
            (Value::DateTime(dt1), Value::DateTime(dt2)) => Ok(Value::Bool(dt1 < dt2)),
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
            (Value::DateTime(dt1), Value::DateTime(dt2)) => Ok(Value::Bool(dt1 <= dt2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} <= {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct CoercedConst {
    value: Value,
}

impl Expression for CoercedConst {
    fn calculate(&self, _json: &[u8]) -> Result<Value> {
        Ok(self.value.clone())
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
            (Value::Array(arr1), v) => Ok(Value::Bool(arr1.contains(&v))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} CONTAINS {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct ContainsAny {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for ContainsAny {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        match (left, right) {
            (Value::String(s1), Value::String(s2)) => {
                let b1: Vec<char> = s1.chars().collect();
                // betting that lists are short and so less expensive than iterating one to create a hash set
                Ok(Value::Bool(s2.chars().any(|b| b1.contains(&b))))
            }
            (Value::Array(arr1), Value::Array(arr2)) => {
                Ok(Value::Bool(arr2.iter().any(|v| arr1.contains(v))))
            }
            (Value::Array(arr), Value::String(s)) => Ok(Value::Bool(
                s.chars()
                    .into_iter()
                    .any(|v| arr.contains(&Value::String(v.to_string()))),
            )),
            (Value::String(s), Value::Array(arr)) => Ok(Value::Bool(arr.iter().any(|v| match v {
                Value::String(s2) => s.contains(s2),
                _ => false,
            }))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} CONTAINS_ANY {:?}",
                l, r
            ))),
        }
    }
}

#[derive(Debug)]
struct ContainsAll {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for ContainsAll {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        match (left, right) {
            (Value::String(s1), Value::String(s2)) => {
                let b1: Vec<char> = s1.chars().collect();
                Ok(Value::Bool(s2.chars().all(|b| b1.contains(&b))))
            }
            (Value::Array(arr1), Value::Array(arr2)) => {
                Ok(Value::Bool(arr2.iter().all(|v| arr1.contains(v))))
            }
            (Value::Array(arr), Value::String(s)) => Ok(Value::Bool(
                s.chars()
                    .into_iter()
                    .all(|v| arr.contains(&Value::String(v.to_string()))),
            )),
            (Value::String(s), Value::Array(arr)) => Ok(Value::Bool(arr.iter().all(|v| match v {
                Value::String(s2) => s.contains(s2),
                _ => false,
            }))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} CONTAINS_ALL {:?}",
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

    #[error("unsupported COERCE: {0}")]
    UnsupportedCOERCE(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sp_add_str_sp() -> anyhow::Result<()> {
        let src = r#"{"field1":"Dean","field2":"Karn"}"#;
        let expression = r#".field1 + " " + .field2"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::String("Dean Karn".to_string()), result);
        Ok(())
    }

    #[test]
    fn sp_add_sp_num() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field1 + .field2"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.33), result);
        Ok(())
    }

    #[test]
    fn sp_sub_sp() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field2 - .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(13.13), result);
        Ok(())
    }

    #[test]
    fn sp_mult_identsp() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":3}"#;
        let expression = r#".field2 * .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.3), result);
        Ok(())
    }

    #[test]
    fn sp_div_sp() -> anyhow::Result<()> {
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
    fn sp_add_num() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 + .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(14.1), result);
        Ok(())
    }

    #[test]
    fn sp_eq_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 == .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn sp_eq_num_true() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 == .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn sp_gt_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 > .field1"#;

        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn sp_gte_num_true() -> anyhow::Result<()> {
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
        assert_eq!(Value::Bool(true), result);

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
    fn contains() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" CONTAINS "i""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" CONTAINS "ea""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["ea"] CONTAINS "ea""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["nope"] CONTAINS "ea""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#"["a",["b","a"]] CONTAINS ["b","a"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn contains_any() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" CONTAINS_ANY "im""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ANY "eac""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ANY "xyz""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ANY ["c","d","e"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ANY ["d","e","f"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#"["a","b","c"] !CONTAINS_ANY ["d","e","f"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"AnnualRevenue":"2000000","NumberOfEmployees":"201","FirstName":"scott"}"#
            .as_bytes();
        let ex =
            Parser::parse(r#".FirstName CONTAINS_ANY ["noah", "emily", "alexandra","scott"]"#)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"AnnualRevenue":"2000000","NumberOfEmployees":"201","FirstName":"scott"}"#
            .as_bytes();
        let ex = Parser::parse(r#".FirstName CONTAINS_ANY ["noah", "emily", "alexandra"]"#)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }

    #[test]
    fn contains_all() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" CONTAINS_ALL "meat""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ALL "cab""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ALL "xyz""#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ALL ["c","a","b"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] CONTAINS_ALL ["a","b"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse(r#"["a","b","c"] !CONTAINS_ALL ["a","b"]"#)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let src = r#"{"AnnualRevenue":"2000000","NumberOfEmployees":"201","FirstName":"scott"}"#
            .as_bytes();
        let ex = Parser::parse(r#".FirstName CONTAINS_ALL ["sc", "ot", "ott","cot"]"#)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"AnnualRevenue":"2000000","NumberOfEmployees":"201","FirstName":"scott"}"#
            .as_bytes();
        let ex = Parser::parse(r#".FirstName CONTAINS_ALL ["sc", "ot", "ott","b"]"#)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

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
        let expression = r#""test" !IN .field1"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#""test" !IN ["b","a","c"]"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
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

        let expression = "1 + (1 / 2)";
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
    fn coerce_datetime() -> anyhow::Result<()> {
        let src = r#"{"name":"2022-01-02"}"#.as_bytes();
        let expression = "COERCE .name _datetime_";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(r#""2022-01-02T00:00:00Z""#, format!("{}", result));

        let src = r#"{"dt1":"2022-01-02","dt2":"2022-01-02"}"#.as_bytes();
        let expression = "COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src =
            r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
                .as_bytes();
        let expression = "COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_ && true == true";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src =
            r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
                .as_bytes();
        let expression = "(COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_) && true == true";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src =
            r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
                .as_bytes();
        let expression = "(COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_) && true == true";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src =
            r#"{"dt1":"2022-07-14T17:50:08.318426000Z","dt2":"2022-07-14T17:50:08.318426001Z"}"#
                .as_bytes();
        let expression = "(COERCE .dt1 _datetime_) == (COERCE .dt2 _datetime_) && true == true";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let src = r#"{"dt1":"2022-07-14T17:50:08.318426000Z"}"#.as_bytes();
        let expression =
            r#"COERCE .dt1 _datetime_ == COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"dt1":"2022-07-14T17:50:08.318426000Z"}"#.as_bytes();
        let expression =
            r#"COERCE .dt1 _datetime_ == COERCE "2022-07-14T17:50:08.318426001Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426000Z" _datetime_ == COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426001Z" _datetime_ > COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426000Z" _datetime_ < COERCE "2022-07-14T17:50:08.318426001Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426000Z" _datetime_ >= COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426000Z" _datetime_ <= COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426001Z" _datetime_ >= COERCE "2022-07-14T17:50:08.318426000Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-07-14T17:50:08.318426000Z" _datetime_ <= COERCE "2022-07-14T17:50:08.318426001Z" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        Ok(())
    }

    #[test]
    fn between() -> anyhow::Result<()> {
        let expression = "1 BETWEEN 0 10";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = "0 BETWEEN 0 10";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = "10 BETWEEN 0 10";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = ".key BETWEEN 0 10";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = "0 BETWEEN .key 10";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = "10 BETWEEN 0 .key";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#""g" BETWEEN "a" "z""#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#""z" BETWEEN "a" "z""#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#"COERCE "2022-01-02" _datetime_ BETWEEN COERCE "2022-01-01" _datetime_ COERCE "2022-01-30" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"COERCE "2022-01-01" _datetime_ BETWEEN COERCE "2022-01-01" _datetime_ COERCE "2022-01-30" _datetime_"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }

    #[test]
    fn parse_exponent_number() -> anyhow::Result<()> {
        let expression = "1e3 == 1000";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = "-1e-3 == -0.001";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        let expression = "+1e-3 == 0.001";
        let ex = Parser::parse(expression)?;
        let result = ex.calculate("".as_bytes())?;
        assert_eq!(Value::Bool(true), result);

        Ok(())
    }

    #[test]
    fn parse_random_expressions() -> anyhow::Result<()> {
        let src = r#"{"AnnualRevenue":"2000000","NumberOfEmployees":"201","FirstName":"scott"}"#
            .as_bytes();
        let expression = r#".NumberOfEmployees > "200" && .AnnualRevenue == "2000000""#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#".AnnualRevenue >= "5000000" || (.NumberOfEmployees > "200" && .AnnualRevenue == "2000000")"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#".AnnualRevenue >= "5000000" || (true && .AnnualRevenue == "2000000")"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#".AnnualRevenue >= "5000000" || (.NumberOfEmployees > "200" && true)"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"true || (.NumberOfEmployees > "200" && .AnnualRevenue == "2000000")"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"false || (.NumberOfEmployees > "200" && .AnnualRevenue == "2000000")"#;
        let ex = Parser::parse(expression)?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        Ok(())
    }
}
