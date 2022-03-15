//! Parser is used to parse an expression for use against JSON data.

use crate::lexer::{Token, Tokenizer};
use anyhow::anyhow;
use gjson::Kind;
use std::collections::BTreeMap;
use std::fmt::{Debug, Display, Formatter};
use thiserror::Error;

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
                let arr = v.array().into_iter().map(|v| v.into()).collect();
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

pub trait Expression: Debug {
    fn calculate(&self, src: &str) -> Result<Value>;
}

type BoxedExpression = Box<dyn Expression>;

pub struct Parser;

impl Parser {
    pub fn parse(expression: &[u8]) -> anyhow::Result<BoxedExpression> {
        let tokens = Tokenizer::tokenize(expression)?;
        let mut pos = 0;
        let result = parse_value(&tokens, &mut pos)?;

        if let Some(result) = result {
            Ok(result)
        } else {
            Err(anyhow!("no expression results found"))
        }
    }
}
fn parse_value(tokens: &[Token], pos: &mut usize) -> anyhow::Result<Option<BoxedExpression>> {
    let tok = tokens.get(*pos);
    *pos += 1;
    match tok {
        Some(Token::Identifier(s)) => parse_op(Box::new(Ident { ident: s.clone() }), tokens, pos),
        Some(Token::String(s)) => parse_op(Box::new(Str { s: s.clone() }), tokens, pos),
        Some(Token::Number(n)) => parse_op(Box::new(Num { n: *n }), tokens, pos),
        Some(Token::Boolean(b)) => parse_op(Box::new(Bool { b: *b }), tokens, pos),
        Some(Token::Null) => parse_op(Box::new(Null {}), tokens, pos),
        Some(Token::OpenBracket) => {
            let mut arr = Vec::new();

            while let Some(v) = parse_value(tokens, pos)? {
                arr.push(v);
            }
            let arr = Arr { arr };
            parse_op(Box::new(arr), tokens, pos)
        }
        Some(Token::Comma) => match parse_value(tokens, pos)? {
            Some(v) => Ok(Some(v)),
            None => Err(anyhow!(" value required after comma: {:?}", tok)),
        },
        Some(Token::CloseBracket) => Ok(None),
        Some(Token::OpenParen) => {
            let op =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value between (")), Ok)?;

            // let operation = Operation { op };
            parse_op(op, tokens, pos)
        }
        Some(Token::CloseParen) => Err(anyhow!("no value between (")),
        None => Ok(None),
        _ => Err(anyhow!("invalid value: {:?}", tok)),
    }
}

fn parse_op(
    value: BoxedExpression,
    tokens: &[Token],
    pos: &mut usize,
) -> anyhow::Result<Option<BoxedExpression>> {
    let tok = tokens.get(*pos);
    *pos += 1;
    match tok {
        Some(Token::In) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after IN")), Ok)?;

            let inn = In { left: value, right };
            Ok(Some(Box::new(inn)))
        }
        Some(Token::Contains) => {
            let right = parse_value(tokens, pos)?
                .map_or_else(|| Err(anyhow!("no value after CONTAINS")), Ok)?;

            let contains = Contains { left: value, right };
            Ok(Some(Box::new(contains)))
        }
        Some(Token::StartsWith) => {
            let right = parse_value(tokens, pos)?
                .map_or_else(|| Err(anyhow!("no value after STARTSWITH")), Ok)?;

            let starts_with = StartsWith { left: value, right };
            Ok(Some(Box::new(starts_with)))
        }
        Some(Token::EndsWith) => {
            let right = parse_value(tokens, pos)?
                .map_or_else(|| Err(anyhow!("no value after ENDSWITH")), Ok)?;

            let ends_with = EndsWith { left: value, right };
            Ok(Some(Box::new(ends_with)))
        }
        Some(Token::And) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after AND")), Ok)?;

            let and = And { left: value, right };
            Ok(Some(Box::new(and)))
        }
        Some(Token::Or) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after OR")), Ok)?;

            let or = Or { left: value, right };
            Ok(Some(Box::new(or)))
        }
        Some(Token::Gt) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after >")), Ok)?;

            let gt = Gt { left: value, right };
            Ok(Some(Box::new(gt)))
        }
        Some(Token::Gte) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after >=")), Ok)?;

            let gte = Gte { left: value, right };
            Ok(Some(Box::new(gte)))
        }
        Some(Token::Lt) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after <")), Ok)?;

            let lt = Lt { left: value, right };
            Ok(Some(Box::new(lt)))
        }
        Some(Token::Lte) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after <=")), Ok)?;

            let lte = Lte { left: value, right };
            Ok(Some(Box::new(lte)))
        }
        Some(Token::Equals) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after =")), Ok)?;

            let eq = Eq { left: value, right };
            Ok(Some(Box::new(eq)))
        }
        Some(Token::Add) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after +")), Ok)?;

            let add = Add { left: value, right };
            Ok(Some(Box::new(add)))
        }
        Some(Token::Subtract) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after -")), Ok)?;

            let sub = Sub { left: value, right };
            Ok(Some(Box::new(sub)))
        }
        Some(Token::Multiply) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after *")), Ok)?;

            let mult = Mult { left: value, right };
            Ok(Some(Box::new(mult)))
        }
        Some(Token::Divide) => {
            let right =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value after /")), Ok)?;

            let div = Div { left: value, right };
            Ok(Some(Box::new(div)))
        }
        Some(Token::CloseBracket) => Ok(Some(value)),
        Some(Token::OpenParen) => {
            let op =
                parse_value(tokens, pos)?.map_or_else(|| Err(anyhow!("no value between (")), Ok)?;

            // let operation = Operation { op };
            parse_op(op, tokens, pos)
        }
        Some(Token::CloseParen) => Ok(Some(value)),
        None => Ok(Some(value)),
        _ => Err(anyhow!("invalid token after ident '{:?}'", tok.unwrap())),
    }
}

#[derive(Debug)]
struct Add {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Add {
    fn calculate(&self, src: &str) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::String(s1), Value::String(ref s2)) => Ok(Value::String(s1 + s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
struct Ident {
    ident: String,
}

impl Expression for Ident {
    fn calculate(&self, src: &str) -> Result<Value> {
        Ok(gjson::get(src, &self.ident).into())
    }
}

#[derive(Debug)]
struct Str {
    s: String,
}

impl Expression for Str {
    fn calculate(&self, _: &str) -> Result<Value> {
        Ok(Value::String(self.s.clone()))
    }
}

#[derive(Debug)]
struct Num {
    n: f64,
}

impl Expression for Num {
    fn calculate(&self, _: &str) -> Result<Value> {
        Ok(Value::Number(self.n))
    }
}

#[derive(Debug)]
struct Bool {
    b: bool,
}

impl Expression for Bool {
    fn calculate(&self, _: &str) -> Result<Value> {
        Ok(Value::Bool(self.b))
    }
}

#[derive(Debug)]
struct Null;

impl Expression for Null {
    fn calculate(&self, _: &str) -> Result<Value> {
        Ok(Value::Null)
    }
}

#[derive(Debug)]
struct Or {
    left: BoxedExpression,
    right: BoxedExpression,
}

impl Expression for Or {
    fn calculate(&self, src: &str) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 || b2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} OR {:?}",
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
    fn calculate(&self, src: &str) -> Result<Value> {
        let left = self.left.calculate(src)?;
        let right = self.right.calculate(src)?;

        match (left, right) {
            (Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 && b2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{:?} AND {:?}",
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
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
    fn calculate(&self, src: &str) -> Result<Value> {
        let mut arr = Vec::new();
        for e in self.arr.iter() {
            arr.push(e.calculate(src)?);
        }
        Ok(Value::Array(arr))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

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

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::String("Dean Karn".to_string()), result);
        Ok(())
    }

    #[test]
    fn ident_add_ident_num() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field1 + .field2"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.33), result);
        Ok(())
    }

    #[test]
    fn ident_sub_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":10.1,"field2":23.23}"#;
        let expression = r#".field2 - .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(13.13), result);
        Ok(())
    }

    #[test]
    fn ident_mult_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":3}"#;
        let expression = r#".field2 * .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.3), result);
        Ok(())
    }

    #[test]
    fn ident_div_ident() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#".field2 / .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(11.1), result);
        Ok(())
    }

    #[test]
    fn num_add_num() -> anyhow::Result<()> {
        let src = "";
        let expression = r#"11.1 + 22.2"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(33.3), result);
        Ok(())
    }

    #[test]
    fn ident_add_num() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 + .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Number(14.1), result);
        Ok(())
    }

    #[test]
    fn ident_eq_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":3,"field2":33.3}"#;
        let expression = r#"11.1 = .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn ident_eq_num_true() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 = .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn ident_gt_num_false() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 > .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn ident_gte_num_true() -> anyhow::Result<()> {
        let src = r#"{"field1":11.1,"field2":33.3}"#;
        let expression = r#"11.1 >= .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src.as_ref())?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn bool_true() -> anyhow::Result<()> {
        let ex = Parser::parse("true = true".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn bool_false() -> anyhow::Result<()> {
        let ex = Parser::parse("false = true".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn null_eq() -> anyhow::Result<()> {
        let ex = Parser::parse("NULL = NULL".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn or() -> anyhow::Result<()> {
        let ex = Parser::parse("true OR false".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false OR true".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false OR false".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn and() -> anyhow::Result<()> {
        let ex = Parser::parse("true AND true".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);

        let ex = Parser::parse("false AND false".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse("true AND false".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse("false AND true".as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn contains() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" CONTAINS "i""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" CONTAINS "ea""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn starts_with() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" STARTSWITH "i""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" STARTSWITH "te""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn ends_with() -> anyhow::Result<()> {
        let ex = Parser::parse(r#""team" ENDSWITH "i""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let ex = Parser::parse(r#""team" ENDSWITH "am""#.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);
        Ok(())
    }

    #[test]
    fn inn() -> anyhow::Result<()> {
        let src = r#"{"field1":["test"]}"#;
        let expression = r#""test" IN .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let src = r#"{"field1":["test"]}"#;
        let expression = r#""me" IN .field1"#;

        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }

    #[test]
    fn inn_arr() -> anyhow::Result<()> {
        let expression = r#""me" IN ["me"]"#;
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#""me" IN ["z"]"#;
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#""me" IN []"#;
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        let expression = r#"[] = []"#;
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(true), result);

        let expression = r#"[] = ["test"]"#;
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Bool(false), result);

        Ok(())
    }

    #[test]
    fn ampersand() -> anyhow::Result<()> {
        let expression = "(1 + 1) / 2";
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Number(1.0), result);

        let expression = "1 + 1 / 2";
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate("")?;
        assert_eq!(Value::Number(1.5), result);
        Ok(())
    }

    #[test]
    fn company_employees() -> anyhow::Result<()> {
        let src = r#"{"name":"Company","properties":{"employees":50}}"#;
        let expression = ".properties.employees > 20";
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(true), result);

        let expression = ".properties.employees > 50";
        let ex = Parser::parse(expression.as_bytes())?;
        let result = ex.calculate(src)?;
        assert_eq!(Value::Bool(false), result);
        Ok(())
    }
}
