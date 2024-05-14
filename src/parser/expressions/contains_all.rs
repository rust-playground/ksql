use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct ContainsAll {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
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
                    .all(|v| arr.contains(&Value::String(v.to_string()))),
            )),
            (Value::String(s), Value::Array(arr)) => Ok(Value::Bool(arr.iter().all(|v| match v {
                Value::String(s2) => s.contains(s2),
                _ => false,
            }))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{l} CONTAINS_ALL {r}",
            ))),
        }
    }
}
