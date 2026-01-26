use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct ContainsAny {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for ContainsAny {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        match (left, right) {
            (Value::String(s1), Value::String(s2)) => {
                // Use String::contains directly - no allocation needed
                Ok(Value::Bool(s2.chars().any(|c| s1.contains(c))))
            }
            (Value::Array(arr1), Value::Array(arr2)) => {
                Ok(Value::Bool(arr2.iter().any(|v| arr1.contains(v))))
            }
            (Value::Array(arr), Value::String(s)) => {
                // Early return: empty cases
                if arr.is_empty() {
                    return Ok(Value::Bool(false));
                }

                // Check each char without allocating Value instances
                Ok(Value::Bool(s.chars().any(|c| {
                    let char_str = c.to_string();
                    arr.iter().any(|v| {
                        if let Value::String(s) = v {
                            s == &char_str
                        } else {
                            false
                        }
                    })
                })))
            }
            (Value::String(s), Value::Array(arr)) => Ok(Value::Bool(arr.iter().any(|v| match v {
                Value::String(s2) => s.contains(s2),
                _ => false,
            }))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{l} CONTAINS_ANY {r}",
            ))),
        }
    }
}
