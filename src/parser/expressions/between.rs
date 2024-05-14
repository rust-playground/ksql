use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Between {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
    pub value: BoxedExpression,
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
                "{v} BETWEEN {lhs} {rhs}",
            ))),
        }
    }
}
