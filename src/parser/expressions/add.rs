use crate::parser::{Error, Expression, Result, Value};
use crate::parser::parse::BoxedExpression;

#[derive(Debug)]
pub(in crate::parser) struct Add {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
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
            (l, r) => Err(Error::UnsupportedTypeComparison(format!("{l} + {r}",))),
        }
    }
}
