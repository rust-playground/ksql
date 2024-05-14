use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Contains {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for Contains {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1.contains(&s2))),
            (Value::Array(arr1), v) => Ok(Value::Bool(arr1.contains(&v))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!(
                "{l} CONTAINS {r}",
            ))),
        }
    }
}
