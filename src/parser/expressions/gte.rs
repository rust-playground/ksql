use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Gte {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for Gte {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1 >= s2)),
            (Value::Number(n1), Value::Number(n2)) => Ok(Value::Bool(n1 >= n2)),
            (Value::DateTime(dt1), Value::DateTime(dt2)) => Ok(Value::Bool(dt1 >= dt2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!("{l} >= {r}",))),
        }
    }
}
