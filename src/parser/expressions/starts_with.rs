use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct StartsWith {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for StartsWith {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

        match (left, right) {
            (Value::String(s1), Value::String(s2)) => Ok(Value::Bool(s1.starts_with(&s2))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!("{l} + {r}",))),
        }
    }
}
