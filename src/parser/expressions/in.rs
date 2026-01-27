use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct In {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for In {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;

        match (left, right) {
            (v, Value::Array(a)) => Ok(Value::Bool(a.contains(&v))),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!("{l} + {r}",))),
        }
    }
}
