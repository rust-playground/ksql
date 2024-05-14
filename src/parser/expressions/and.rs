use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct And {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for And {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;

        if let Value::Bool(is_true) = left {
            if !is_true {
                return Ok(left);
            }
        }

        let right = self.right.calculate(json)?;

        match (left, right) {
            (Value::Bool(b1), Value::Bool(b2)) => Ok(Value::Bool(b1 && b2)),
            (l, r) => Err(Error::UnsupportedTypeComparison(format!("{l} && {r}",))),
        }
    }
}
