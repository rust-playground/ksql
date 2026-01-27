use crate::parser::parse::BoxedExpression;
use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Eq {
    pub left: BoxedExpression,
    pub right: BoxedExpression,
}

impl Expression for Eq {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let left = self.left.calculate(json)?;
        let right = self.right.calculate(json)?;
        Ok(Value::Bool(left == right))
    }
}
