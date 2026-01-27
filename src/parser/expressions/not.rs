use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Not {
    pub value: BoxedExpression,
}

impl Expression for Not {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let v = self.value.calculate(json)?;
        match v {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            v => Err(Error::UnsupportedTypeComparison(format!("{v:?} for !"))),
        }
    }
}
