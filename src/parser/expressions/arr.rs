use crate::parser::parse::BoxedExpression;
use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Arr {
    pub arr: Vec<BoxedExpression>,
}

impl Expression for Arr {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let mut arr = Vec::new();
        for e in &self.arr {
            arr.push(e.calculate(json)?);
        }
        Ok(Value::Array(arr))
    }
}
