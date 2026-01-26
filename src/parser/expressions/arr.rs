use crate::parser::parse::BoxedExpression;
use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Arr {
    pub arr: Vec<BoxedExpression>,
}

impl Expression for Arr {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        // Early return: empty array
        if self.arr.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }

        // Pre-allocate with exact capacity to avoid reallocations
        let mut arr = Vec::with_capacity(self.arr.len());
        for e in &self.arr {
            arr.push(e.calculate(json)?);
        }
        Ok(Value::Array(arr))
    }
}
