use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Null;

impl Expression for Null {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Null)
    }
}
