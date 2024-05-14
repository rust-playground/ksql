use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Str {
    pub s: String,
}

impl Expression for Str {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::String(self.s.clone()))
    }
}
