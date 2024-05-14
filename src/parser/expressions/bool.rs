use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Bool {
    pub b: bool,
}

impl Expression for Bool {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Bool(self.b))
    }
}
