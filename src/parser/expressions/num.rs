use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct Num {
    pub n: f64,
}

impl Expression for Num {
    fn calculate(&self, _: &[u8]) -> Result<Value> {
        Ok(Value::Number(self.n))
    }
}
