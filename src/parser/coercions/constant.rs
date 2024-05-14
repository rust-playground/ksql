use crate::parser::{Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct CoercedConst {
    pub value: Value,
}

impl Expression for CoercedConst {
    fn calculate(&self, _json: &[u8]) -> crate::parser::parse::Result<Value> {
        Ok(self.value.clone())
    }
}
