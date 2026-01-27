use crate::parser::{Expression, Result, Value};

#[derive(Debug)]
pub(in crate::parser) struct SelectorPath {
    pub ident: String,
}

impl Expression for SelectorPath {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        Ok(unsafe { gjson::get_bytes(json, &self.ident).into() })
    }
}
