use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct CoerceUppercase {
    pub value: BoxedExpression,
}

impl Expression for CoerceUppercase {
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let v = self.value.calculate(json)?;
        match v {
            Value::String(s) => Ok(Value::String(s.to_uppercase())),
            v => Err(Error::UnsupportedCOERCE(format!("{v} COERCE uppercase",))),
        }
    }
}
