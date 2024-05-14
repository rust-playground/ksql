use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct COERCEDateTime {
    pub value: BoxedExpression,
}

impl Expression for COERCEDateTime {
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let value = self.value.calculate(json)?;

        match value {
            Value::String(ref s) => match anydate::parse_utc(s) {
                Err(_) => Ok(Value::Null),
                Ok(dt) => Ok(Value::DateTime(dt)),
            },
            Value::Null => Ok(value),
            value => Err(Error::UnsupportedCOERCE(
                format!("{value} COERCE datetime",),
            )),
        }
    }
}
