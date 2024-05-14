use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct COERCENumber {
    pub value: BoxedExpression,
}

impl Expression for COERCENumber {
    #[allow(clippy::cast_precision_loss)]
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let value = self.value.calculate(json)?;
        match value {
            Value::String(s) => Ok(Value::Number(
                s.parse::<f64>()
                    .map_err(|e| Error::UnsupportedCOERCE(e.to_string()))?,
            )),
            Value::Number(num) => Ok(Value::Number(num)),
            Value::Bool(b) => Ok(Value::Number(if b { 1.0 } else { 0.0 })),
            Value::DateTime(dt) => Ok(Value::Number(
                dt.timestamp_nanos_opt().unwrap_or_default() as f64
            )),
            _ => Err(Error::UnsupportedCOERCE(
                format!("{value} COERCE datetime",),
            )),
        }
    }
}
