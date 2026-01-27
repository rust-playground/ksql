use crate::parser::parse::BoxedExpression;
use crate::parser::{Expression, Value};
use chrono::SecondsFormat;

#[derive(Debug)]
pub(in crate::parser) struct COERCEString {
    pub value: BoxedExpression,
}

impl Expression for COERCEString {
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let value = self.value.calculate(json)?;
        match value {
            Value::Null => Ok(Value::String("null".to_string())),
            Value::String(s) => Ok(Value::String(s)),
            Value::Number(num) => Ok(Value::String(num.to_string())),
            Value::Bool(b) => Ok(Value::String(b.to_string())),
            Value::DateTime(dt) => Ok(Value::String(
                dt.to_rfc3339_opts(SecondsFormat::AutoSi, true),
            )),
            Value::Array(_) | Value::Object(_) => Ok(Value::String(value.to_string())),
        }
    }
}
