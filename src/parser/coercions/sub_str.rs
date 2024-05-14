use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct CoerceSubstr {
    pub value: BoxedExpression,
    pub start_idx: Option<usize>,
    pub end_idx: Option<usize>,
}

impl Expression for CoerceSubstr {
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let v = self.value.calculate(json)?;
        match v {
            Value::String(s) => match (self.start_idx, self.end_idx) {
                (Some(start), Some(end)) => Ok(s
                    .get(start..end)
                    .map_or_else(|| Value::Null, |s| Value::String(s.to_string()))),
                (Some(start), None) => Ok(s
                    .get(start..)
                    .map_or_else(|| Value::Null, |s| Value::String(s.to_string()))),
                (None, Some(end)) => Ok(s
                    .get(..end)
                    .map_or_else(|| Value::Null, |s| Value::String(s.to_string()))),
                _ => Err(Error::UnsupportedCOERCE(format!(
                    "COERCE substr for {s}, [{:?}:{:?}]",
                    self.start_idx, self.end_idx
                ))),
            },
            v => Err(Error::UnsupportedCOERCE(format!("{v} COERCE substr",))),
        }
    }
}
