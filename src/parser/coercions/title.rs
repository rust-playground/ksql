use crate::parser::parse::BoxedExpression;
use crate::parser::{Error, Expression, Value};

#[derive(Debug)]
pub(in crate::parser) struct CoerceTitle {
    pub value: BoxedExpression,
}

impl Expression for CoerceTitle {
    fn calculate(&self, json: &[u8]) -> crate::parser::parse::Result<Value> {
        let v = self.value.calculate(json)?;
        match v {
            Value::String(s) => {
                let mut c = s.chars();
                match c.next() {
                    None => Ok(Value::String(s)),
                    Some(f) => Ok(Value::String(
                        f.to_uppercase().collect::<String>() + c.as_str().to_lowercase().as_str(),
                    )),
                }
            }
            v => Err(Error::UnsupportedCOERCE(format!("{v} COERCE title",))),
        }
    }
}
