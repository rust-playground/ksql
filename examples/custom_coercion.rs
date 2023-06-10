use ksql::parser::{Error, Expression, Parser, Result, Value};

#[derive(Debug)]
struct Star {
    expression: Box<dyn Expression>,
}

impl Expression for Star {
    fn calculate(&self, json: &[u8]) -> Result<Value> {
        let inner = self.expression.calculate(json)?;

        match inner {
            Value::String(s) => Ok(Value::String("*".repeat(s.len()))),
            value => Err(Error::UnsupportedCOERCE(format!(
                "cannot star value {value}"
            ))),
        }
    }
}

fn main() -> anyhow::Result<()> {
    {
        // Add custom coercion to the parser.
        // REMEMBER: coercions start and end with an _(underscore).
        let mut hm = ksql::parser::custom_coercions().write().unwrap();
        hm.insert("_star_".to_string(), |_, expression| {
            Ok((true, Box::new(Star { expression })))
        });
    }

    let expression = r#"COERCE "My Name" _star_"#;
    let ex = Parser::parse(expression)?;
    let result = ex.calculate("{}".as_bytes())?;
    println!("result: {result}");
    assert_eq!(Value::String("*******".to_string()), result);

    Ok(())
}
