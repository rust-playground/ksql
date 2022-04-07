use ksql::parser::{Parser, Value};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let src = r#"{"name":"MyCompany", "properties":{"employees": 50}"#.as_bytes();
    let ex = Parser::parse(".properties.employees > 20")?;
    let result = ex.calculate(src)?;
    println!("{}", &result);
    assert_eq!(Value::Bool(true), result);

    Ok(())
}
