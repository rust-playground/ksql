# ksql &emsp; [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/ksql.svg
[crates.io]: https://crates.io/crates/ksql

**Is a JSON data expression lexer, parser, cli and library.**

#### How to install CLI
```shell
~ cargo install ksql
```

#### Expressions
Expressions support most mathematical and string expressions see [here](/LEXER.md) for details of the lexer support and rules.

#### Usage
```rust
use ksql::parser::{Parser, Value};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>{
    let src = r#"{"name":"MyCompany", "properties":{"employees": 50}"#;
    let expression = ".properties.employees > 20";
    let ex = Parser::parse(expression.as_bytes())?;
    let result = ex.calculate(src)?;
    assert_eq!(Value::Bool(true), result);
    Ok(())
}
```

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Proteus by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
</sub>
