# ksql &emsp; [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/ksql.svg
[crates.io]: https://crates.io/crates/ksql

**Is a JSON data expression lexer, parser, cli and library.**

#### How to install CLI
```shell
~ cargo install ksql
```

#### Usage
```rust
use ksql::parser::{Parser, Value};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>{
    let src = r#"{"name":"MyCompany", "properties":{"employees": 50}"#.as_bytes();
    let expression = ".properties.employees > 20";
    let ex = Parser::parse(expression.as_bytes())?;
    let result = ex.calculate(src)?;
    assert_eq!(Value::Bool(true), result);
    Ok(())
}
```

#### CLI Usage
```shell
~ ksql '(.field1 + 1) /2' '{"field1": 1}'
or
echo '{"field1": 1}' | ksql '(.field1 + 1) /2'
```

#### Expressions
Expressions support most mathematical and string expressions see below for details:

#### Syntax & Rules

| Token          | Example                  | Syntax Rules                                                                                                                                                                              |
|----------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Equals`       | `==`                     | supports both `==` and `=`.                                                                                                                                                               |
| `Add`          | `+`                      | N/A                                                                                                                                                                                       |
| `Subtract`     | `-`                      | N/A                                                                                                                                                                                       |
| `Multiply`     | `*`                      | N/A                                                                                                                                                                                       |
| `Divide`       | `/`                      | N/A                                                                                                                                                                                       |
| `Gt`           | `>`                      | N/A                                                                                                                                                                                       |
| `Gte`          | `>=`                     | N/A                                                                                                                                                                                       |
| `Lt`           | `<`                      | N/A                                                                                                                                                                                       |
| `Lte`          | `<=`                     | N/A                                                                                                                                                                                       |
| `OpenParen`    | `(`                      | N/A                                                                                                                                                                                       |
| `CloseParen`   | `)`                      | N/A                                                                                                                                                                                       |
| `OpenBracket`  | `[`                      | N/A                                                                                                                                                                                       |
| `CloseBracket` | `]`                      | N/A                                                                                                                                                                                       |
| `Comma`        | `,`                      | N/A                                                                                                                                                                                       |
| `QuotedString` | `"sample text"`          | Must start and end with an unescaped `"` character                                                                                                                                        |
| `Number`       | `123.45`                 | Must start and end with a valid `0-9` digit.                                                                                                                                              |
| `BoolenTrue`   | `true`                   | Accepts `true` as a boolean only.                                                                                                                                                         |
| `BoolenFalse`  | `false`                  | Accepts `false` as a boolean only.                                                                                                                                                        |
| `Identifier`   | `.identifier`            | Starts with a `.` and ends with whitespace blank space. This crate currently uses [gjson](https://github.com/tidwall/gjson.rs) and so the full gjson syntax for identifiers is supported. |
| `And`          | `&&`                     | N/A                                                                                                                                                                                       |
| `Not`          | `!`                      | Must be before Boolean identifier or expression or be followed by an operation                                                                                                            |
| `Or`           | <code>&vert;&vert;<code> | N/A                                                                                                                                                                                       |
| `Contains`     | `CONTAINS `              | Ends with whitespace blank space.                                                                                                                                                         |
| `In`           | `IN `                    | Ends with whitespace blank space.                                                                                                                                                         |
| `StartsWith`   | `STARTSWITH `            | Ends with whitespace blank space.                                                                                                                                                         |
| `EndsWith`     | `ENDSWITH `              | Ends with whitespace blank space.                                                                                                                                                         |
| `NULL`         | `NULL`                   | N/A                                                                                                                                                                                       |

    
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
