//! # KSQL
//!
//! Is a JSON data expression lexer, parser, cli and library.
//!
//! #### Expressions
//! Expressions support most mathematical and string expressions see [here](https://github.com/rust-playground/ksql/LEXER.md) for details of the lexer support and rules.
//!
//! ```rust
//! use ksql::parser::{Parser, Value};
//! use std::error::Error;
//!
//! fn main() -> Result<(), Box<dyn Error>>{
//!     let src = r#"{"name":"MyCompany", "properties":{"employees": 50}"#;
//!     let expression = ".properties.employees > 20";
//!
//!     let ex = Parser::parse(expression.as_bytes())?;
//!     let result = ex.calculate(src)?;
//!     assert_eq!(Value::Bool(true), result);
//!     Ok(())
//! }
//! ```
//!
//!

/// KSQL Expression lexer
pub mod lexer;

/// KSQL Expression parser
pub mod parser;
