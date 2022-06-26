#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use ksql::lexer::{Token, Tokenizer};
use ksql::parser::{Parser, Value};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

fn benchmark_lexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lex_individual");
    for (name, src) in [
        ("identifier", ".field1"),
        ("string", r#""My Name""#),
        ("number", "123.34"),
        ("bool", "true"),
        ("single_ident", "="),
        ("double_ident", ">="),
        ("string_ident", "CONTAINS"),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Tokenizer::new(src).next();
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("lex_expression");
    for (name, src) in [
        (
            "string_ident_add_multi",
            r#".first_name + " " + .last_name"#,
        ),
        ("math", r#"1 + 1 / 2"#),
        ("math_paren", r#"(1 + 1) / 2"#),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Tokenizer::new(src)
                    .collect::<StdResult<Vec<Token>, _>>()
                    .unwrap();
            })
        });
    }
    group.finish();
}

fn benchmark_expressions(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for (name, src, expression) in [
        ("num_num", "".as_bytes(), "1 + 1"),
        ("ident_num", r#"{"field1":1}"#.as_bytes(), ".field1 + 1"),
        (
            "ident_ident",
            r#"{"field1":1,"field2":1}"#.as_bytes(),
            ".field + .field2",
        ),
        (
            "fname_lname",
            r#"{"first_name":"Joey","last_name":"Bloggs"}"#.as_bytes(),
            r#".first_name + " " + .last_name"#,
        ),
    ]
    .iter()
    {
        let ex = Parser::parse(expression).unwrap();
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = ex.calculate(src);
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("complex");
    for (name, src, expression) in [
        ("paren_div", "".as_bytes(), "(1 + 1) / 2"),
        (
            "paren_div_idents",
            r#"{"field1":1,"field2":1,"field3":2}"#.as_bytes(),
            "(.field1 + .field2) / .field3",
        ),
        (
            "company_employees",
            r#"{"name":"Company","properties":{"employees":50}}"#.as_bytes(),
            ".properties.employees > 20",
        ),
        (
            "not_paren_not",
            r#"{"f1":true,"f2":false}"#.as_bytes(),
            "!(.f1 != .f2)",
        ),
    ]
    .iter()
    {
        let ex = Parser::parse(expression).unwrap();
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = ex.calculate(src);
            })
        });
    }
    group.finish();
}

fn benchmark_display(c: &mut Criterion) {
    let mut m = BTreeMap::new();
    m.insert("key".to_string(), Value::String("value".to_string()));
    m.insert("key2".to_string(), Value::String("value2".to_string()));

    let mut group = c.benchmark_group("display");
    for (name, val) in [
        ("null", Value::Null),
        ("string", Value::String("string".to_string())),
        ("number", Value::Number(64.1)),
        ("bool", Value::Bool(true)),
        ("object", Value::Object(m)),
        (
            "array",
            Value::Array(vec![
                Value::String("string".to_string()),
                Value::Number(64.1),
            ]),
        ),
    ]
    .iter()
    {
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = format!("{}", val);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_display,
    // benchmark_expressions,
    // benchmark_lexer
);
criterion_main!(benches);
