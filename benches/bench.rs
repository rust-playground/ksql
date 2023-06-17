#[macro_use]
extern crate criterion;

use chrono::Utc;
use criterion::{Criterion, Throughput};
use ksql::lexer::{Token, Tokenizer};
use ksql::parser::{Parser, Value};
use std::collections::BTreeMap;
use std::result::Result as StdResult;

fn benchmark_lexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lex_individual");
    for (name, src) in [
        ("selector_path", ".field1"),
        ("string", r#""My Name""#),
        ("number", "123.34"),
        ("bool", "true"),
        ("single_tok", "="),
        ("double_tok", ">="),
        ("contians_tok", "CONTAINS"),
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
        ("string_sp_add_multi", r#".first_name + " " + .last_name"#),
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

fn benchmark_expressions_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("expressions_execution_add");
    for (name, src, expression) in [
        (
            "coerce_spsubstr_const_eg_str",
            r#"{}"#.as_bytes(),
            r#"COERCE "Mr. Joeybloggs" _substr_[4:] == "JoeyBloggs""#,
        ),
        (
            "coerce_spsubstr_eg_str",
            r#"{"name":"Mr. Joeybloggs"}"#.as_bytes(),
            r#"COERCE .name _substr_[4:] == "JoeyBloggs""#,
        ),
        ("num_num", "".as_bytes(), "1 + 1"),
        ("sp_num", r#"{"field1":1}"#.as_bytes(), ".field1 + 1"),
        (
            "sp_sp",
            r#"{"field1":1,"field2":1}"#.as_bytes(),
            ".field + .field2",
        ),
        (
            "fname_lname",
            r#"{"first_name":"Joey","last_name":"Bloggs"}"#.as_bytes(),
            r#".first_name + " " + .last_name"#,
        ),
        (
            "coerce_const_dt_coerce_const_dt_eq",
            r#""#.as_bytes(),
            r#"COERCE "2022-07-15T00:00:00.000000000Z" _datetime_ == COERCE "2022-07-15" _datetime_"#,
        ),
        (
            "coerce_spdt_coerce_const_dt_eq",
            r#"{"dt1":"2022-07-15T00:00:00.000000000Z"}"#.as_bytes(),
            r#"COERCE .dt1 _datetime_ == COERCE "2022-07-15" _datetime_"#,
        ),
        (
            "coerce_spdt_coerce_spdt_eq",
            r#"{"dt1":"2022-07-15T00:00:00.000000000Z","dt2":"2022-07-15"}"#.as_bytes(),
            r#"COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_"#,
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

    let mut group = c.benchmark_group("expressions_execution_complex");
    for (name, src, expression) in [
        ("paren_div", "".as_bytes(), "(1 + 1) / 2"),
        (
            "paren_div_sps",
            r#"{"field1":1,"field2":1,"field3":2}"#.as_bytes(),
            "(.field1 + .field2) / .field3",
        ),
        (
            "company_employees",
            r#"{"name":"Company","properties":{"employees":50}}"#.as_bytes(),
            ".properties.employees > 20",
        ),
        (
            "not_paren_sp_not_sp",
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

fn benchmark_expressions_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("expressions_parsing_add");
    for (name, expression) in [
        ("num_num", "1 + 1"),
        ("sp_num", ".field1 + 1"),
        ("sp_sp", ".field + .field2"),
        ("fname_lname", r#".first_name + " " + .last_name"#),
        (
            "coerce_dt_coerce_dt_eq",
            r#"COERCE .dt1 _datetime_ == COERCE .dt2 _datetime_"#,
        ),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(expression.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Parser::parse(expression).unwrap();
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("expressions_parsing_complex");
    for (name, expression) in [
        ("paren_div", "(1 + 1) / 2"),
        ("paren_div_sps", "(.field1 + .field2) / .field3"),
        ("company_employees", ".properties.employees > 20"),
        ("not_paren_sp_not_sp", "!(.f1 != .f2)"),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(expression.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Parser::parse(expression).unwrap();
            })
        });
    }
    group.finish();
}

fn benchmark_serialize_json(c: &mut Criterion) {
    let mut m = BTreeMap::new();
    m.insert("key".to_string(), Value::String("value".to_string()));
    m.insert("key2".to_string(), Value::String("value2".to_string()));

    let mut group = c.benchmark_group("serialize_json");
    for (name, val) in [
        ("null", Value::Null),
        ("string", Value::String("string".to_string())),
        ("datetime", Value::DateTime(Utc::now())),
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
                let _res = serde_json::to_string(&val);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_expressions_execution,
    benchmark_expressions_parsing,
    benchmark_serialize_json,
    benchmark_lexer
);
criterion_main!(benches);
