#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use ksql::lexer::Tokenizer;
use ksql::parser::Parser;

fn benchmark_lexer(c: &mut Criterion) {
    let mut group = c.benchmark_group("lex_individual");
    for (name, src) in [
        ("identifier", ".field1".as_bytes()),
        ("string", r#""My Name""#.as_bytes()),
        ("number", "123.34".as_bytes()),
        ("bool", "true".as_bytes()),
        ("single_ident", "=".as_bytes()),
        ("double_ident", ">=".as_bytes()),
        ("string_ident", "CONTAINS".as_bytes()),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Tokenizer::tokenize(src);
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("lex_expression");
    for (name, src) in [
        (
            "string_ident_add_multi",
            r#".first_name + " " + .last_name"#.as_bytes(),
        ),
        ("math", r#"1 + 1 / 2"#.as_bytes()),
        ("math_paren", r#"(1 + 1) / 2"#.as_bytes()),
    ]
    .iter()
    {
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = Tokenizer::tokenize(src);
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
        let ex = Parser::parse(expression.as_bytes()).unwrap();
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
        let ex = Parser::parse(expression.as_bytes()).unwrap();
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_function(*name, |b| {
            b.iter(|| {
                let _res = ex.calculate(src);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_expressions, benchmark_lexer);
criterion_main!(benches);
