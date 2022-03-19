use ksql::parser::Parser;
use std::env;
use std::io::{stdin, Read};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage();
        return Ok(());
    }

    let mut s;
    if args[2] == "-" {
        s = String::new();
        let mut reader = stdin();
        reader.read_to_string(&mut s)?;
    } else {
        s = args[2].clone();
    }

    let ex = Parser::parse(args[1].as_bytes())?;

    println!("{}", ex.calculate(&s)?);
    Ok(())
}

fn usage() {
    println!("ksql <expression> <json>");
    println!("or");
    println!("echo '{{}}' | ksql <expression> -");
}
