use ksql::parser::Parser;
use std::env;
use std::io::{stdin, Read};

fn main() -> anyhow::Result<()> {
    let is_pipe = !atty::is(atty::Stream::Stdin);
    let args: Vec<String> = env::args().collect();
    if (args.len() < 2 && !is_pipe) || (args.is_empty() && is_pipe) {
        usage();
        return Ok(());
    }

    let ex = Parser::parse(args[1].as_bytes())?;

    let mut s;
    if is_pipe {
        s = String::new();
        let mut reader = stdin();
        reader.read_to_string(&mut s)?;
    } else {
        s = args[2].clone();
    }

    println!("{}", ex.calculate(s.as_bytes())?);
    Ok(())
}

fn usage() {
    println!("ksql <expression> <json>");
    println!("or");
    println!("echo '{{}}' | ksql <expression> -");
}
