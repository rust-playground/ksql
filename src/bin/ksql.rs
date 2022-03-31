use clap::Parser as ClapParser;
use ksql::parser::Parser;
use std::env;
use std::io::{stdin, BufRead};

#[derive(Debug, ClapParser)]
#[clap(version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = env!("CARGO_PKG_DESCRIPTION"))]
pub struct Opts {
    /// ksql expression to apply to input.
    #[clap()]
    pub expression: String,

    /// JSON data to apply expression against or piped from STDIN
    #[clap()]
    pub data: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let opts: Opts = Opts::parse();

    let is_pipe = !atty::is(atty::Stream::Stdin);

    let ex = Parser::parse(opts.expression.as_bytes())?;

    if is_pipe {
        let stdin = stdin();
        let mut stdin = stdin.lock(); // locking is optional

        let mut line = String::new();

        while stdin.read_line(&mut line)? > 0 {
            println!("{}", ex.calculate(line.as_bytes())?);
            line.clear();
        }
        Ok(())
    } else {
        match opts.data {
            None => Err(anyhow::anyhow!("No data provided")),
            Some(ref data) => {
                println!("{}", ex.calculate(data.as_bytes())?);
                Ok(())
            }
        }
    }
}
