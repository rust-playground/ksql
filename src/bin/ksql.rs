use clap::Parser as ClapParser;
use ksql::parser::Parser;
use std::env;
use std::io::{stdin, stdout, BufRead, Write};

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

    let stdout = stdout();
    let mut stdout = stdout.lock();

    if is_pipe {
        let stdin = stdin();
        let mut stdin = stdin.lock();

        let mut data = Vec::new();

        while stdin.read_until(b'\n', &mut data)? > 0 {
            let v = ex.calculate(&data)?;
            writeln!(stdout, "{}", v)?;
            data.clear();
        }
        Ok(())
    } else {
        match opts.data {
            None => Err(anyhow::anyhow!("No data provided")),
            Some(data) => {
                let v = ex.calculate(data.as_bytes())?;
                writeln!(stdout, "{}", v)?;
                Ok(())
            }
        }
    }
}
