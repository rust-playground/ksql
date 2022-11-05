use clap::Parser as ClapParser;
use ksql::parser::Parser;
use std::env;
use std::io::{stdin, stdout, BufRead, BufReader, Write};

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

    match opts.data {
        None => process(opts.expression, &mut stdin().lock()),
        Some(data) => process(opts.expression, &mut BufReader::new(data.as_bytes())),
    }
}

fn process<R>(expression: String, reader: &mut R) -> anyhow::Result<()>
where
    R: BufRead,
{
    let ex = Parser::parse(&expression)?;

    let stdout = stdout();
    let mut stdout = stdout.lock();
    let mut data = Vec::new();

    while reader.read_until(b'\n', &mut data)? > 0 {
        let v = ex.calculate(&data)?;
        serde_json::to_writer(&mut stdout, &v)?;
        let _ = stdout.write(&[b'\n'])?;
        data.clear();
    }
    Ok(())
}
