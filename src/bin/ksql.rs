use clap::Parser as ClapParser;
use ksql::parser::{Parser, Value};
use std::env;
use std::io::{stdin, stdout, BufRead, Write};

#[derive(Debug, ClapParser)]
#[clap(version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = env!("CARGO_PKG_DESCRIPTION"))]
pub struct Opts {
    /// Indicates if the original data will be output after applying the expression.
    ///
    /// The results of the expression MUST be a boolean otherwise the output will be ignored.
    #[clap(short, long, default_value = "false")]
    pub output_original: bool,

    /// ksql expression to apply to input.
    #[clap()]
    pub expression: String,

    /// JSON data to apply expression against or piped from STDIN
    #[clap()]
    pub data: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let opts: Opts = Opts::parse();

    let ex = Parser::parse(&opts.expression)?;

    let stdout = stdout();
    let mut stdout = stdout.lock();

    if let Some(data) = opts.data {
        let bytes = data.as_bytes();
        let v = ex.calculate(bytes)?;
        if opts.output_original {
            match v {
                Value::Bool(output_original) if output_original => {
                    stdout.write_all(bytes)?;
                    let _ = stdout.write(&[b'\n'])?;
                }
                _ => {}
            }
        } else {
            serde_json::to_writer(&mut stdout, &v)?;
            let _ = stdout.write(&[b'\n'])?;
        }
    } else {
        let stdin = stdin();
        let mut stdin = stdin.lock();
        let mut data = Vec::new();

        if opts.output_original {
            while stdin.read_until(b'\n', &mut data)? > 0 {
                let v = ex.calculate(&data)?;
                match v {
                    Value::Bool(output_original) if output_original => {
                        stdout.write_all(&data)?;
                    }
                    _ => {}
                }
                data.clear();
            }
        } else {
            while stdin.read_until(b'\n', &mut data)? > 0 {
                let v = ex.calculate(&data)?;
                serde_json::to_writer(&mut stdout, &v)?;
                let _ = stdout.write(&[b'\n'])?;
                data.clear();
            }
        }
    }
    Ok(())
}
