use clap::Parser as ClapParser;
use ksql::parser::{Expression, Parser, Value};
use memmap2::Mmap;
use std::env;
use std::fs::File;
use std::io::{stdin, stdout, BufRead, BufReader, BufWriter, Write};

#[derive(Debug, ClapParser)]
#[clap(version = env!("CARGO_PKG_VERSION"), author = env!("CARGO_PKG_AUTHORS"), about = env!("CARGO_PKG_DESCRIPTION"))]
pub struct Opts {
    /// Indicates if the original data will be output after applying the expression.
    ///
    /// The results of the expression MUST be a boolean otherwise the output will be ignored.
    #[clap(short, long, default_value = "false")]
    pub output_original: bool,

    /// Input file to process. If not provided, stdin will be used.
    #[clap(short, long)]
    pub file: Option<String>,

    /// ksql expression to apply to input.
    #[clap()]
    pub expression: String,
}

fn main() -> anyhow::Result<()> {
    let opts: Opts = Opts::parse();

    let ex = Parser::parse(&opts.expression)?;

    let mut stdout = BufWriter::new(stdout().lock());

    if let Some(file) = opts.file {
        process_file(&file, &ex, &mut stdout, opts.output_original)?;
    } else {
        process_stdin(&ex, &mut stdout, opts.output_original)?;
    }
    Ok(())
}

#[inline]
fn process_file(
    file: &str,
    ex: &dyn Expression,
    stdout: &mut impl Write,
    output_original: bool,
) -> anyhow::Result<()> {
    let file = File::open(file)?;
    let mmap = unsafe { Mmap::map(&file)? };

    if output_original {
        for data in mmap.split(|b| *b == b'\n') {
            process_line_original_output(data, ex, &mut *stdout)?;
        }
    } else {
        for data in mmap.split(|b| *b == b'\n') {
            process_line(data, ex, &mut *stdout)?;
        }
    }
    Ok(())
}

#[inline]
fn process_stdin(
    ex: &dyn Expression,
    stdout: &mut impl Write,
    output_original: bool,
) -> anyhow::Result<()> {
    let mut stdin = BufReader::new(stdin().lock());
    let mut data = Vec::new();

    if output_original {
        while stdin.read_until(b'\n', &mut data)? > 0 {
            process_line_original_output(&data, ex, &mut *stdout)?;
            data.clear();
        }
    } else {
        while stdin.read_until(b'\n', &mut data)? > 0 {
            process_line(&data, ex, &mut *stdout)?;
            data.clear();
        }
    }
    Ok(())
}

#[inline]
fn process_line(line: &[u8], ex: &dyn Expression, stdout: &mut impl Write) -> anyhow::Result<()> {
    let v = ex.calculate(line)?;
    serde_json::to_writer(&mut *stdout, &v)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

#[inline]
fn process_line_original_output(
    line: &[u8],
    ex: &dyn Expression,
    stdout: &mut impl Write,
) -> anyhow::Result<()> {
    let v = ex.calculate(line)?;
    if let Value::Bool(true) = v {
        stdout.write_all(line)?;
        stdout.write_all(b"\n")?;
    }
    Ok(())
}
