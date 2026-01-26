use clap::Parser as ClapParser;
use crossbeam_channel::bounded;
use ksql::parser::{Expression, Parser, Value};
use memchr::{memchr, memchr_iter};
use memmap2::Mmap;
use std::cmp::max;
use std::env;
use std::fs::File;
use std::io::{stdin, stdout, BufRead, BufWriter, Write};
use std::sync::Arc;

const NEWLINE: u8 = b'\n';
const NEWLINE_SLICE: &[u8] = b"\n";

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

    /// Number of parallel threads to use when processing a file. Defaults to number of available CPUs.
    /// Set to 1 to maintain output order matching input order (FIFO). Parallel processing may produce out-of-order results.
    #[clap(short, long)]
    pub pthreads: Option<usize>,

    /// Batch size for parallel processing. Larger batches improve throughput but use more memory.
    #[clap(long, default_value = "10000")]
    pub batch_size: usize,

    /// ksql expression to apply to input.
    #[clap()]
    pub expression: String,
}

fn main() -> anyhow::Result<()> {
    let opts: Opts = Opts::parse();

    if opts.file.is_some() {
        process_file(&opts)?;
    } else {
        process_stdin(&opts)?;
    }

    Ok(())
}

#[inline]
fn process_file(opts: &Opts) -> anyhow::Result<()> {
    let file = File::open(opts.file.as_ref().unwrap())?;
    let map = unsafe { Mmap::map(&file)? };
    let nthreads = max(
        opts.pthreads
            .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get())
            - 1,
        1,
    );
    let mut stdout = BufWriter::new(stdout().lock());

    let ex = Arc::new(Parser::parse(&opts.expression).unwrap());

    if opts.output_original {
        std::thread::scope(|scope| {
            let mut at = 0;
            let (tx, rx) = std::sync::mpsc::sync_channel(nthreads * 2);
            let (return_tx, return_rx) = bounded(nthreads * 2);
            let chunk_size = map.len() / nthreads;
            let batch_size = opts.batch_size;

            for _ in 0..nthreads {
                let start = at;
                let end = (at + chunk_size).min(map.len());
                let end = if end == map.len() {
                    map.len()
                } else {
                    let newline_at = memchr(NEWLINE, &map[end..]).unwrap();
                    end + newline_at + 1
                };
                let map = &map[start..end];
                if map.is_empty() {
                    break;
                }
                at = end;
                let tx = tx.clone();
                let ex = ex.clone();
                let return_rx = return_rx.clone();
                scope.spawn(move || {
                    process_chunk_original(map, &tx, &return_rx, batch_size, ex.as_ref()).unwrap();
                });
            }

            drop(tx);

            for mut results in rx {
                for r in &results {
                    stdout.write_all(r).unwrap();
                    stdout.write_all(NEWLINE_SLICE).unwrap();
                }
                // Return Vec to pool for reuse
                results.clear();
                let _ = return_tx.try_send(results);
            }
        });
    } else {
        std::thread::scope(|scope| {
            let mut at = 0;
            let (tx, rx) = std::sync::mpsc::sync_channel(nthreads * 2);
            let (return_tx, return_rx) = bounded(nthreads * 2);
            let chunk_size = map.len() / nthreads;
            let batch_size = opts.batch_size;

            for _ in 0..nthreads {
                let start = at;
                let end = (at + chunk_size).min(map.len());
                let end = if end == map.len() {
                    map.len()
                } else {
                    let newline_at = memchr(NEWLINE, &map[end..]).unwrap();
                    end + newline_at + 1
                };
                let map = &map[start..end];
                if map.is_empty() {
                    break;
                }
                at = end;
                let tx = tx.clone();
                let ex = ex.clone();
                let return_rx = return_rx.clone();
                scope.spawn(move || {
                    process_chunk(map, &tx, &return_rx, batch_size, ex.as_ref()).unwrap();
                });
            }

            drop(tx);

            for mut results in rx {
                for r in &results {
                    serde_json::to_writer(&mut stdout, r).unwrap();
                    stdout.write_all(NEWLINE_SLICE).unwrap();
                }
                // Return Vec to pool for reuse
                results.clear();
                let _ = return_tx.try_send(results);
            }
        });
    }

    Ok(())
}

#[inline]
fn process_stdin(opts: &Opts) -> anyhow::Result<()> {
    let ex = Parser::parse(&opts.expression).unwrap();
    let mut stdin = stdin().lock();
    let mut stdout = BufWriter::new(stdout().lock());
    let mut line = Vec::new();

    if opts.output_original {
        while stdin.read_until(NEWLINE, &mut line)? > 0 {
            let v = ex.calculate(&line)?;
            if let Value::Bool(true) = v {
                stdout.write_all(&line)?;
                stdout.write_all(NEWLINE_SLICE)?;
            }
            line.clear();
        }
    } else {
        while stdin.read_until(NEWLINE, &mut line)? > 0 {
            let v = ex.calculate(&line)?;
            serde_json::to_writer(&mut stdout, &v)?;
            stdout.write_all(NEWLINE_SLICE)?;
            line.clear();
        }
    }
    Ok(())
}

#[inline]
fn process_chunk(
    chunk: &[u8],
    tx: &std::sync::mpsc::SyncSender<Vec<Value>>,
    return_rx: &crossbeam_channel::Receiver<Vec<Value>>,
    batch_size: usize,
    ex: &dyn Expression,
) -> anyhow::Result<()> {
    let mut start = 0;
    let mut to_write = return_rx
        .try_recv()
        .ok()
        .map(|mut v| {
            v.clear();
            v
        })
        .unwrap_or_else(|| Vec::with_capacity(batch_size));

    for end in memchr_iter(NEWLINE, chunk) {
        let line = &chunk[start..end];

        let v = ex.calculate(line)?;
        to_write.push(v);

        if to_write.len() >= batch_size {
            tx.send(to_write)?;
            to_write = return_rx
                .try_recv()
                .ok()
                .map(|mut v| {
                    v.clear();
                    v
                })
                .unwrap_or_else(|| Vec::with_capacity(batch_size));
        }

        start = end + 1;
    }

    if !to_write.is_empty() {
        tx.send(to_write)?;
    }
    Ok(())
}

#[inline]
fn process_chunk_original(
    chunk: &[u8],
    tx: &std::sync::mpsc::SyncSender<Vec<Vec<u8>>>,
    return_rx: &crossbeam_channel::Receiver<Vec<Vec<u8>>>,
    batch_size: usize,
    ex: &dyn Expression,
) -> anyhow::Result<()> {
    let mut start = 0;
    let mut to_write = return_rx
        .try_recv()
        .ok()
        .map(|mut v| {
            v.clear();
            v
        })
        .unwrap_or_else(|| Vec::with_capacity(batch_size));

    let newline_indices = memchr_iter(NEWLINE, chunk);

    for newline_index in newline_indices {
        let line = &chunk[start..newline_index];

        let v = ex.calculate(line)?;
        if let Value::Bool(true) = v {
            to_write.push(line.to_vec());

            if to_write.len() >= batch_size {
                tx.send(to_write)?;
                to_write = return_rx
                    .try_recv()
                    .ok()
                    .map(|mut v| {
                        v.clear();
                        v
                    })
                    .unwrap_or_else(|| Vec::with_capacity(batch_size));
            }
        }

        start = newline_index + 1;
    }

    if !to_write.is_empty() {
        tx.send(to_write)?;
    }
    Ok(())
}
