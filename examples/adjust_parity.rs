//! Re-run the Python compute_adjusted_ref_alt cases through the Rust
//! implementation so the two dumps can be diffed.
//!
//! Usage: adjust_parity <cases tsv>

use std::io::{BufRead, BufReader, BufWriter, Write};

use liftover_indels::adjust::compute_adjusted_ref_alt;
use liftover_indels::error::VariantError;
use liftover_indels::refdiff::RefDiff;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases = BufReader::new(std::fs::File::open(&args[1]).expect("cases"));
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in cases.lines() {
        let line = line.unwrap();
        let f: Vec<&str> = line.split('\t').collect();
        let (var_ref, var_alt) = (f[0], f[1]);
        let dstart: i64 = f[2].parse().unwrap();
        let (dref, dalt) = (f[3], f[4]);
        let ls: i64 = f[5].parse().unwrap();
        let le: i64 = f[6].parse().unwrap();

        let diff = RefDiff {
            start: dstart,
            end: dstart + dref.len() as i64,
            ref_allele: dref.to_string(),
            alt_allele: dalt.to_string(),
        };

        let res = match compute_adjusted_ref_alt(var_ref, var_alt, &diff, ls, le) {
            Ok((r, a)) => format!("OK\t{r}\t{a}"),
            Err(VariantError::Unliftable(m)) => format!("UNLIFTABLE\t{m}\t"),
            Err(VariantError::Mismatch(m)) => format!("MISMATCH\t{m}\t"),
        };
        writeln!(
            out,
            "{var_ref}\t{var_alt}\t{dstart}\t{dref}\t{dalt}\t{ls}\t{le}\t{res}"
        )
        .unwrap();
    }
}
