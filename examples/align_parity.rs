//! Re-run the Python alignment reference cases through the Rust implementation
//! so the two dumps can be diffed.
//!
//! Usage: align_parity <cases tsv: ref\talt>

use std::io::{BufRead, BufReader, BufWriter, Write};

use liftover_indels::align::{global_align, variant_from_alignment, TieBreak};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases = BufReader::new(std::fs::File::open(&args[1]).expect("cases"));
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in cases.lines() {
        let line = line.unwrap();
        let mut it = line.split('\t');
        let r = it.next().unwrap();
        let a = it.next().unwrap_or("");
        for (label, tb) in [("left", TieBreak::Left), ("right", TieBreak::Right)] {
            let (aln_ref, aln_alt) = global_align(r.as_bytes(), a.as_bytes(), tb);
            // The Python harness passes the reference itself as the window.
            let rendered = if r.is_empty() {
                "ERR:IndexError".to_string()
            } else {
                match variant_from_alignment(&aln_ref, &aln_alt, 1000, r.as_bytes()) {
                    None => "None".to_string(),
                    Some((pos, vr, va)) => format!("({pos}, '{vr}', '{va}')"),
                }
            };
            writeln!(
                out,
                "{r}\t{a}\t{label}\t{}\t{}\t{rendered}",
                String::from_utf8_lossy(&aln_ref),
                String::from_utf8_lossy(&aln_alt)
            )
            .unwrap();
        }
    }
}
