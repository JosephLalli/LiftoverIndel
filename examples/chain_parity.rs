//! Dump `convert_coordinate` results for a list of positions so they can be
//! diffed byte-for-byte against the same dump produced by pyliftover.
//!
//! Usage: chain_parity <chain file> <positions tsv: chrom\tpos>

use std::io::{BufRead, BufReader, BufWriter, Write};

use liftover_indels::chain::LiftOver;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lo = LiftOver::from_file(std::path::Path::new(&args[1])).expect("chain load");
    let queries = BufReader::new(std::fs::File::open(&args[2]).expect("positions"));
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in queries.lines() {
        let line = line.unwrap();
        let mut it = line.split('\t');
        let chrom = it.next().unwrap();
        let pos: i64 = it.next().unwrap().parse().unwrap();
        match lo.convert_coordinate(chrom, pos) {
            None => writeln!(out, "{chrom}\t{pos}\tNONE").unwrap(),
            Some(hits) => {
                let rendered: Vec<String> = hits
                    .iter()
                    .map(|h| format!("{},{},{},{}", h.chrom, h.pos, h.strand, h.score))
                    .collect();
                writeln!(out, "{chrom}\t{pos}\t{}", rendered.join(";")).unwrap();
            }
        }
    }
}
