//! Target reference loading.
//!
//! The whole target sequence is held in memory, uppercased, with the degenerate
//! IUPAC codes the Python lists collapsed to `N`. Restricting to a contig set
//! stops reading once every requested contig has been seen, so a single-chromosome
//! run does not pay for the rest of the genome.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use flate2::read::MultiGzDecoder;

/// Codes replaced with `N` before any reference comparison.
const DEGENERATE: &[u8] = b"UWSMKRYBDHV*";

pub type Genome = HashMap<String, Vec<u8>>;

fn open_maybe_gzip(path: &Path) -> std::io::Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if path.to_string_lossy().ends_with(".gz") {
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

/// Load the target reference, optionally restricted to `chroms`.
pub fn load_target_genome(path: &Path, chroms: Option<&HashSet<String>>) -> Result<Genome, String> {
    let reader = BufReader::new(
        open_maybe_gzip(path).map_err(|e| format!("could not open target FASTA {path:?}: {e}"))?,
    );

    let mut seqs: Genome = HashMap::new();
    let mut current: Option<String> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut keep = false;

    // Returns true once every requested contig has been collected.
    let finish = |seqs: &mut Genome, name: Option<String>, buf: &mut Vec<u8>, keep: bool| {
        if let (Some(name), true) = (name, keep) {
            seqs.insert(name, std::mem::take(buf));
        } else {
            buf.clear();
        }
    };

    for line in reader.lines() {
        let line = line.map_err(|e| format!("error reading target FASTA: {e}"))?;
        if let Some(header) = line.strip_prefix('>') {
            finish(&mut seqs, current.take(), &mut buf, keep);
            if let Some(n) = chroms {
                if seqs.len() >= n.len() {
                    return Ok(normalize(seqs));
                }
            }
            // Record id is the header up to the first whitespace, as Biopython parses it.
            let id = header.split_whitespace().next().unwrap_or("").to_string();
            keep = chroms.map_or(true, |c| c.contains(&id));
            current = Some(id);
        } else {
            if keep {
                buf.extend(line.trim_end().bytes().map(|b| b.to_ascii_uppercase()));
            }
        }
    }
    finish(&mut seqs, current.take(), &mut buf, keep);
    Ok(normalize(seqs))
}

fn normalize(mut seqs: Genome) -> Genome {
    for seq in seqs.values_mut() {
        for base in seq.iter_mut() {
            if DEGENERATE.contains(base) {
                *base = b'N';
            }
        }
    }
    seqs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(body: &str, suffix: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn uppercases_and_joins_wrapped_lines() {
        let f = write(">chr1 some description\nacgt\nACGT\n", ".fa");
        let g = load_target_genome(f.path(), None).unwrap();
        assert_eq!(g["chr1"], b"ACGTACGT".to_vec());
        // The description after the first space is not part of the id.
        assert!(!g.contains_key("chr1 some description"));
    }

    #[test]
    fn degenerate_codes_become_n() {
        let f = write(">c\nACGTRYKMSWBDHVUN*\n", ".fa");
        let g = load_target_genome(f.path(), None).unwrap();
        assert_eq!(g["c"], b"ACGTNNNNNNNNNNNNN".to_vec());
    }

    #[test]
    fn contig_filter_selects_only_requested() {
        let f = write(">a\nAAAA\n>b\nCCCC\n>c\nGGGG\n", ".fa");
        let want: HashSet<String> = ["b".to_string()].into_iter().collect();
        let g = load_target_genome(f.path(), Some(&want)).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g["b"], b"CCCC".to_vec());
    }

    #[test]
    fn gzipped_input_is_decompressed() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(b">chr1\nacgt\n").unwrap();
        let gz = enc.finish().unwrap();
        let mut f = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
        f.write_all(&gz).unwrap();
        f.flush().unwrap();
        let g = load_target_genome(f.path(), None).unwrap();
        assert_eq!(g["chr1"], b"ACGT".to_vec());
    }
}
