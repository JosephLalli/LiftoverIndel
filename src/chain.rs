//! UCSC `.over.chain` coordinate conversion.
//!
//! This mirrors the `pyliftover` implementation the Python tool relies on, including
//! the details that decide off-by-one behaviour: zero-length alignment blocks are
//! dropped, minus-strand results are mirrored with `target_size - 1 - position`, and
//! a point query returns every block satisfying `start <= x < end`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use flate2::read::MultiGzDecoder;

/// A single conversion result: target contig, 0-based target position, target
/// strand and the score of the chain that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversion {
    pub chrom: String,
    pub pos: i64,
    pub strand: char,
    pub score: i64,
}

struct Chain {
    score: i64,
    target_name: String,
    target_size: i64,
    target_strand: char,
}

/// One gap-free alignment block, half-open in source coordinates.
struct Block {
    source_start: i64,
    source_end: i64,
    target_start: i64,
    chain: usize,
}

/// Blocks for a single source contig, sorted by start with a running maximum of
/// `source_end` so a stabbing query can prune without visiting every block.
struct ContigIndex {
    blocks: Vec<Block>,
    max_end: Vec<i64>,
}

pub struct LiftOver {
    chains: Vec<Chain>,
    index: HashMap<String, ContigIndex>,
}

fn open_maybe_gzip(path: &Path) -> std::io::Result<Box<dyn Read>> {
    let file = File::open(path)?;
    let is_gz = path
        .to_string_lossy()
        .to_lowercase()
        .ends_with(".gz");
    if is_gz {
        Ok(Box::new(MultiGzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

impl LiftOver {
    /// Parse a chain file. `.gz` inputs are decompressed, matching pyliftover's
    /// extension-based detection.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let reader = BufReader::new(
            open_maybe_gzip(path).map_err(|e| format!("could not open chain file {path:?}: {e}"))?,
        );

        let mut chains: Vec<Chain> = Vec::new();
        // Collected per source contig before sorting.
        let mut pending: HashMap<String, Vec<Block>> = HashMap::new();
        let mut source_size: HashMap<String, i64> = HashMap::new();
        let mut target_size: HashMap<String, i64> = HashMap::new();

        let mut lines = reader.lines();
        while let Some(line) = lines.next() {
            let line = line.map_err(|e| format!("error reading chain file: {e}"))?;
            if line.is_empty() || line.starts_with('#') || line.starts_with('\r') {
                continue;
            }
            if !line.starts_with("chain") {
                continue;
            }

            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 12 && f.len() != 13 {
                return Err(format!("Invalid chain format. ({line})"));
            }
            let parse = |s: &str, what: &str| -> Result<i64, String> {
                s.parse::<i64>()
                    .map_err(|_| format!("Invalid {what} in chain header. ({line})"))
            };

            let score = parse(f[1], "score")?;
            let src_name = f[2].to_string();
            let src_size = parse(f[3], "source size")?;
            if f[4] != "+" {
                return Err(format!(
                    "Source strand in an .over.chain file must be +. ({line})"
                ));
            }
            let src_start = parse(f[5], "source start")?;
            let src_end = parse(f[6], "source end")?;
            let tgt_name = f[7].to_string();
            let tgt_size = parse(f[8], "target size")?;
            let tgt_strand = match f[9] {
                "+" => '+',
                "-" => '-',
                _ => return Err(format!("Target strand must be - or +. ({line})")),
            };
            let tgt_start = parse(f[10], "target start")?;
            let tgt_end = parse(f[11], "target end")?;

            // pyliftover rejects chain files that disagree about a contig's length.
            if let Some(&known) = source_size.get(&src_name) {
                if known != src_size {
                    return Err(format!(
                        "Chains have inconsistent specification of source chromosome size for {src_name} ({known} vs {src_size})"
                    ));
                }
            } else {
                source_size.insert(src_name.clone(), src_size);
            }
            if let Some(&known) = target_size.get(&tgt_name) {
                if known != tgt_size {
                    return Err(format!(
                        "Chains have inconsistent specification of target chromosome size for {tgt_name} ({known} vs {tgt_size})"
                    ));
                }
            } else {
                target_size.insert(tgt_name.clone(), tgt_size);
            }

            let chain_idx = chains.len();
            chains.push(Chain {
                score,
                target_name: tgt_name,
                target_size: tgt_size,
                target_strand: tgt_strand,
            });

            let slot = pending.entry(src_name).or_default();

            // Alignment block lines: `size dt dq`, terminated by a lone `size`.
            let mut sfrom = src_start;
            let mut tfrom = tgt_start;
            loop {
                let raw = lines
                    .next()
                    .ok_or_else(|| format!("Unexpected end of chain file. ({line})"))?
                    .map_err(|e| format!("error reading chain file: {e}"))?;
                let parts: Vec<&str> = raw.split_whitespace().collect();
                if parts.len() == 3 {
                    let size: i64 = parts[0]
                        .parse()
                        .map_err(|_| format!("Invalid block size. ({line})"))?;
                    let sgap: i64 = parts[1]
                        .parse()
                        .map_err(|_| format!("Invalid source gap. ({line})"))?;
                    let tgap: i64 = parts[2]
                        .parse()
                        .map_err(|_| format!("Invalid target gap. ({line})"))?;
                    // Zero-length blocks are never indexed by pyliftover.
                    if size > 0 {
                        slot.push(Block {
                            source_start: sfrom,
                            source_end: sfrom + size,
                            target_start: tfrom,
                            chain: chain_idx,
                        });
                    }
                    sfrom += size + sgap;
                    tfrom += size + tgap;
                } else if parts.len() == 1 {
                    let size: i64 = parts[0]
                        .parse()
                        .map_err(|_| format!("Invalid final block size. ({line})"))?;
                    if size > 0 {
                        slot.push(Block {
                            source_start: sfrom,
                            source_end: sfrom + size,
                            target_start: tfrom,
                            chain: chain_idx,
                        });
                    }
                    if sfrom + size != src_end || tfrom + size != tgt_end {
                        return Err(format!(
                            "Alignment blocks do not match specified block sizes. ({line})"
                        ));
                    }
                    break;
                } else {
                    return Err(format!(
                        "Expecting one number on the last line of alignments block. ({line})"
                    ));
                }
            }
        }

        let mut index = HashMap::with_capacity(pending.len());
        for (contig, mut blocks) in pending {
            blocks.sort_by_key(|b| b.source_start);
            let mut max_end = Vec::with_capacity(blocks.len());
            let mut running = i64::MIN;
            for b in &blocks {
                running = running.max(b.source_end);
                max_end.push(running);
            }
            index.insert(contig, ContigIndex { blocks, max_end });
        }

        Ok(LiftOver { chains, index })
    }

    /// Convert a 0-based source coordinate.
    ///
    /// Returns `None` when the contig is absent from every chain (pyliftover
    /// returns `None` in that case, which the caller treats as unliftable), and
    /// otherwise every block that covers the position, ordered by decreasing
    /// chain score.
    pub fn convert_coordinate(&self, chrom: &str, pos: i64) -> Option<Vec<Conversion>> {
        let contig = self.index.get(chrom)?;

        // Every candidate has source_start <= pos, so only that prefix is searched;
        // the prefix maximum of source_end lets us stop as soon as no earlier block
        // can still reach the query point.
        let mut hi = contig.blocks.partition_point(|b| b.source_start <= pos);
        let mut results: Vec<Conversion> = Vec::new();
        while hi > 0 {
            if contig.max_end[hi - 1] <= pos {
                break;
            }
            let b = &contig.blocks[hi - 1];
            if b.source_start <= pos && pos < b.source_end {
                let chain = &self.chains[b.chain];
                let mut target_pos = b.target_start + (pos - b.source_start);
                if chain.target_strand == '-' {
                    target_pos = chain.target_size - 1 - target_pos;
                }
                results.push(Conversion {
                    chrom: chain.target_name.clone(),
                    pos: target_pos,
                    strand: chain.target_strand,
                    score: chain.score,
                });
            }
            hi -= 1;
        }

        results.sort_by(|a, b| b.score.cmp(&a.score));
        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_chain(body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn plus_strand_offsets_are_preserved() {
        // 10 source bases at 100 map to 10 target bases at 500.
        let f = write_chain("chain 255 src 1000 + 100 110 tgt 2000 + 500 510 1\n10\n");
        let lo = LiftOver::from_file(f.path()).unwrap();
        let hits = lo.convert_coordinate("src", 103).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].pos, 503);
        assert_eq!(hits[0].strand, '+');
        // Half-open: the end coordinate itself is outside the block.
        assert!(lo.convert_coordinate("src", 110).unwrap().is_empty());
        assert_eq!(lo.convert_coordinate("src", 100).unwrap()[0].pos, 500);
    }

    #[test]
    fn minus_strand_mirrors_about_target_size() {
        let f = write_chain("chain 255 src 1000 + 100 110 tgt 2000 - 500 510 1\n10\n");
        let lo = LiftOver::from_file(f.path()).unwrap();
        let hits = lo.convert_coordinate("src", 103).unwrap();
        // 500 + 3 = 503; mirrored: 2000 - 1 - 503 = 1496.
        assert_eq!(hits[0].pos, 1496);
        assert_eq!(hits[0].strand, '-');
    }

    #[test]
    fn zero_length_blocks_are_not_indexed() {
        // A zero-size block sits between two real ones and must never match.
        let f = write_chain("chain 255 src 1000 + 0 20 tgt 1000 + 0 20 1\n10 0 0\n0 0 0\n10\n");
        let lo = LiftOver::from_file(f.path()).unwrap();
        assert_eq!(lo.convert_coordinate("src", 5).unwrap().len(), 1);
        assert_eq!(lo.convert_coordinate("src", 15).unwrap().len(), 1);
    }

    #[test]
    fn unknown_contig_is_none_not_empty() {
        let f = write_chain("chain 255 src 1000 + 100 110 tgt 2000 + 500 510 1\n10\n");
        let lo = LiftOver::from_file(f.path()).unwrap();
        assert!(lo.convert_coordinate("nope", 5).is_none());
        assert!(lo.convert_coordinate("src", 5).unwrap().is_empty());
    }

    #[test]
    fn overlapping_chains_all_report_sorted_by_score() {
        let f = write_chain(
            "chain 100 src 1000 + 0 20 tgtA 1000 + 0 20 1\n20\nchain 900 src 1000 + 0 20 tgtB 1000 + 100 120 2\n20\n",
        );
        let lo = LiftOver::from_file(f.path()).unwrap();
        let hits = lo.convert_coordinate("src", 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chrom, "tgtB");
        assert_eq!(hits[1].chrom, "tgtA");
    }
}
