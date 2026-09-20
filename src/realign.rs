//! Haplotype realignment for variants that sit *near* an assembly difference
//! without overlapping it.
//!
//! When an indel lands close to a place where the two assemblies disagree, its
//! lifted representation can be wrong even though nothing overlaps. The source
//! haplotype is rebuilt over a local window, the variant's ALT is substituted
//! into it, and that haplotype is globally aligned back to the target reference
//! to recover the correct position and alleles.

use crate::align::{global_align, variant_from_alignment, TieBreak};
use crate::error::{Result, VariantError};
use crate::fasta::Genome;
use crate::refdiff::{RefDiff, RefDiffIndex};

#[derive(Debug, Clone, Copy)]
pub struct RealignConfig {
    pub enabled: bool,
    /// Emit the same per-decision traces the Python writes under --debug.
    pub debug: bool,
    /// How far from the variant to look for a reference difference.
    pub distance: i64,
    /// Bases added either side of the realignment window.
    pub flank: i64,
    /// Hard cap on the window, which bounds the alignment cost.
    pub max_window: i64,
}

impl Default for RealignConfig {
    fn default() -> Self {
        RealignConfig {
            enabled: true,
            debug: false,
            distance: 50,
            flank: 20,
            max_window: 200,
        }
    }
}

/// Reconstruct the source-assembly sequence across a window, plus a map from
/// target coordinate to offset within that reconstruction.
///
/// Only the first `len(REF)` bases of each difference's ALT get a mapping entry,
/// so a target base consumed by an insertion maps to the inserted sequence.
pub fn build_source_reference(
    target_seq: &[u8],
    window_start: i64,
    window_end: i64,
    diffs: &[&RefDiff],
) -> Result<(Vec<u8>, std::collections::HashMap<i64, usize>)> {
    let mut source: Vec<u8> = Vec::new();
    let mut target_to_source = std::collections::HashMap::new();
    let mut cursor = window_start;

    for diff in diffs {
        if diff.start < cursor {
            return Err(VariantError::Unliftable("overlapping ref diffs in window"));
        }
        for pos in cursor..diff.start {
            target_to_source.insert(pos, source.len());
            source.push(target_seq[(pos - window_start) as usize]);
        }
        let ref_len = diff.ref_allele.len();
        for (i, base) in diff.alt_allele.bytes().enumerate() {
            if i < ref_len {
                target_to_source.insert(diff.start + i as i64, source.len());
            }
            source.push(base);
        }
        cursor = diff.end;
    }
    for pos in cursor..window_end {
        target_to_source.insert(pos, source.len());
        source.push(target_seq[(pos - window_start) as usize]);
    }
    Ok((source, target_to_source))
}

/// `seq[a:b]` with Python's clamping.
fn pslice(seq: &[u8], a: usize, b: usize) -> &[u8] {
    let len = seq.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    &seq[a..b]
}

/// Candidate placement, ordered exactly as the Python tuple is: by distance to
/// the difference, then offset, then position, alleles, and finally the label
/// (`left` < `right` < `shift`).
type Candidate = (i64, i64, i64, String, String, &'static str);

fn make_candidate(
    pos: i64,
    r: &str,
    a: &str,
    diff: &RefDiff,
    label: &'static str,
) -> Candidate {
    let var_start = pos;
    let var_end = pos + r.len() as i64;
    let dist = if var_end <= diff.start {
        diff.start - var_end
    } else if diff.end <= var_start {
        var_start - diff.end
    } else {
        0
    };
    let offset = (pos - diff.start).abs();
    (dist, offset, pos, r.to_string(), a.to_string(), label)
}

/// Try to re-derive a variant's representation near a reference difference.
///
/// Returns `None` when realignment does not apply or finds nothing better; the
/// caller then keeps the variant as lifted.
#[allow(clippy::too_many_arguments)]
pub fn attempt_haplotype_realignment(
    var_ref: &str,
    var_alt: &str,
    chrom: &str,
    span_start: i64,
    span_end: i64,
    ref_diffs: &RefDiffIndex,
    genome: &Genome,
    cfg: &RealignConfig,
) -> Result<Option<(i64, String, String)>> {
    if !cfg.enabled {
        return Ok(None);
    }
    // Only indels are realigned; a substitution's position is unambiguous.
    if var_ref.len() == var_alt.len() {
        return Ok(None);
    }
    let Some(contig_seq) = genome.get(chrom) else {
        return Ok(None);
    };

    let search_start = (span_start - cfg.distance).max(0);
    let search_end = (span_end + cfg.distance).min(contig_seq.len() as i64);
    let nearby = ref_diffs.overlapping(chrom, search_start, search_end);
    let indels: Vec<&RefDiff> = nearby.into_iter().filter(|d| d.is_indel()).collect();
    if indels.is_empty() {
        return Ok(None);
    }

    // Nearest indel difference; ties resolve to the earliest by (start, end),
    // which is the order `overlapping` returns.
    let diff = indels
        .iter()
        .copied()
        .min_by_key(|d| d.distance_to_span(span_start, span_end))
        .expect("non-empty");
    if diff.distance_to_span(span_start, span_end) > cfg.distance {
        return Ok(None);
    }

    // Shrink the flank until the window is small enough, still contains both the
    // variant and the difference, and holds no *other* difference.
    let target_len = contig_seq.len() as i64;
    let mut flank = cfg.flank;
    let mut window: Option<(i64, i64)> = None;
    while flank >= 0 {
        let window_start = (span_start.min(diff.start) - flank).max(0);
        let window_end = (span_end.max(diff.end) + flank).min(target_len);
        let fits = window_end - window_start <= cfg.max_window
            && window_start <= span_start
            && span_start < window_end
            && window_start < span_end
            && span_end <= window_end
            && window_start <= diff.start
            && diff.end <= window_end;
        if fits {
            let in_window = ref_diffs.overlapping(chrom, window_start, window_end);
            let has_other = in_window.iter().any(|x| {
                !(x.start == diff.start
                    && x.end == diff.end
                    && x.ref_allele == diff.ref_allele
                    && x.alt_allele == diff.alt_allele)
            });
            window = Some((window_start, window_end));
            if !has_other {
                break;
            }
        }
        flank -= 1;
    }
    // `flank < 0` means every candidate window still held another difference.
    if flank < 0 {
        return Ok(None);
    }
    let Some((window_start, window_end)) = window else {
        return Ok(None);
    };

    let target_seq = &contig_seq[window_start as usize..window_end as usize];
    let (source_ref, target_to_source) =
        build_source_reference(target_seq, window_start, window_end, &[diff])?;

    let Some(&source_idx) = target_to_source.get(&span_start) else {
        if cfg.debug {
            eprintln!(
                "realign: no target_to_source for {chrom}:{span_start} in window {window_start}-{window_end}"
            );
        }
        return Ok(None);
    };
    if pslice(&source_ref, source_idx, source_idx + var_ref.len()) != var_ref.as_bytes() {
        if cfg.debug {
            eprintln!(
                "realign: source ref mismatch at {chrom}:{span_start} ({} != {var_ref})",
                String::from_utf8_lossy(pslice(&source_ref, source_idx, source_idx + var_ref.len()))
            );
        }
        return Ok(None);
    }

    // Substitute the variant's ALT into the reconstructed source haplotype.
    let mut source_alt: Vec<u8> = Vec::with_capacity(source_ref.len() + var_alt.len());
    source_alt.extend_from_slice(&source_ref[..source_idx]);
    source_alt.extend_from_slice(var_alt.as_bytes());
    source_alt.extend_from_slice(&source_ref[(source_idx + var_ref.len()).min(source_ref.len())..]);

    let mut best: Option<Candidate> = None;
    for (label, tb) in [("left", TieBreak::Left), ("right", TieBreak::Right)] {
        let (aln_ref, aln_alt) = global_align(target_seq, &source_alt, tb);
        let Some((pos, r, a)) = variant_from_alignment(&aln_ref, &aln_alt, window_start, target_seq)
        else {
            continue;
        };
        if r == a {
            continue;
        }
        let off = (pos - window_start) as usize;
        if pslice(target_seq, off, off + r.len()) != r.as_bytes() {
            continue;
        }
        let cand = make_candidate(pos, &r, &a, diff, label);
        if best.as_ref().map_or(true, |b| cand < *b) {
            best = Some(cand);
        }
    }

    // An insertion of a single repeated base next to a deletion difference is
    // ambiguous across the run; propose the placement just past the difference.
    let ins_bases = &var_alt.as_bytes()[var_ref.len().min(var_alt.len())..];
    let extra = &diff.ref_allele.as_bytes()
        [diff.alt_allele.len().min(diff.ref_allele.len())..];
    let uniform = ins_bases.is_empty() || ins_bases.iter().all(|b| *b == ins_bases[0]);
    if uniform && !ins_bases.is_empty() {
        let base = ins_bases[0];
        let k = ins_bases.len();
        if var_ref.len() < var_alt.len()
            && extra.first() == Some(&base)
            && k > 0
            && k <= extra.len()
        {
            let run_start = span_start + 1;
            let run_end = diff.start;
            let run_uniform = run_end <= run_start
                || contig_seq[run_start as usize..run_end as usize]
                    .iter()
                    .all(|b| *b == base);
            if run_uniform {
                let pos = diff.start + k as i64;
                let r = String::from_utf8_lossy(&extra[k - 1..]).into_owned();
                let a = (base as char).to_string();
                let off = (pos - window_start) as usize;
                if r != a && pslice(target_seq, off, off + r.len()) == r.as_bytes() {
                    let cand = make_candidate(pos, &r, &a, diff, "shift");
                    if best.as_ref().map_or(true, |b| cand < *b) {
                        best = Some(cand);
                    }
                }
            }
        }
    }

    match best {
        None => {
            if cfg.debug {
                eprintln!("realign: no variant from alignment for {chrom}:{span_start}");
            }
            Ok(None)
        }
        Some((_, _, pos, r, a, label)) => {
            if cfg.debug {
                eprintln!(
                    "realign({label}): {chrom}:{span_start} -> {pos} {r}/{a} window {window_start}-{window_end}"
                );
            }
            Ok(Some((pos, r, a)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn diff(start: i64, r: &str, a: &str) -> RefDiff {
        RefDiff {
            start,
            end: start + r.len() as i64,
            ref_allele: r.to_string(),
            alt_allele: a.to_string(),
        }
    }

    #[test]
    fn source_reference_without_differences_is_the_target() {
        let target = b"ACGTACGT";
        let (src, map) = build_source_reference(target, 100, 108, &[]).unwrap();
        assert_eq!(src, target.to_vec());
        assert_eq!(map[&100], 0);
        assert_eq!(map[&107], 7);
    }

    #[test]
    fn a_deletion_difference_shortens_the_source() {
        // Target ACGT at 102..106; the source has only A there.
        let target = b"TTACGTTT";
        let d = diff(102, "ACGT", "A");
        let (src, map) = build_source_reference(target, 100, 108, &[&d]).unwrap();
        assert_eq!(String::from_utf8_lossy(&src), "TTATT");
        // Only the first base of the difference gets a mapping.
        assert_eq!(map[&102], 2);
        assert!(!map.contains_key(&103));
        assert_eq!(map[&106], 3);
    }

    #[test]
    fn an_insertion_difference_lengthens_the_source() {
        // Target has a single A at 102 where the source has ACGT.
        let target = b"TTATTTTT";
        let d = diff(102, "A", "ACGT");
        let (src, map) = build_source_reference(target, 100, 108, &[&d]).unwrap();
        assert_eq!(String::from_utf8_lossy(&src), "TTACGTTTTTT");
        assert_eq!(map[&102], 2);
        assert_eq!(map[&103], 6);
    }

    #[test]
    fn out_of_order_differences_are_rejected() {
        let target = b"ACGTACGT";
        let a = diff(104, "A", "C");
        let b = diff(100, "A", "C");
        let err = build_source_reference(target, 100, 108, &[&a, &b]).unwrap_err();
        assert_eq!(
            err,
            VariantError::Unliftable("overlapping ref diffs in window")
        );
    }

    fn genome_with(seq: &[u8]) -> Genome {
        let mut g = HashMap::new();
        g.insert("chr1".to_string(), seq.to_vec());
        g
    }

    #[test]
    fn substitutions_are_never_realigned() {
        let g = genome_with(b"ACGTACGTAC");
        let idx = RefDiffIndex::build(HashMap::new());
        let got = attempt_haplotype_realignment(
            "A",
            "C",
            "chr1",
            2,
            3,
            &idx,
            &g,
            &RealignConfig::default(),
        )
        .unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn realignment_is_skipped_when_disabled() {
        let g = genome_with(b"ACGTACGTAC");
        let mut m = HashMap::new();
        m.insert("chr1".to_string(), vec![diff(5, "AC", "A")]);
        let idx = RefDiffIndex::build(m);
        let cfg = RealignConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(attempt_haplotype_realignment("AC", "A", "chr1", 1, 3, &idx, &g, &cfg)
            .unwrap()
            .is_none());
    }

    #[test]
    fn no_nearby_indel_difference_means_no_realignment() {
        let g = genome_with(b"ACGTACGTAC");
        let mut m = HashMap::new();
        // A substitution difference does not trigger realignment.
        m.insert("chr1".to_string(), vec![diff(5, "A", "C")]);
        let idx = RefDiffIndex::build(m);
        assert!(attempt_haplotype_realignment(
            "AC",
            "A",
            "chr1",
            1,
            3,
            &idx,
            &g,
            &RealignConfig::default()
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn candidate_ordering_prefers_closer_then_left() {
        let d = diff(100, "AC", "A");
        let near = make_candidate(98, "A", "AT", &d, "left");
        let far = make_candidate(80, "A", "AT", &d, "left");
        assert!(near < far);
        let left = make_candidate(98, "A", "AT", &d, "left");
        let right = make_candidate(98, "A", "AT", &d, "right");
        let shift = make_candidate(98, "A", "AT", &d, "shift");
        assert!(left < right && right < shift);
    }
}
