//! Global (Needleman-Wunsch) alignment used to re-derive a variant's
//! representation against the target reference.
//!
//! Unit-cost edit distance with an explicit traceback matrix. The order in which
//! equally-scoring moves are preferred is what decides where an indel is placed,
//! so both preference orders are reproduced exactly as the Python has them:
//! `left` prefers diagonal, then up (consume reference), then left (consume
//! query); `right` prefers diagonal, then left, then up.

use crate::alleles::normalize_variant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TieBreak {
    Left,
    Right,
}

const DIAG: u8 = 0;
const UP: u8 = 1;
const LEFT: u8 = 2;

/// Align `ref_seq` against `alt_seq`, returning the two gapped strings.
pub fn global_align(ref_seq: &[u8], alt_seq: &[u8], tie_break: TieBreak) -> (Vec<u8>, Vec<u8>) {
    let n = ref_seq.len();
    let m = alt_seq.len();
    let width = m + 1;

    let mut dp = vec![0i32; (n + 1) * width];
    let mut trace = vec![DIAG; (n + 1) * width];

    for i in 1..=n {
        dp[i * width] = i as i32;
        trace[i * width] = UP;
    }
    for j in 1..=m {
        dp[j] = j as i32;
        trace[j] = LEFT;
    }

    for i in 1..=n {
        let ref_base = ref_seq[i - 1];
        for j in 1..=m {
            let alt_base = alt_seq[j - 1];
            let cost = if ref_base == alt_base { 0 } else { 1 };
            let diag = dp[(i - 1) * width + (j - 1)] + cost;
            let up = dp[(i - 1) * width + j] + 1;
            let left = dp[i * width + (j - 1)] + 1;
            let best = diag.min(up).min(left);
            dp[i * width + j] = best;

            // Preference order decides indel placement on ties.
            let mv = match tie_break {
                TieBreak::Left => {
                    if diag == best {
                        DIAG
                    } else if up == best {
                        UP
                    } else {
                        LEFT
                    }
                }
                TieBreak::Right => {
                    if diag == best {
                        DIAG
                    } else if left == best {
                        LEFT
                    } else {
                        UP
                    }
                }
            };
            trace[i * width + j] = mv;
        }
    }

    let mut aln_ref: Vec<u8> = Vec::with_capacity(n + m);
    let mut aln_alt: Vec<u8> = Vec::with_capacity(n + m);
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        match trace[i * width + j] {
            DIAG => {
                aln_ref.push(ref_seq[i - 1]);
                aln_alt.push(alt_seq[j - 1]);
                i -= 1;
                j -= 1;
            }
            UP => {
                aln_ref.push(ref_seq[i - 1]);
                aln_alt.push(b'-');
                i -= 1;
            }
            _ => {
                aln_ref.push(b'-');
                aln_alt.push(alt_seq[j - 1]);
                j -= 1;
            }
        }
    }
    aln_ref.reverse();
    aln_alt.reverse();
    (aln_ref, aln_alt)
}

/// Collapse every differing column of an alignment into a single variant.
///
/// All mismatching columns are merged into one REF/ALT pair even when they are
/// separated by matches; the position is the reference offset of the first
/// difference. Returns `None` when the sequences are identical.
pub fn variant_from_alignment(
    aln_ref: &[u8],
    aln_alt: &[u8],
    window_start: i64,
    target_seq: &[u8],
) -> Option<(i64, String, String)> {
    let mut ref_idx: i64 = 0;
    let mut first_diff: Option<i64> = None;
    let mut ref_allele: Vec<u8> = Vec::new();
    let mut alt_allele: Vec<u8> = Vec::new();

    for (&r, &a) in aln_ref.iter().zip(aln_alt.iter()) {
        if r == a {
            if r != b'-' {
                ref_idx += 1;
            }
            continue;
        }
        if first_diff.is_none() {
            first_diff = Some(ref_idx);
        }
        if r != b'-' {
            ref_allele.push(r);
        }
        if a != b'-' {
            alt_allele.push(a);
        }
        if r != b'-' {
            ref_idx += 1;
        }
    }

    let first_diff = first_diff?;
    let pos = window_start + first_diff;
    let r = String::from_utf8_lossy(&ref_allele).into_owned();
    let a = String::from_utf8_lossy(&alt_allele).into_owned();
    Some(normalize_variant(pos, &r, &a, target_seq, window_start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[u8]) -> String {
        String::from_utf8_lossy(v).into_owned()
    }

    #[test]
    fn identical_sequences_align_without_gaps() {
        let (r, a) = global_align(b"ACGT", b"ACGT", TieBreak::Left);
        assert_eq!(s(&r), "ACGT");
        assert_eq!(s(&a), "ACGT");
        assert!(variant_from_alignment(&r, &a, 100, b"ACGT").is_none());
    }

    #[test]
    fn substitution_is_a_single_diagonal_mismatch() {
        let (r, a) = global_align(b"ACGT", b"AGGT", TieBreak::Left);
        assert_eq!((s(&r).as_str(), s(&a).as_str()), ("ACGT", "AGGT"));
        let v = variant_from_alignment(&r, &a, 100, b"ACGT").unwrap();
        assert_eq!(v, (101, "C".to_string(), "G".to_string()));
    }

    #[test]
    fn empty_query_is_all_gaps() {
        let (r, a) = global_align(b"ACGT", b"", TieBreak::Left);
        assert_eq!((s(&r).as_str(), s(&a).as_str()), ("ACGT", "----"));
    }

    #[test]
    fn empty_reference_is_all_gaps() {
        let (r, a) = global_align(b"", b"ACGT", TieBreak::Left);
        assert_eq!((s(&r).as_str(), s(&a).as_str()), ("----", "ACGT"));
    }

    // A homopolymer deletion is placed identically either way: the traceback
    // starts at the far corner and the diagonal preference wins first.
    // Values verified against the Python implementation.
    #[test]
    fn homopolymer_deletion_is_tie_break_independent() {
        for tb in [TieBreak::Left, TieBreak::Right] {
            let (r, a) = global_align(b"GAAAAT", b"GAAAT", tb);
            assert_eq!((s(&r).as_str(), s(&a).as_str()), ("GAAAAT", "G-AAAT"));
        }
    }

    // Where the tie-break does bite it moves the called position, which is why
    // the realignment tries both and scores the candidates. Verified against Python.
    #[test]
    fn tie_break_changes_the_called_position() {
        let (lr, la) = global_align(b"ATATA", b"AATAA", TieBreak::Left);
        assert_eq!((s(&lr).as_str(), s(&la).as_str()), ("-ATATA", "AATA-A"));
        assert_eq!(
            variant_from_alignment(&lr, &la, 1000, b"ATATA").unwrap(),
            (1000, "T".to_string(), "A".to_string())
        );

        let (rr, ra) = global_align(b"ATATA", b"AATAA", TieBreak::Right);
        assert_eq!((s(&rr).as_str(), s(&ra).as_str()), ("ATAT-A", "A-ATAA"));
        assert_eq!(
            variant_from_alignment(&rr, &ra, 1000, b"ATATA").unwrap(),
            (1001, "T".to_string(), "A".to_string())
        );
    }

    // An alignment can normalise to REF == ALT; the realignment treats that as a
    // REF/ALT flip rather than a variant call.
    #[test]
    fn alignment_can_normalise_to_equal_alleles() {
        let (r, a) = global_align(b"CGCTCTGGT", b"CGTCTCGGT", TieBreak::Left);
        let v = variant_from_alignment(&r, &a, 1000, b"CGCTCTGGT").unwrap();
        assert_eq!(v, (1002, "T".to_string(), "T".to_string()));
    }

    #[test]
    fn multiple_differences_collapse_into_one_allele() {
        // Non-adjacent mismatches are merged, matching the Python.
        let (r, a) = global_align(b"ACGTACGT", b"AGGTACCT", TieBreak::Left);
        let v = variant_from_alignment(&r, &a, 100, b"ACGTACGT").unwrap();
        assert_eq!(v.0, 101);
        assert_eq!(v.1, "CG");
        assert_eq!(v.2, "GC");
    }
}
