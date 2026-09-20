//! Allele string manipulation shared by the liftover and realignment paths.
//!
//! These are deliberate one-for-one transliterations of the Python helpers. Where
//! a routine behaves surprisingly the behaviour is preserved rather than corrected,
//! because the Python tool's output is the reference this port is measured against.

/// Reverse complement. Bases outside `ACGT` (including `N`) are passed through
/// unchanged, matching the Python dictionary lookup with a default.
pub fn rev_comp(seq: &str) -> String {
    seq.bytes().rev().map(|b| complement_base(b) as char).collect()
}

fn complement_base(b: u8) -> u8 {
    match b {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        other => other,
    }
}

/// Trim the maximal shared suffix while retaining a valid allele base.
pub fn trim_identical_suffix(ref_allele: &str, alt_allele: &str) -> (String, String) {
    let mut r = ref_allele.as_bytes().to_vec();
    let mut a = alt_allele.as_bytes().to_vec();
    while r.len() > 1 && a.len() > 1 && r[r.len() - 1] == a[a.len() - 1] {
        r.pop();
        a.pop();
    }
    (to_string(&r), to_string(&a))
}

/// Trim a shared prefix and then a shared suffix, always leaving at least one base
/// on each side.
pub fn trim_common(ref_allele: &str, alt_allele: &str) -> (String, String) {
    let mut r: &[u8] = ref_allele.as_bytes();
    let mut a: &[u8] = alt_allele.as_bytes();
    while r.len() > 1 && a.len() > 1 && r[0] == a[0] {
        r = &r[1..];
        a = &a[1..];
    }
    while r.len() > 1 && a.len() > 1 && r[r.len() - 1] == a[a.len() - 1] {
        r = &r[..r.len() - 1];
        a = &a[..a.len() - 1];
    }
    (to_string(r), to_string(a))
}

/// Positions within `ref_allele` that the alternate allele edits.
///
/// For equal-length alleles this is the set of mismatching offsets; otherwise the
/// Python returns `range(1, len(ref))`, treating base 0 as the shared anchor.
pub fn variant_edit_positions(ref_allele: &str, alt_allele: &str) -> Vec<usize> {
    if ref_allele.len() == alt_allele.len() {
        ref_allele
            .bytes()
            .zip(alt_allele.bytes())
            .enumerate()
            .filter(|(_, (r, a))| r != a)
            .map(|(i, _)| i)
            .collect()
    } else {
        (1..ref_allele.len()).collect()
    }
}

/// Shift an indel left through a homopolymer/repeat run within the window.
///
/// Note the deletion branch appends to ALT without shortening it, so repeated
/// shifts can equalise the allele lengths and fall through to the insertion
/// branch. That is the Python behaviour and is reproduced exactly.
pub fn left_shift_variant(
    mut pos: i64,
    ref_allele: &str,
    alt_allele: &str,
    target_seq: &[u8],
    window_start: i64,
) -> (i64, String, String) {
    if ref_allele.len() == alt_allele.len() {
        return (pos, ref_allele.to_string(), alt_allele.to_string());
    }
    let mut r = ref_allele.as_bytes().to_vec();
    let mut a = alt_allele.as_bytes().to_vec();
    while pos > window_start {
        let idx = (pos - window_start - 1) as usize;
        let prev_base = target_seq[idx];
        if r.len() > a.len() {
            if *r.last().unwrap() != prev_base {
                break;
            }
            let mut new_r = vec![prev_base];
            new_r.extend_from_slice(&r[..r.len() - 1]);
            r = new_r;
            let mut new_a = vec![prev_base];
            new_a.extend_from_slice(&a);
            a = new_a;
        } else {
            if *a.last().unwrap() != prev_base {
                break;
            }
            let mut new_r = vec![prev_base];
            new_r.extend_from_slice(&r);
            r = new_r;
            let mut new_a = vec![prev_base];
            new_a.extend_from_slice(&a[..a.len() - 1]);
            a = new_a;
        }
        pos -= 1;
    }
    (pos, to_string(&r), to_string(&a))
}

/// Anchor an empty allele, trim, left-shift, then trim again.
pub fn normalize_variant(
    mut pos: i64,
    ref_allele: &str,
    alt_allele: &str,
    target_seq: &[u8],
    window_start: i64,
) -> (i64, String, String) {
    let mut r = ref_allele.to_string();
    let mut a = alt_allele.to_string();

    if r.is_empty() || a.is_empty() {
        let anchor = if pos == window_start {
            target_seq[(pos - window_start) as usize]
        } else {
            let base = target_seq[(pos - window_start - 1) as usize];
            pos -= 1;
            base
        } as char;
        if r.is_empty() {
            r = anchor.to_string();
            a = format!("{anchor}{a}");
        } else {
            a = anchor.to_string();
            r = format!("{anchor}{r}");
        }
    }

    let (r2, a2) = trim_common(&r, &a);
    let (pos, r3, a3) = left_shift_variant(pos, &r2, &a2, target_seq, window_start);
    let (r4, a4) = trim_common(&r3, &a3);
    (pos, r4, a4)
}

fn to_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rev_comp_passes_through_unknown_bases() {
        assert_eq!(rev_comp("ACGT"), "ACGT");
        assert_eq!(rev_comp("AACC"), "GGTT");
        assert_eq!(rev_comp("ANG"), "CNT");
    }

    // The two regressions recorded in CHANGELOG v1.0.1.
    #[test]
    fn trim_identical_suffix_keeps_one_base_and_difference() {
        assert_eq!(
            trim_identical_suffix("AC", "CC"),
            ("A".to_string(), "C".to_string())
        );
        assert_eq!(
            trim_identical_suffix("CAT", "TAT"),
            ("C".to_string(), "T".to_string())
        );
    }

    #[test]
    fn trim_identical_suffix_never_empties_an_allele() {
        assert_eq!(
            trim_identical_suffix("AA", "AA"),
            ("A".to_string(), "A".to_string())
        );
    }

    #[test]
    fn trim_common_strips_prefix_then_suffix() {
        assert_eq!(
            trim_common("GACT", "GTCT"),
            ("A".to_string(), "T".to_string())
        );
        assert_eq!(
            trim_common("GA", "GAT"),
            ("A".to_string(), "AT".to_string())
        );
    }

    #[test]
    fn variant_edit_positions_matches_python_branches() {
        assert_eq!(variant_edit_positions("ACGT", "AGGT"), vec![1]);
        // Unequal lengths: every base after the anchor.
        assert_eq!(variant_edit_positions("ACGT", "A"), vec![1, 2, 3]);
        assert_eq!(variant_edit_positions("A", "ACGT"), Vec::<usize>::new());
    }

    #[test]
    fn normalize_anchors_an_empty_alt() {
        // Window "GTTTTA" at 100; deleting the T at 102 with an empty ALT.
        let seq = b"GTTTTA";
        let (pos, r, a) = normalize_variant(102, "T", "", seq, 100);
        // Anchored on the preceding base, then shifted left through the T run.
        assert_eq!(&seq[(pos - 100) as usize..(pos - 100) as usize + r.len()], r.as_bytes());
        assert!(r.len() > a.len());
    }

    #[test]
    fn normalize_leaves_a_snv_in_place() {
        let seq = b"ACGTACGT";
        let (pos, r, a) = normalize_variant(102, "G", "T", seq, 100);
        assert_eq!((pos, r.as_str(), a.as_str()), (102, "G", "T"));
    }

    #[test]
    fn left_shift_is_a_noop_for_equal_lengths() {
        let seq = b"AAAAAA";
        assert_eq!(
            left_shift_variant(103, "A", "C", seq, 100),
            (103, "A".to_string(), "C".to_string())
        );
    }
}
