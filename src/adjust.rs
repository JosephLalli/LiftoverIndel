//! Rewrite a lifted variant's alleles when it overlaps a single assembly difference.
//!
//! The variant arrives in source-allele form at target coordinates; the overlapping
//! reference difference tells us how the two assemblies disagree over that span, and
//! this substitutes the target sequence into REF while carrying the variant's own
//! edit across into ALT.
//!
//! Python slice expressions clamp silently rather than panicking, so the helpers
//! below reproduce that clamping; several branches rely on it.

use crate::alleles::variant_edit_positions;
use crate::error::{Result, VariantError};
use crate::refdiff::RefDiff;

/// `s[a:b]` with Python's clamping (indices are byte offsets; alleles are ASCII).
fn pslice(s: &str, a: usize, b: usize) -> &str {
    let len = s.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    &s[a..b]
}

/// `s[a:]`
fn pfrom(s: &str, a: usize) -> &str {
    &s[a.min(s.len())..]
}

/// `s[:b]`
fn pto(s: &str, b: usize) -> &str {
    &s[..b.min(s.len())]
}

/// Drop a shared trailing base while both alleles keep more than one.
fn trim_suffix(mut r: String, mut a: String) -> (String, String) {
    while r.len() > 1 && a.len() > 1 && r.as_bytes()[r.len() - 1] == a.as_bytes()[a.len() - 1] {
        r.pop();
        a.pop();
    }
    (r, a)
}

/// Adjust a variant's REF/ALT for the assembly difference it overlaps.
///
/// `lifted_start` and `lifted_end` are the target coordinates the variant's start
/// and end mapped to; they arrive unordered because a minus-strand chain swaps them.
pub fn compute_adjusted_ref_alt(
    var_ref: &str,
    var_alt: &str,
    diff: &RefDiff,
    lifted_start: i64,
    lifted_end: i64,
) -> Result<(String, String)> {
    let start = lifted_start.min(lifted_end);
    let end = lifted_start.max(lifted_end);
    let overlap_start = start.max(diff.start);
    let overlap_end = end.min(diff.end);
    if overlap_start >= overlap_end {
        return Ok((var_ref.to_string(), var_alt.to_string()));
    }

    let ref_len = diff.ref_allele.len();
    let alt_len = diff.alt_allele.len();

    // A variant starting inside an insertion relative to the source has no
    // unambiguous target representation.
    if diff.start < lifted_start && lifted_start < diff.end && ref_len < alt_len {
        return Err(VariantError::Unliftable("start inside deletion ref diff"));
    }

    let rel_start = (overlap_start - diff.start) as usize;
    let rel_end = (overlap_end - diff.start) as usize;

    let target_sub = pslice(&diff.ref_allele, rel_start, rel_end).to_string();
    let source_sub = if ref_len > alt_len {
        let src_start = rel_start;
        let src_end = rel_end.min(alt_len);
        if src_start < src_end {
            pslice(&diff.alt_allele, src_start, src_end).to_string()
        } else {
            String::new()
        }
    } else {
        pslice(&diff.alt_allele, rel_start, rel_end).to_string()
    };

    let offset = (overlap_start - start) as usize;

    // The variant begins exactly at an indel difference and its REF opens with the
    // source-side allele: splice the target-side allele in and keep the rest.
    if ref_len != alt_len && offset == 0 && var_ref.starts_with(&diff.alt_allele) {
        let prefix = &diff.alt_allele;
        let suffix = pfrom(var_ref, prefix.len());
        let extra = pfrom(&diff.ref_allele, prefix.len());
        let new_ref = format!("{}{}", diff.ref_allele, suffix);
        let new_alt = if var_ref.len() == var_alt.len() {
            if var_alt.len() < prefix.len() {
                return Err(VariantError::Unliftable(
                    "variant alt shorter than ref diff prefix",
                ));
            }
            format!(
                "{}{}{}",
                pto(var_alt, prefix.len()),
                extra,
                pfrom(var_alt, prefix.len())
            )
        } else {
            var_alt.to_string()
        };
        return Ok(trim_suffix(new_ref, new_alt));
    }

    if !source_sub.is_empty() {
        if offset + source_sub.len() > var_ref.len() {
            return Err(VariantError::Unliftable("ref diff out of bounds"));
        }
        if pslice(var_ref, offset, offset + source_sub.len()) != source_sub {
            return Err(VariantError::Unliftable("source ref mismatch"));
        }
    }

    let new_ref = format!(
        "{}{}{}",
        pto(var_ref, offset),
        target_sub,
        pfrom(var_ref, offset + source_sub.len())
    );

    let new_alt = if ref_len != alt_len {
        if var_ref.len() != var_alt.len() {
            // Both the variant and the assembly difference are indels: only
            // liftable when the variant's own edit avoids the difference.
            let edits = variant_edit_positions(var_ref, var_alt);
            if edits
                .iter()
                .any(|&p| p >= offset && p < offset + source_sub.len())
            {
                return Err(VariantError::Unliftable(
                    "variant edits overlap indel ref diff",
                ));
            }
            if !source_sub.is_empty()
                && pslice(var_alt, offset, offset + source_sub.len()) != source_sub
            {
                return Err(VariantError::Unliftable("source alt mismatch in ref diff"));
            }
            format!(
                "{}{}{}",
                pto(var_alt, offset),
                target_sub,
                pfrom(var_alt, offset + source_sub.len())
            )
        } else if !source_sub.is_empty() {
            // Substitution over an indel difference: keep the variant's own
            // substituted bases, take the target sequence elsewhere.
            let mut replacement: Vec<u8> = target_sub.as_bytes().to_vec();
            let n = source_sub.len().min(replacement.len());
            for i in 0..n {
                let src_i = offset + i;
                if src_i >= var_alt.len() || src_i >= var_ref.len() {
                    return Err(VariantError::Unliftable("ref diff out of bounds"));
                }
                if var_alt.as_bytes()[src_i] != var_ref.as_bytes()[src_i] {
                    replacement[i] = var_alt.as_bytes()[src_i];
                }
            }
            format!(
                "{}{}{}",
                pto(var_alt, offset),
                String::from_utf8_lossy(&replacement),
                pfrom(var_alt, offset + source_sub.len())
            )
        } else {
            // Nothing consumed on the source side: insert the target sequence.
            format!(
                "{}{}{}",
                pto(var_alt, offset),
                target_sub,
                pfrom(var_alt, offset)
            )
        }
    } else {
        // Equal-length difference: copy target bases in wherever the variant did
        // not itself edit the base.
        let mut alt_list: Vec<u8> = var_alt.as_bytes().to_vec();
        for (i, &tgt_base) in target_sub.as_bytes().iter().enumerate() {
            let src_i = offset + i;
            if src_i >= var_ref.len() {
                return Err(VariantError::Unliftable("ref diff out of bounds"));
            }
            if src_i < var_alt.len() && var_alt.as_bytes()[src_i] == var_ref.as_bytes()[src_i] {
                alt_list[src_i] = tgt_base;
            }
        }
        String::from_utf8_lossy(&alt_list).into_owned()
    };

    Ok(trim_suffix(new_ref, new_alt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(start: i64, r: &str, a: &str) -> RefDiff {
        RefDiff {
            start,
            end: start + r.len() as i64,
            ref_allele: r.to_string(),
            alt_allele: a.to_string(),
        }
    }

    #[test]
    fn no_overlap_returns_alleles_unchanged() {
        let diff = d(100, "A", "G");
        let got = compute_adjusted_ref_alt("C", "T", &diff, 200, 201).unwrap();
        assert_eq!(got, ("C".to_string(), "T".to_string()));
    }

    #[test]
    fn substitution_over_a_snv_difference_takes_the_target_base() {
        // Target has A at 100 where the source has G; the variant is G>T there.
        let diff = d(100, "A", "G");
        let got = compute_adjusted_ref_alt("G", "T", &diff, 100, 101).unwrap();
        // REF becomes the target base; the variant's own edit is preserved.
        assert_eq!(got, ("A".to_string(), "T".to_string()));
    }

    #[test]
    fn a_variant_matching_the_difference_collapses_to_equal_alleles() {
        // Source G, target A, and the variant says G>A: after lifting, REF == ALT,
        // which the caller turns into a REF/ALT flip.
        let diff = d(100, "A", "G");
        let got = compute_adjusted_ref_alt("G", "A", &diff, 100, 101).unwrap();
        assert_eq!(got.0, got.1);
    }

    #[test]
    fn source_ref_mismatch_is_unliftable() {
        // The difference says the source base is G, but the variant claims C.
        let diff = d(100, "A", "G");
        let err = compute_adjusted_ref_alt("C", "T", &diff, 100, 101).unwrap_err();
        assert_eq!(err, VariantError::Unliftable("source ref mismatch"));
    }

    // Expected values below were taken from the Python implementation.
    #[test]
    fn variant_edit_inside_an_indel_difference_is_unliftable() {
        // An equal-length variant whose edits land inside the difference span.
        let diff = d(100, "TG", "TTT");
        let err = compute_adjusted_ref_alt("TTT", "CGAGC", &diff, 102, 99).unwrap_err();
        assert_eq!(
            err,
            VariantError::Unliftable("variant edits overlap indel ref diff")
        );
    }

    #[test]
    fn a_variant_spanning_a_deletion_difference_absorbs_the_extra_bases() {
        // Source A, target ACGT. A deletion starting one base in keeps its own
        // ALT and grows REF by the bases the target has and the source lacks.
        let diff = d(100, "ACGT", "A");
        let got = compute_adjusted_ref_alt("ACGT", "A", &diff, 101, 105).unwrap();
        assert_eq!(got, ("ACGTCGT".to_string(), "A".to_string()));
    }

    #[test]
    fn indel_difference_at_offset_zero_splices_then_trims() {
        // The target's extra CGT is spliced into both alleles and then trimmed
        // back off as a shared suffix, leaving the original SNV.
        let diff = d(100, "ACGT", "A");
        let got = compute_adjusted_ref_alt("A", "T", &diff, 100, 101).unwrap();
        assert_eq!(got, ("A".to_string(), "T".to_string()));
    }

    #[test]
    fn trailing_shared_bases_are_trimmed_but_one_is_kept() {
        assert_eq!(
            trim_suffix("ACGT".into(), "TCGT".into()),
            ("A".to_string(), "T".to_string())
        );
        assert_eq!(
            trim_suffix("AA".into(), "AA".into()),
            ("A".to_string(), "A".to_string())
        );
    }

    #[test]
    fn python_slice_clamping_is_reproduced() {
        assert_eq!(pslice("ABC", 1, 99), "BC");
        assert_eq!(pslice("ABC", 5, 9), "");
        assert_eq!(pfrom("ABC", 99), "");
        assert_eq!(pto("ABC", 99), "ABC");
        // b < a yields empty, as Python does.
        assert_eq!(pslice("ABC", 2, 1), "");
    }
}
