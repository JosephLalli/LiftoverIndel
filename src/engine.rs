//! The liftover engine: everything needed to move one variant between assemblies.
//!
//! This is the part of the tool that has nothing to do with VCF records. The
//! binary drives it over a BCF; the C API in [`crate::ffi`] drives it one variant
//! at a time from another language.
//!
//! What it does *not* do is genotypes. Deciding that a site's REF and ALT have
//! swapped is an allele-level fact and is reported as [`Lift::flipped`]; rewriting
//! the sample genotypes to match needs every record at that position, which only
//! the caller has. The same goes for the rule that a site may flip at most once —
//! `already_flipped` is threaded in by the caller.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use rust_htslib::bcf::{Read, Reader};

use crate::adjust::compute_adjusted_ref_alt;
use crate::alleles::{rev_comp, trim_identical_suffix};
use crate::chain::LiftOver;
use crate::error::{Result, VariantError};
use crate::fasta::{load_target_genome, Genome};
use crate::realign::{attempt_haplotype_realignment, RealignConfig};
use crate::refdiff::{RefDiff, RefDiffIndex};

/// `seq[a:b]` with Python's clamping.
pub(crate) fn pslice(seq: &[u8], a: usize, b: usize) -> &[u8] {
    let len = seq.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    &seq[a..b]
}

/// cyvcf2's `is_snp`: a single reference base and an unambiguous single ALT base.
/// The reference base itself is not checked, so `N` -> `G` counts as a SNP.
pub fn is_snp(r: &str, a: &str) -> bool {
    r.len() == 1 && matches!(a, "A" | "C" | "G" | "T")
}

/// The alleles and coordinates of a variant as the liftover works on them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarState {
    pub chrom: String,
    /// 0-based start.
    pub start: i64,
    pub ref_allele: String,
    pub alt_allele: String,
}

impl VarState {
    pub fn end(&self) -> i64 {
        self.start + self.ref_allele.len() as i64
    }
}

/// Verify a lifted allele against the target reference.
pub fn check_var_ref(v: &VarState, genome: &Genome) -> Result<()> {
    let Some(seq) = genome.get(&v.chrom) else {
        return Err(VariantError::Mismatch("contig absent from target reference"));
    };
    let target_ref = pslice(seq, v.start.max(0) as usize, v.end().max(0) as usize);
    if target_ref.iter().any(|b| !matches!(b, b'A' | b'C' | b'G' | b'T')) {
        return Err(VariantError::Unliftable(
            "degenerate base in target reference",
        ));
    }
    if v.ref_allele.as_bytes() == target_ref {
        Ok(())
    } else {
        Err(VariantError::Mismatch("lifted REF does not match target"))
    }
}

/// Coordinate liftover plus reverse-strand allele handling.
///
/// Returns the target coordinates of the variant's start and end, or `None` when
/// either endpoint fails to map uniquely. `v` is updated in place.
pub fn perform_clean_liftover(
    v: &mut VarState,
    lo: &LiftOver,
    genome: &Genome,
) -> Option<((String, i64), (String, i64))> {
    let (r, a) = trim_identical_suffix(&v.ref_allele, &v.alt_allele);
    v.ref_allele = r;
    v.alt_allele = a;

    let start_hits = lo.convert_coordinate(&v.chrom, v.start)?;
    let end_hits = lo.convert_coordinate(&v.chrom, v.end())?;
    if start_hits.len() != 1 || end_hits.len() != 1 {
        return None;
    }
    let mut new_start = (
        start_hits[0].chrom.clone(),
        start_hits[0].pos,
        start_hits[0].strand,
    );
    let mut new_end = (end_hits[0].chrom.clone(), end_hits[0].pos, end_hits[0].strand);

    if new_start.2 == '-' {
        std::mem::swap(&mut new_start, &mut new_end);
        if !is_snp(&v.ref_allele, &v.alt_allele) {
            // The anchor base is read from the target; the rest is complemented.
            let seq = genome.get(&new_start.0)?;
            let idx = usize::try_from(new_start.1).ok()?;
            let base = *seq.get(idx)? as char;
            v.ref_allele = format!("{base}{}", rev_comp(&v.ref_allele[1..]));
            v.alt_allele = format!("{base}{}", rev_comp(&v.alt_allele[1..]));
        } else {
            new_start.1 += 1;
            new_end = (new_start.0.clone(), new_end.1 + 1, new_end.2);
            v.ref_allele = rev_comp(&v.ref_allele);
            v.alt_allele = rev_comp(&v.alt_allele);
        }
    }

    v.chrom = new_start.0.clone();
    v.start = new_start.1;
    Some(((new_start.0, new_start.1), (new_end.0, new_end.1)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    pub flip: bool,
    pub realigned: bool,
}

/// A site may only flip once. The Python raises `AssertionError` here, which its
/// handler catches alongside `ValueError`, so a second flip is diverted to the
/// ref-mismatch sidecar rather than aborting the run. The check happens before
/// the reference is validated, so it takes precedence over a reference failure.
fn guard_single_flip(already_flipped: bool) -> Result<()> {
    if already_flipped {
        Err(VariantError::Mismatch("double flip at site"))
    } else {
        Ok(())
    }
}

/// Resolve a variant whose lifted span touches no assembly difference: try
/// haplotype realignment, then validate against the target reference.
#[allow(clippy::too_many_arguments)]
pub fn resolve_without_overlap(
    state: &mut VarState,
    chrom: &str,
    span_start: i64,
    span_end: i64,
    ref_diffs: &RefDiffIndex,
    genome: &Genome,
    realign: &RealignConfig,
    already_flipped: bool,
) -> Result<Resolution> {
    let mut flip = false;
    let mut realigned = false;
    let attempt = attempt_haplotype_realignment(
        &state.ref_allele,
        &state.alt_allele,
        chrom,
        span_start,
        span_end,
        ref_diffs,
        genome,
        realign,
    )?;
    if let Some((pos, r, a)) = attempt {
        state.start = pos;
        if r == a {
            guard_single_flip(already_flipped)?;
            flip = true;
            std::mem::swap(&mut state.ref_allele, &mut state.alt_allele);
        } else {
            state.ref_allele = r;
            state.alt_allele = a;
        }
        realigned = true;
    }
    check_var_ref(state, genome)?;
    Ok(Resolution { flip, realigned })
}

/// Resolve a variant whose lifted span overlaps exactly one assembly difference.
pub fn resolve_with_overlap(
    state: &mut VarState,
    diff: &RefDiff,
    lifted_start: i64,
    lifted_end: i64,
    genome: &Genome,
    already_flipped: bool,
) -> Result<Resolution> {
    let (new_ref, new_alt) = compute_adjusted_ref_alt(
        &state.ref_allele,
        &state.alt_allele,
        diff,
        lifted_start,
        lifted_end,
    )?;
    let mut flip = false;
    if new_ref == new_alt {
        guard_single_flip(already_flipped)?;
        flip = true;
        std::mem::swap(&mut state.ref_allele, &mut state.alt_allele);
    } else {
        state.ref_allele = new_ref;
        state.alt_allele = new_alt;
    }
    // The Python asserts this before checking the reference.
    if state.ref_allele.len() != state.alt_allele.len()
        && state.ref_allele.as_bytes().first() != state.alt_allele.as_bytes().first()
    {
        return Err(VariantError::Mismatch("anchor base disagrees after adjust"));
    }
    check_var_ref(state, genome)?;
    Ok(Resolution {
        flip,
        realigned: false,
    })
}

/// Read the assembly-differences VCF/BCF into a queryable index.
///
/// Contigs declared in the header are seeded even when they carry no records, so
/// a lifted variant landing on an empty-but-declared contig is not mistaken for
/// one the file does not cover.
pub fn load_ref_diffs(
    path: &str,
    chrom_filter: Option<&HashSet<String>>,
    threads: usize,
    progress: bool,
) -> std::result::Result<RefDiffIndex, String> {
    let mut reader = Reader::from_path(path).map_err(|e| format!("could not open {path}: {e}"))?;
    if threads > 1 {
        let _ = reader.set_threads(threads);
    }
    let header = reader.header().clone();

    let mut per_contig: HashMap<String, Vec<RefDiff>> = HashMap::new();
    for rid in 0..header.contig_count() {
        if let Ok(name) = header.rid2name(rid) {
            let name = String::from_utf8_lossy(name).into_owned();
            if chrom_filter.map_or(true, |f| f.contains(&name)) {
                per_contig.entry(name).or_default();
            }
        }
    }

    let mut record = reader.empty_record();
    while let Some(res) = reader.read(&mut record) {
        res.map_err(|e| format!("error reading {path}: {e}"))?;
        let rid = match record.rid() {
            Some(r) => r,
            None => continue,
        };
        let chrom = String::from_utf8_lossy(
            header
                .rid2name(rid)
                .map_err(|e| format!("bad contig id in {path}: {e}"))?,
        )
        .into_owned();
        if let Some(f) = chrom_filter {
            if !f.contains(&chrom) {
                continue;
            }
        }
        let alleles = record.alleles();
        let start = record.pos();
        // Every assembly difference must have an ALT: the liftover rules read it
        // unconditionally. The Python raises an uncaught IndexError if one is ever
        // used, so reject it up front rather than carry an empty allele silently.
        let (Some(r), Some(a)) = (alleles.first(), alleles.get(1)) else {
            return Err(format!(
                "{path}: record at {chrom}:{} has no ALT allele; the assembly-differences \
                 VCF must give both alleles of every difference",
                start + 1
            ));
        };
        let ref_allele = String::from_utf8_lossy(r).into_owned();
        let alt_allele = String::from_utf8_lossy(a).into_owned();
        per_contig.entry(chrom).or_default().push(RefDiff {
            start,
            end: start + i64::from(record.rlen() as i32),
            ref_allele,
            alt_allele,
        });
    }
    if progress {
        eprintln!("Organizing vcf containing variation between builds...");
    }
    Ok(RefDiffIndex::build(per_contig))
}

/// A successfully lifted variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lift {
    pub chrom: String,
    /// 0-based start in the target assembly.
    pub start: i64,
    pub ref_allele: String,
    pub alt_allele: String,
    /// REF and ALT swapped. Genotypes must be rewritten by the caller.
    pub flipped: bool,
    /// The representation came from haplotype realignment.
    pub realigned: bool,
}

/// Why a variant did not lift. The three failure kinds correspond to the three
/// sidecar files the command line tool writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Lifted(Lift),
    /// No unique target coordinate, an `N` in the overlapping difference, or a
    /// liftover rule that declined.
    Unliftable(&'static str),
    /// The lifted span covered more than one assembly difference.
    MultipleOverlaps,
    /// The lifted allele disagreed with the target reference.
    Mismatch(&'static str),
}

/// A loaded liftover: chain file, assembly differences and target reference.
///
/// Construct once and reuse; loading dominates the cost of lifting a variant.
pub struct Engine {
    lo: LiftOver,
    ref_diffs: RefDiffIndex,
    genome: Genome,
    realign: RealignConfig,
}

impl Engine {
    /// Load a chain file, an assembly-differences VCF/BCF in *target* coordinates,
    /// and the target reference FASTA.
    pub fn load(
        chain_path: &str,
        ref_diffs_path: &str,
        target_fasta_path: &str,
        chroms: Option<&HashSet<String>>,
        realign: RealignConfig,
        threads: usize,
    ) -> std::result::Result<Self, String> {
        let genome = load_target_genome(Path::new(target_fasta_path), chroms)?;
        let lo = LiftOver::from_file(Path::new(chain_path))?;
        let ref_diffs = load_ref_diffs(ref_diffs_path, chroms, threads, false)?;
        Ok(Engine {
            lo,
            ref_diffs,
            genome,
            realign,
        })
    }

    pub fn from_parts(
        lo: LiftOver,
        ref_diffs: RefDiffIndex,
        genome: Genome,
        realign: RealignConfig,
    ) -> Self {
        Engine {
            lo,
            ref_diffs,
            genome,
            realign,
        }
    }

    pub fn ref_diffs(&self) -> &RefDiffIndex {
        &self.ref_diffs
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    /// Lift one variant. `start` is 0-based; alleles are the source assembly's.
    ///
    /// `already_flipped` should be true when another variant at this same position
    /// has already flipped, which the Python treats as a reference mismatch.
    pub fn lift(
        &self,
        chrom: &str,
        start: i64,
        ref_allele: &str,
        alt_allele: &str,
        already_flipped: bool,
    ) -> Outcome {
        let mut state = VarState {
            chrom: chrom.to_string(),
            start,
            ref_allele: ref_allele.to_string(),
            alt_allele: alt_allele.to_string(),
        };
        let Some((lifted_start, lifted_end)) =
            perform_clean_liftover(&mut state, &self.lo, &self.genome)
        else {
            return Outcome::Unliftable("no unique target coordinate");
        };

        let target_chrom = lifted_start.0.clone();
        let span_start = lifted_start.1.min(lifted_end.1);
        let span_end = lifted_start.1.max(lifted_end.1);

        if !self.ref_diffs.has_contig(&target_chrom) {
            return Outcome::Unliftable("target contig absent from assembly differences");
        }
        let overlap = self
            .ref_diffs
            .overlapping(&target_chrom, span_start, span_end);
        if overlap.len() > 1 {
            return Outcome::MultipleOverlaps;
        }
        if overlap.iter().any(|d| d.ref_allele.contains('N')) {
            return Outcome::Unliftable("N in overlapping assembly difference");
        }

        let resolved = if overlap.is_empty() {
            resolve_without_overlap(
                &mut state,
                &target_chrom,
                span_start,
                span_end,
                &self.ref_diffs,
                &self.genome,
                &self.realign,
                already_flipped,
            )
        } else {
            resolve_with_overlap(
                &mut state,
                overlap[0],
                lifted_start.1,
                lifted_end.1,
                &self.genome,
                already_flipped,
            )
        };

        match resolved {
            Ok(Resolution { flip, realigned }) => Outcome::Lifted(Lift {
                chrom: state.chrom,
                start: state.start,
                ref_allele: state.ref_allele,
                alt_allele: state.alt_allele,
                flipped: flip,
                realigned,
            }),
            Err(VariantError::Unliftable(m)) => Outcome::Unliftable(m),
            Err(VariantError::Mismatch(m)) => Outcome::Mismatch(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_snp_matches_cyvcf2() {
        assert!(is_snp("A", "G"));
        assert!(is_snp("N", "G"));
        assert!(!is_snp("A", "GT"));
        assert!(!is_snp("AT", "G"));
        assert!(!is_snp("A", "*"));
        assert!(!is_snp("A", "g"));
    }

    #[test]
    fn var_state_end_follows_the_ref_allele() {
        let mut v = VarState {
            chrom: "c".into(),
            start: 100,
            ref_allele: "ACGT".into(),
            alt_allele: "A".into(),
        };
        assert_eq!(v.end(), 104);
        v.ref_allele = "A".into();
        assert_eq!(v.end(), 101);
    }

    #[test]
    fn check_var_ref_compares_case_sensitively() {
        let mut g: Genome = HashMap::new();
        g.insert("c".to_string(), b"ACGTACGT".to_vec());
        let ok = VarState {
            chrom: "c".into(),
            start: 2,
            ref_allele: "GT".into(),
            alt_allele: "G".into(),
        };
        assert!(check_var_ref(&ok, &g).is_ok());
        // The soft-masked form of the same base does not match.
        let lower = VarState {
            ref_allele: "gt".into(),
            ..ok.clone()
        };
        assert_eq!(
            check_var_ref(&lower, &g),
            Err(VariantError::Mismatch("lifted REF does not match target"))
        );
    }

    #[test]
    fn check_var_ref_rejects_a_degenerate_target() {
        let mut g: Genome = HashMap::new();
        g.insert("c".to_string(), b"ACNTACGT".to_vec());
        let v = VarState {
            chrom: "c".into(),
            start: 2,
            ref_allele: "NT".into(),
            alt_allele: "N".into(),
        };
        assert_eq!(
            check_var_ref(&v, &g),
            Err(VariantError::Unliftable(
                "degenerate base in target reference"
            ))
        );
    }

    #[test]
    fn a_second_flip_at_a_site_is_a_mismatch() {
        assert!(guard_single_flip(false).is_ok());
        assert_eq!(
            guard_single_flip(true),
            Err(VariantError::Mismatch("double flip at site"))
        );
    }
}
