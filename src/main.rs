//! Liftover variants between genome references in an indel-aware manner.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rust_htslib::bcf::header::Header;
use rust_htslib::bcf::record::GenotypeAllele;
use rust_htslib::bcf::{Format, Read, Reader, Record, Writer};

use liftover_indels::chain::LiftOver;
use liftover_indels::cli::Args;
use liftover_indels::engine::{
    load_ref_diffs, perform_clean_liftover, resolve_with_overlap, resolve_without_overlap,
    Resolution, VarState,
};
use liftover_indels::error::VariantError;
use liftover_indels::fasta::{load_target_genome, Genome};
use liftover_indels::realign::RealignConfig;

/// INFO definitions added to the output header, in the order the Python adds them.
const INFO_LINES: &[&str] = &[
    r#"##INFO=<ID=SRC_CHROM,Number=1,Type=Character,Description="Original contig of variant before liftover">"#,
    r#"##INFO=<ID=SRC_POS,Number=1,Type=Character,Description="Original position of variant before liftover">"#,
    r#"##INFO=<ID=Original_REF,Number=1,Type=Character,Description="Original reference sequence of variant before liftover">"#,
    r#"##INFO=<ID=Original_ALT,Number=1,Type=Character,Description="Original alt sequence of variant before liftover">"#,
    r#"##INFO=<ID=SRC_REF_ALT,Number=1,Type=Character,Description="Original ref,alt of variant before liftover">"#,
    r#"##INFO=<ID=Original_ID,Number=1,Type=Character,Description="Original variant id of variant before liftover">"#,
    r#"##INFO=<ID=Flipped_during_liftover,Number=1,Type=Character,Description="REF/ALT were flipped during liftover. GTs were altered accordingly.">"#,
    r#"##INFO=<ID=Realigned_during_liftover,Number=1,Type=Character,Description="ALT haplotype was realigned to target reference near build differences.">"#,
];

fn main() {
    let args = Args::parse();
    if let Err(msg) = run(args) {
        eprintln!("{msg}");
        std::process::exit(1);
    }
}

struct Ctx {
    debug: bool,
    genome: Genome,
    realign: RealignConfig,
}

impl Ctx {
    fn debug(&self, msg: impl std::fmt::Display) {
        if self.debug {
            eprintln!("{msg}");
        }
    }
}

/// A sample's genotype exactly as cyvcf2's `genotype.array()` row presents it:
/// the first two allele slots plus the phase flag.
///
/// Missing alleles are `-1`; a haploid sample's second slot is `-2`, htslib's
/// vector-end marker. Both are non-zero, which is what the flip rule keys on, but
/// they are kept distinct because the rule sums slots across every record at the
/// site and the two values are not interchangeable in that sum.
#[derive(Clone, Copy)]
struct SampleGt {
    a: [i16; 2],
    phased: bool,
}

fn read_genotypes(rec: &Record, n_samples: usize) -> Vec<SampleGt> {
    let mut out = Vec::with_capacity(n_samples);
    match rec.genotypes() {
        Ok(gts) => {
            for s in 0..n_samples {
                let g = gts.get(s);
                // Slots past the sample's ploidy hold htslib's vector-end marker,
                // which cyvcf2 surfaces as -2.
                let mut a = [-2i16; 2];
                for (i, slot) in a.iter_mut().enumerate() {
                    if let Some(allele) = g.get(i) {
                        *slot = match allele.index() {
                            Some(v) => v as i16,
                            None => -1,
                        };
                    }
                }
                // cyvcf2 takes the phase from the second allele's bit, and reports
                // a haploid genotype as phased.
                let phased = match g.get(1) {
                    Some(GenotypeAllele::Phased(_)) | Some(GenotypeAllele::PhasedMissing) => true,
                    Some(_) => false,
                    None => true,
                };
                out.push(SampleGt { a, phased });
            }
        }
        Err(_) => {
            for _ in 0..n_samples {
                out.push(SampleGt {
                    a: [-1, -1],
                    phased: false,
                });
            }
        }
    }
    out
}

/// Swap REF and ALT and rewrite the genotypes accordingly.
///
/// An allele becomes the new ALT exactly when the sample carries no alternate
/// allele at the site across *all* records at that position. Missing alleles are
/// `-1`, never `0`, so a half-missing genotype collapses to homozygous reference
/// rather than staying missing. Arithmetic is 16-bit and wrapping to match numpy.
fn flip_genotypes(group: &[Vec<SampleGt>], idx: usize) -> Vec<GenotypeAllele> {
    let own = &group[idx];
    let mut out = Vec::with_capacity(own.len() * 2);
    for (s, gt) in own.iter().enumerate() {
        let mut sum = gt.a;
        for (j, other) in group.iter().enumerate() {
            if j == idx {
                continue;
            }
            for k in 0..2 {
                sum[k] = sum[k].wrapping_add(other[s].a[k]);
            }
        }
        // Every sample is written with two alleles, because the Python assigns the
        // whole (samples x 2) array back. A haploid sample therefore comes out of a
        // flip as a diploid homozygote.
        for slot in sum.iter() {
            let allele = i32::from(*slot == 0);
            out.push(if gt.phased {
                GenotypeAllele::Phased(allele)
            } else {
                GenotypeAllele::Unphased(allele)
            });
        }
    }
    out
}

/// Buckets of records that failed to lift, keyed by source position so the
/// emission order matches the Python's insertion-ordered dict.
#[derive(Default)]
struct Sidecar {
    order: Vec<(String, i64)>,
    index: HashMap<(String, i64), usize>,
    groups: Vec<Vec<Record>>,
}

impl Sidecar {
    fn push(&mut self, key: (String, i64), rec: Record) {
        let slot = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                let i = self.groups.len();
                self.index.insert(key.clone(), i);
                self.order.push(key);
                self.groups.push(Vec::new());
                i
            }
        };
        self.groups[slot].push(rec);
    }

    /// Number of distinct positions, which is what the Python summary counts.
    fn positions(&self) -> usize {
        self.groups.len()
    }

    fn write(&self, writer: &mut Writer) -> Result<(), String> {
        for group in &self.groups {
            for rec in group {
                writer.write(rec).map_err(|e| format!("write failed: {e}"))?;
            }
        }
        Ok(())
    }
}

/// How cyvcf2 maps an output path onto an htslib write mode.
fn output_spec(path: &str) -> (String, bool, Format) {
    if path == "/dev/stdout" || path == "-" {
        // cyvcf2's 'wbu': uncompressed BCF on stdout.
        return ("-".to_string(), true, Format::Bcf);
    }
    let lower = path.to_lowercase();
    if lower.ends_with(".bcf") {
        (path.to_string(), false, Format::Bcf)
    } else if lower.ends_with(".vcf.gz") {
        (path.to_string(), false, Format::Vcf)
    } else {
        (path.to_string(), true, Format::Vcf)
    }
}

/// `'.'.join(out.replace('.gz','').split('.')[:-1])`, quirks included: an output
/// of `-` yields an empty base and therefore sidecars named `.unliftable.bcf`.
fn sidecar_base(out_vcf: &str) -> String {
    let stripped = out_vcf.replace(".gz", "");
    let parts: Vec<&str> = stripped.split('.').collect();
    parts[..parts.len().saturating_sub(1)].join(".")
}

fn progress_spinner(quiet: bool, msg: &str) -> Option<ProgressBar> {
    if quiet {
        return None;
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}").unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(msg.to_string());
    Some(pb)
}

fn run(args: Args) -> Result<(), String> {
    let chrom_filter: Option<HashSet<String>> =
        args.chrom.as_ref().map(|c| c.iter().cloned().collect());

    let ctx_debug = args.debug;
    if ctx_debug {
        if let Some(f) = &chrom_filter {
            let mut v: Vec<&String> = f.iter().collect();
            v.sort();
            eprintln!("restricting to contigs: {v:?}");
        }
    }

    eprintln!("Loading target reference genome...");
    let genome = load_target_genome(Path::new(&args.target_fasta), chrom_filter.as_ref())?;

    eprintln!("Loading chainfile...");
    let lo = LiftOver::from_file(Path::new(&args.chain))?;

    eprintln!("Loading vcf containing variation between builds...");
    let ref_diffs = load_ref_diffs(
        &args.ref_diffs_vcf,
        chrom_filter.as_ref(),
        args.threads,
        true,
    )?;

    let ctx = Ctx {
        debug: args.debug,
        genome,
        realign: RealignConfig {
            enabled: !args.no_realign,
            debug: args.debug,
            distance: args.realign_distance,
            flank: args.realign_flank,
            max_window: args.realign_max_window,
        },
    };

    let mut reader = Reader::from_path(&args.input_vcf)
        .map_err(|e| format!("could not open {}: {e}", args.input_vcf))?;
    if args.threads > 1 {
        let _ = reader.set_threads(args.threads);
    }
    let in_header = reader.header().clone();
    let n_samples = in_header.sample_count() as usize;

    let mut out_header = Header::from_template(&in_header);
    for line in INFO_LINES {
        out_header.push_record(line.as_bytes());
    }

    let (out_path, uncompressed, format) = output_spec(&args.output_vcf);
    if out_path != "-" && Path::new(&out_path).exists() {
        std::fs::remove_file(&out_path).map_err(|e| format!("could not remove {out_path}: {e}"))?;
    }
    let mut writer = Writer::from_path(&out_path, &out_header, uncompressed, format)
        .map_err(|e| format!("could not open output {out_path}: {e}"))?;

    eprintln!("Lifting...");
    let mut unliftable = Sidecar::default();
    let mut multiple_overlaps = Sidecar::default();
    let mut ref_seq_problems = Sidecar::default();

    let spinner = progress_spinner(args.quiet, "Lifting");
    let mut processed: u64 = 0;

    let mut groups = PositionGroups::new(&mut reader, &in_header);
    while let Some(group) = groups.next_group()? {
        let (key, mut records) = group;
        // Snapshot genotypes before any record is mutated: a flip reads every
        // record at the site, and the Python reads them in their original state.
        let group_gts: Vec<Vec<SampleGt>> = records
            .iter()
            .map(|r| read_genotypes(r, n_samples))
            .collect();

        for rec in records.iter_mut() {
            writer.translate(rec);
        }

        // Stage one: coordinate liftover for every record at this position.
        let mut states: Vec<Option<VarState>> = Vec::with_capacity(records.len());
        let mut lifted: Vec<(usize, (String, i64), (String, i64))> = Vec::new();
        let mut originals: Vec<VarState> = Vec::with_capacity(records.len());

        for (i, rec) in records.iter_mut().enumerate() {
            let chrom = contig_name(&in_header, rec)?;
            let alleles = rec.alleles();
            let ref_allele = String::from_utf8_lossy(alleles[0]).into_owned();
            let alt_allele = alleles
                .get(1)
                .map(|a| String::from_utf8_lossy(a).into_owned())
                .unwrap_or_default();
            let original = VarState {
                chrom: chrom.clone(),
                start: rec.pos(),
                ref_allele: ref_allele.clone(),
                alt_allele: alt_allele.clone(),
            };
            add_original_info_tags(rec, &original)?;
            originals.push(original);

            let mut state = VarState {
                chrom,
                start: rec.pos(),
                ref_allele,
                alt_allele,
            };
            match perform_clean_liftover(&mut state, &lo, &ctx.genome) {
                Some((s, e)) => {
                    lifted.push((i, s, e));
                    states.push(Some(state));
                }
                None => {
                    ctx.debug(format!(
                        "unliftable: missing coords for {}:{}",
                        originals[i].chrom,
                        originals[i].start + 1
                    ));
                    states.push(None);
                }
            }
        }

        // Stage two: resolve each lifted record against the assembly differences.
        let mut emit: Vec<usize> = Vec::new();
        let mut to_unliftable: Vec<usize> = Vec::new();
        let mut to_multiple: Vec<usize> = Vec::new();
        let mut to_mismatch: Vec<usize> = Vec::new();
        let mut flips: Vec<usize> = Vec::new();
        let mut realigned_flags: Vec<usize> = Vec::new();
        let mut already_flipped = false;

        for (i, st) in states.iter().enumerate() {
            if st.is_none() {
                to_unliftable.push(i);
            }
        }

        for (i, lifted_start, lifted_end) in &lifted {
            let i = *i;
            let state = states[i].as_mut().expect("lifted record has state");
            let chrom = lifted_start.0.clone();
            let span_start = lifted_start.1.min(lifted_end.1);
            let span_end = lifted_start.1.max(lifted_end.1);

            if !ref_diffs.has_contig(&chrom) {
                return Err(format!(
                    "target contig {chrom} is absent from the assembly-differences VCF; \
                     restrict with --chrom or supply differences covering it"
                ));
            }
            let overlap = ref_diffs.overlapping(&chrom, span_start, span_end);
            ctx.debug(format!(
                "processing {}:{} span {span_start}-{span_end} overlaps {}",
                state.chrom,
                state.start + 1,
                overlap.len()
            ));

            if overlap.len() > 1 {
                ctx.debug(format!("multiple overlaps for {}:{}", state.chrom, state.start + 1));
                to_multiple.push(i);
                continue;
            }
            if overlap.iter().any(|d| d.ref_allele.contains('N')) {
                ctx.debug(format!(
                    "unliftable: N in ref diff for {}:{}",
                    state.chrom,
                    state.start + 1
                ));
                to_unliftable.push(i);
                continue;
            }

            let outcome = if overlap.is_empty() {
                resolve_without_overlap(
                    state,
                    &chrom,
                    span_start,
                    span_end,
                    &ref_diffs,
                    &ctx.genome,
                    &ctx.realign,
                    already_flipped,
                )
            } else {
                resolve_with_overlap(
                    state,
                    overlap[0],
                    lifted_start.1,
                    lifted_end.1,
                    &ctx.genome,
                    already_flipped,
                )
            };

            // Captured after resolution: a realignment moves the position before
            // it can fail, and the Python's trace reflects that.
            let (dbg_chrom, dbg_pos) = {
                let s = states[i].as_ref().expect("lifted record has state");
                (s.chrom.clone(), s.start + 1)
            };
            match outcome {
                Ok(Resolution { flip, realigned }) => {
                    if flip {
                        already_flipped = true;
                        flips.push(i);
                    }
                    if realigned {
                        realigned_flags.push(i);
                    }
                    emit.push(i);
                }
                Err(VariantError::Unliftable(_)) => {
                    ctx.debug(format!("unliftable: overlap rule for {dbg_chrom}:{dbg_pos}"));
                    to_unliftable.push(i);
                }
                Err(VariantError::Mismatch(_)) => {
                    ctx.debug(format!("mismatch: ref/alt for {dbg_chrom}:{dbg_pos}"));
                    to_mismatch.push(i);
                }
            }
        }

        // Stage three: apply the resolved state to the records and emit.
        for &i in &flips {
            let gts = flip_genotypes(&group_gts, i);
            records[i]
                .push_genotypes(&gts)
                .map_err(|e| format!("could not write genotypes: {e}"))?;
            records[i]
                .push_info_string(b"Flipped_during_liftover", &[b"Flipped"])
                .map_err(|e| format!("could not set INFO: {e}"))?;
        }
        for &i in &realigned_flags {
            records[i]
                .push_info_string(b"Realigned_during_liftover", &[b"Realigned"])
                .map_err(|e| format!("could not set INFO: {e}"))?;
        }
        for &i in &emit {
            let st = states[i].as_ref().expect("emitted record has state");
            apply_state(&mut records[i], st, &out_header)?;
        }

        for &i in &to_unliftable {
            revert_variant(&mut records[i], &originals[i], &out_header)?;
        }
        for &i in &to_multiple {
            revert_variant(&mut records[i], &originals[i], &out_header)?;
        }
        for &i in &to_mismatch {
            revert_variant(&mut records[i], &originals[i], &out_header)?;
        }

        // Written in the order the Python appends them.
        for &i in &emit {
            writer
                .write(&records[i])
                .map_err(|e| format!("write failed: {e}"))?;
        }
        for &i in &to_unliftable {
            unliftable.push(key.clone(), records[i].clone());
        }
        for &i in &to_multiple {
            multiple_overlaps.push(key.clone(), records[i].clone());
        }
        for &i in &to_mismatch {
            ref_seq_problems.push(key.clone(), records[i].clone());
        }

        processed += 1;
        if let Some(s) = &spinner {
            if processed % 10_000 == 0 {
                s.set_message(format!("Lifting ({processed} positions)"));
                s.tick();
            }
        }
    }
    if let Some(s) = spinner {
        s.finish_and_clear();
    }
    drop(writer);

    eprintln!(
        "{} variants could not be lifted over because they did not have a start and/or end coordinate in the target assembly.",
        unliftable.positions()
    );
    eprintln!(
        "{} variants could not be lifted over because they had multiple potential liftover spots.",
        multiple_overlaps.positions()
    );
    eprintln!(
        "{} variants had an unspecified error in liftover.",
        ref_seq_problems.positions()
    );
    eprintln!("Writing these variants to disk...");

    let base = sidecar_base(&args.output_vcf);
    for (suffix, sidecar) in [
        (".unliftable.bcf", &unliftable),
        (".multiple_overlaps.bcf", &multiple_overlaps),
        (".ref_seq_mismatches.bcf", &ref_seq_problems),
    ] {
        let path = format!("{base}{suffix}");
        let mut w = Writer::from_path(&path, &out_header, uncompressed, format)
            .map_err(|e| format!("could not open {path}: {e}"))?;
        sidecar.write(&mut w)?;
    }

    eprintln!("\nDone!");
    eprintln!("Note: lifted vcf file still requires indel normalizing, sorting, and recalculation of INFO fields. Format fields besides GT are no longer reliable.");
    Ok(())
}

fn contig_name(header: &rust_htslib::bcf::header::HeaderView, rec: &Record) -> Result<String, String> {
    let rid = rec.rid().ok_or_else(|| "record without a contig".to_string())?;
    Ok(String::from_utf8_lossy(
        header.rid2name(rid).map_err(|e| format!("bad contig: {e}"))?,
    )
    .into_owned())
}

fn add_original_info_tags(rec: &mut Record, v: &VarState) -> Result<(), String> {
    let id = rec.id();
    let id = if id == b"." || id.is_empty() {
        ".".to_string()
    } else {
        String::from_utf8_lossy(&id).into_owned()
    };
    let pos_str = (v.start + 1).to_string();
    let ref_alt = format!("{},{}", v.ref_allele, v.alt_allele);
    let pairs: [(&[u8], &str); 6] = [
        (b"SRC_CHROM", &v.chrom),
        (b"SRC_POS", &pos_str),
        (b"Original_REF", &v.ref_allele),
        (b"Original_ALT", &v.alt_allele),
        (b"Original_ID", &id),
        (b"SRC_REF_ALT", &ref_alt),
    ];
    for (tag, value) in pairs {
        rec.push_info_string(tag, &[value.as_bytes()])
            .map_err(|e| format!("could not set INFO {}: {e}", String::from_utf8_lossy(tag)))?;
    }
    Ok(())
}

fn apply_state(
    rec: &mut Record,
    v: &VarState,
    header: &Header,
) -> Result<(), String> {
    let _ = header;
    let rid = rec
        .header()
        .name2rid(v.chrom.as_bytes())
        .map_err(|_| format!("contig {} missing from output header", v.chrom))?;
    rec.set_rid(Some(rid));
    rec.set_pos(v.start);
    rec.set_alleles(&[v.ref_allele.as_bytes(), v.alt_allele.as_bytes()])
        .map_err(|e| format!("could not set alleles: {e}"))?;
    Ok(())
}

/// Restore the pre-liftover coordinates and alleles. Genotypes and the INFO tags
/// added during liftover are deliberately left as they are, matching the Python.
fn revert_variant(rec: &mut Record, original: &VarState, header: &Header) -> Result<(), String> {
    apply_state(rec, original, header)
}

/// Group consecutive records that share a position.
///
/// Reproduces the Python generator, including its handling of star alleles: a
/// record that opens a new position and whose ALT contains `*` is dropped and the
/// pending group is *not* emitted, so the boundary is deferred to the next record.
struct PositionGroups<'a> {
    reader: &'a mut Reader,
    header: &'a rust_htslib::bcf::header::HeaderView,
    current: Option<(String, i64)>,
    pending: Vec<Record>,
    done: bool,
}

impl<'a> PositionGroups<'a> {
    fn new(reader: &'a mut Reader, header: &'a rust_htslib::bcf::header::HeaderView) -> Self {
        PositionGroups {
            reader,
            header,
            current: None,
            pending: Vec::new(),
            done: false,
        }
    }

    fn next_group(&mut self) -> Result<Option<((String, i64), Vec<Record>)>, String> {
        loop {
            if self.done {
                if self.pending.is_empty() {
                    return Ok(None);
                }
                let key = self.current.take().expect("pending implies a key");
                return Ok(Some((key, std::mem::take(&mut self.pending))));
            }

            let mut record = self.reader.empty_record();
            match self.reader.read(&mut record) {
                None => {
                    self.done = true;
                    continue;
                }
                Some(res) => res.map_err(|e| format!("error reading input: {e}"))?,
            }

            let chrom = contig_name(self.header, &record)?;
            let pos = record.pos();

            let Some((cur_chrom, cur_pos)) = self.current.clone() else {
                self.current = Some((chrom, pos));
                self.pending.push(record);
                continue;
            };

            if pos != cur_pos {
                if pos <= cur_pos && chrom == cur_chrom {
                    return Err(format!(
                        "Variant at position {pos} is after a variant at {cur_pos}. \
                         Input variant file must be sorted before liftover."
                    ));
                }
                let alleles = record.alleles();
                let Some(alt) = alleles.get(1) else {
                    // No ALT: the Python logs and drops the record without
                    // closing the pending group.
                    eprintln!("{chrom} {} {} []", pos + 1, String::from_utf8_lossy(alleles[0]));
                    continue;
                };
                if alt.contains(&b'*') {
                    continue;
                }
                let key = (cur_chrom, cur_pos);
                let group = std::mem::take(&mut self.pending);
                self.current = Some((chrom, pos));
                self.pending.push(record);
                return Ok(Some((key, group)));
            }
            self.pending.push(record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt(a0: i16, a1: i16, phased: bool) -> SampleGt {
        SampleGt {
            a: [a0, a1],
            phased,
        }
    }

    fn alleles(v: &[GenotypeAllele]) -> Vec<(i32, bool)> {
        v.iter()
            .map(|g| match g {
                GenotypeAllele::Phased(i) => (*i, true),
                GenotypeAllele::Unphased(i) => (*i, false),
                GenotypeAllele::PhasedMissing => (-1, true),
                GenotypeAllele::UnphasedMissing => (-1, false),
            })
            .collect()
    }

    #[test]
    fn flip_turns_reference_alleles_into_alternates() {
        // One record at the site: 0|1 becomes 1|0.
        let group = vec![vec![gt(0, 1, true)]];
        assert_eq!(alleles(&flip_genotypes(&group, 0)), vec![(1, true), (0, true)]);
    }

    #[test]
    fn flip_preserves_the_phase_flag() {
        let group = vec![vec![gt(0, 1, false)]];
        assert_eq!(
            alleles(&flip_genotypes(&group, 0)),
            vec![(1, false), (0, false)]
        );
    }

    // Missing is -1, not 0, so it does not survive the flip as missing.
    #[test]
    fn a_half_missing_genotype_collapses_to_homozygous_reference() {
        let group = vec![vec![gt(1, -1, true)]];
        assert_eq!(alleles(&flip_genotypes(&group, 0)), vec![(0, true), (0, true)]);
    }

    #[test]
    fn a_fully_missing_genotype_collapses_to_homozygous_reference() {
        let group = vec![vec![gt(-1, -1, false)]];
        assert_eq!(
            alleles(&flip_genotypes(&group, 0)),
            vec![(0, false), (0, false)]
        );
    }

    // A haploid sample carries the vector-end marker in its second slot and is
    // written back as a diploid homozygote.
    #[test]
    fn a_haploid_genotype_becomes_a_diploid_homozygote() {
        let group = vec![vec![gt(1, -2, true)]];
        assert_eq!(alleles(&flip_genotypes(&group, 0)), vec![(0, true), (0, true)]);
    }

    // The rule sums across every record at the site: an allele becomes the new
    // ALT only if the sample is reference for all of them.
    #[test]
    fn flip_accounts_for_other_records_at_the_same_position() {
        // Sample is 0|0 here but 0|1 at the other record, so slot 1 is not free.
        let group = vec![vec![gt(0, 0, true)], vec![gt(0, 1, true)]];
        assert_eq!(alleles(&flip_genotypes(&group, 0)), vec![(1, true), (0, true)]);
    }

    #[test]
    fn output_spec_follows_the_extension() {
        assert_eq!(output_spec("a.bcf"), ("a.bcf".to_string(), false, Format::Bcf));
        assert_eq!(
            output_spec("a.vcf.gz"),
            ("a.vcf.gz".to_string(), false, Format::Vcf)
        );
        assert_eq!(output_spec("a.vcf"), ("a.vcf".to_string(), true, Format::Vcf));
        assert_eq!(output_spec("-"), ("-".to_string(), true, Format::Bcf));
        assert_eq!(
            output_spec("/dev/stdout"),
            ("-".to_string(), true, Format::Bcf)
        );
    }

    #[test]
    fn sidecar_base_drops_the_last_extension() {
        assert_eq!(sidecar_base("out.bcf"), "out");
        assert_eq!(sidecar_base("out.vcf.gz"), "out");
        assert_eq!(sidecar_base("a.b.bcf"), "a.b");
        // Stdout leaves an empty base, so the sidecars become dotfiles.
        assert_eq!(sidecar_base("-"), "");
    }
}
