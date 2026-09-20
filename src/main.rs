//! Liftover variants between genome references in an indel-aware manner.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rust_htslib::bcf::header::Header;
use rust_htslib::bcf::record::GenotypeAllele;
use rust_htslib::bcf::{Format, Read, Reader, Record, Writer};

use liftover_indels::adjust::compute_adjusted_ref_alt;
use liftover_indels::alleles::{rev_comp, trim_identical_suffix};
use liftover_indels::chain::LiftOver;
use liftover_indels::cli::Args;
use liftover_indels::error::VariantError;
use liftover_indels::fasta::{load_target_genome, Genome};
use liftover_indels::realign::{attempt_haplotype_realignment, RealignConfig};
use liftover_indels::refdiff::{RefDiff, RefDiffIndex};

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

/// A sample's genotype as the Python sees it: the first two alleles (`-1` when
/// missing) plus whether the genotype is phased.
#[derive(Clone, Copy)]
struct SampleGt {
    a: [i16; 2],
    phased: bool,
    ploidy: usize,
}

fn read_genotypes(rec: &Record, n_samples: usize) -> Vec<SampleGt> {
    let mut out = Vec::with_capacity(n_samples);
    match rec.genotypes() {
        Ok(gts) => {
            for s in 0..n_samples {
                let g = gts.get(s);
                let mut a = [-1i16; 2];
                for (i, slot) in a.iter_mut().enumerate() {
                    if let Some(allele) = g.get(i) {
                        *slot = match allele.index() {
                            Some(v) => v as i16,
                            None => -1,
                        };
                    }
                }
                // cyvcf2 reports a genotype as phased from the second allele's bit.
                let phased = match g.get(1).or_else(|| g.first()) {
                    Some(GenotypeAllele::Phased(_)) | Some(GenotypeAllele::PhasedMissing) => true,
                    _ => false,
                };
                out.push(SampleGt {
                    a,
                    phased,
                    ploidy: g.len(),
                });
            }
        }
        Err(_) => {
            for _ in 0..n_samples {
                out.push(SampleGt {
                    a: [-1, -1],
                    phased: false,
                    ploidy: 2,
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
        let n = gt.ploidy.min(2).max(1);
        for slot in sum.iter().take(n) {
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

/// `seq[a:b]` with Python's clamping.
fn pslice(seq: &[u8], a: usize, b: usize) -> &[u8] {
    let len = seq.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    &seq[a..b]
}

fn is_snp(r: &str, a: &str) -> bool {
    r.len() == 1 && matches!(a, "A" | "C" | "G" | "T")
}

/// The alleles and coordinates of a record as the liftover works on them.
struct VarState {
    chrom: String,
    start: i64,
    ref_allele: String,
    alt_allele: String,
}

impl VarState {
    fn end(&self) -> i64 {
        self.start + self.ref_allele.len() as i64
    }
}

/// Verify a lifted allele against the target reference.
fn check_var_ref(v: &VarState, genome: &Genome) -> Result<(), VariantError> {
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
fn perform_clean_liftover(
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

fn load_ref_diffs(
    path: &str,
    chrom_filter: Option<&HashSet<String>>,
    threads: usize,
    quiet: bool,
) -> Result<RefDiffIndex, String> {
    let mut reader =
        Reader::from_path(path).map_err(|e| format!("could not open {path}: {e}"))?;
    if threads > 1 {
        let _ = reader.set_threads(threads);
    }
    let header = reader.header().clone();

    let mut per_contig: HashMap<String, Vec<RefDiff>> = HashMap::new();
    // Contigs declared in the header start empty, as the Python pre-seeds them.
    for rid in 0..header.contig_count() {
        if let Ok(name) = header.rid2name(rid) {
            let name = String::from_utf8_lossy(name).into_owned();
            if chrom_filter.map_or(true, |f| f.contains(&name)) {
                per_contig.entry(name).or_default();
            }
        }
    }

    let spinner = progress_spinner(quiet, "Loading variation between builds");
    let mut record = reader.empty_record();
    let mut n: u64 = 0;
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
        if alleles.is_empty() {
            continue;
        }
        let ref_allele = String::from_utf8_lossy(alleles[0]).into_owned();
        let alt_allele = alleles
            .get(1)
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .unwrap_or_default();
        let start = record.pos();
        per_contig.entry(chrom).or_default().push(RefDiff {
            start,
            end: start + i64::from(record.rlen() as i32),
            ref_allele,
            alt_allele,
        });
        n += 1;
        if n % 100_000 == 0 {
            if let Some(s) = &spinner {
                s.set_message(format!("Loading variation between builds ({n} records)"));
                s.tick();
            }
        }
    }
    if let Some(s) = spinner {
        s.finish_and_clear();
    }
    eprintln!("Organizing vcf containing variation between builds...");
    Ok(RefDiffIndex::build(per_contig))
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
        args.quiet,
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
            let state = states[i].as_mut_slice_hack();
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
                    &ctx,
                    already_flipped,
                )
            } else {
                resolve_with_overlap(
                    state,
                    overlap[0],
                    lifted_start.1,
                    lifted_end.1,
                    &ctx,
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

struct Resolution {
    flip: bool,
    realigned: bool,
}

/// A site may only flip once. The Python raises `AssertionError` here, which its
/// handler catches alongside `ValueError`, so a second flip is diverted to the
/// ref-mismatch sidecar rather than aborting the run. The check happens before
/// the reference is validated, so it takes precedence over a reference failure.
fn guard_single_flip(already_flipped: bool) -> Result<(), VariantError> {
    if already_flipped {
        Err(VariantError::Mismatch("double flip at site"))
    } else {
        Ok(())
    }
}

fn resolve_without_overlap(
    state: &mut VarState,
    chrom: &str,
    span_start: i64,
    span_end: i64,
    ref_diffs: &RefDiffIndex,
    ctx: &Ctx,
    already_flipped: bool,
) -> Result<Resolution, VariantError> {
    let mut flip = false;
    let mut realigned = false;
    let attempt = attempt_haplotype_realignment(
        &state.ref_allele,
        &state.alt_allele,
        chrom,
        span_start,
        span_end,
        ref_diffs,
        &ctx.genome,
        &ctx.realign,
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
    check_var_ref(state, &ctx.genome)?;
    Ok(Resolution { flip, realigned })
}

fn resolve_with_overlap(
    state: &mut VarState,
    diff: &RefDiff,
    lifted_start: i64,
    lifted_end: i64,
    ctx: &Ctx,
    already_flipped: bool,
) -> Result<Resolution, VariantError> {
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
    check_var_ref(state, &ctx.genome)?;
    Ok(Resolution {
        flip,
        realigned: false,
    })
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

/// Small helper so a `&mut Option<VarState>` can be used where the state is known
/// to be present.
trait OptionStateExt {
    fn as_mut_slice_hack(&mut self) -> &mut VarState;
}

impl OptionStateExt for Option<VarState> {
    fn as_mut_slice_hack(&mut self) -> &mut VarState {
        self.as_mut().expect("lifted record must have state")
    }
}
