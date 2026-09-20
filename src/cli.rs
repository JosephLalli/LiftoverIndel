//! Command line interface, mirroring the Python argparse definition.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "liftover_indels",
    version,
    about = "Liftover variants between genome references in an indel-aware manner."
)]
pub struct Args {
    /// VCF/BCF file to lift over
    #[arg(long = "input-vcf")]
    pub input_vcf: String,

    /// VCF/BCF of assembly differences (must be in target assembly coordinates)
    #[arg(long = "ref-diffs-vcf")]
    pub ref_diffs_vcf: String,

    /// Output VCF/BCF path. Use /dev/stdout or - for stdout
    #[arg(long = "output-vcf")]
    pub output_vcf: String,

    /// Chain file for coordinate liftover
    #[arg(long = "chain")]
    pub chain: String,

    /// Target reference FASTA (may be gzipped)
    #[arg(long = "target-fasta")]
    pub target_fasta: String,

    /// Restrict liftover to these contigs (e.g. --chrom chr1 chr22)
    #[arg(long = "chrom", num_args = 1..)]
    pub chrom: Option<Vec<String>>,

    /// Disable haplotype realignment near reference differences
    #[arg(long = "no-realign", default_value_t = false, help_heading = "haplotype realignment")]
    pub no_realign: bool,

    /// Max distance (bp) to search for nearby ref diffs
    #[arg(long = "realign-distance", default_value_t = 50, help_heading = "haplotype realignment")]
    pub realign_distance: i64,

    /// Flanking bases added to each side of the realignment window
    #[arg(long = "realign-flank", default_value_t = 20, help_heading = "haplotype realignment")]
    pub realign_flank: i64,

    /// Maximum total realignment window size in bp
    #[arg(long = "realign-max-window", default_value_t = 200, help_heading = "haplotype realignment")]
    pub realign_max_window: i64,

    /// Threads for VCF/BCF reading
    #[arg(long = "threads", default_value_t = 2)]
    pub threads: usize,

    /// Enable verbose debug logging to stderr
    #[arg(long = "debug", default_value_t = false)]
    pub debug: bool,

    /// Suppress progress bars
    #[arg(long = "quiet", default_value_t = false)]
    pub quiet: bool,
}
