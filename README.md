# LiftoverIndel
Tool to liftover variants between references in an indel-aware manner. Importantly, this tool identifies variants that overlap a region of the target reference that has an indel relative to the origin reference and incorporates that indel into the lifted variant. When a variant lands near (but not directly overlapping) a reference difference between builds, the tool performs haplotype realignment -- reconstructing the source haplotype in a local window and using global alignment against the target reference to find the correct variant representation.

Under active development, but should work for most CHM13/GRCh38 reference liftovers.

### Please note:
- Input variants must be biallelic.
- Output variants should be left-aligned and sorted.
- BCF or VCF files can both be read, but using bcf files allows for a *much* quicker liftover run. I encourage the conversion of all files, even the reference liftover vcfs, to bcf format for this purpose.
- Output format should be autodetected from the provided output file extension.
- **Only genotypes** are lifted over; while some of the other info/format fields may be correct, these fields are explicitly not touched during liftover. Some/most non-GT fields **will be wrong**. If lifting these fields is important to you, please post an issue and I will do my best to add this feature.

### Requirements

```
pip install -r requirements.txt
```

Dependencies:
- pyliftover
- tqdm
- intervaltree
- cyvcf2
- numpy
- biopython

### Usage

Once requirements are installed:
```
python3 liftover_indels.py \
    --input-vcf vcf_to_lift.bcf \
    --ref-diffs-vcf vcf_of_assembly_differences.bcf \
    --output-vcf lifted_over_output.bcf \
    --chain chainfile.chain \
    --target-fasta target_fasta.fasta \
    [options]
```

For example, to lift only chr22 with debug logging and 4 reader threads:
```
python3 liftover_indels.py \
    --input-vcf input.bcf \
    --ref-diffs-vcf chm13v2-grch38.sort.bcf \
    --output-vcf output.bcf \
    --chain chm13v2-grch38.chain \
    --target-fasta GRCh38.fasta \
    --chrom chr22 --debug --threads 4
```

- `--ref-diffs-vcf` must be in **target** assembly coordinates.
- `--output-vcf` can be set to `/dev/stdout` or `-` for piping.
- Run `python3 liftover_indels.py --help` for full option details.

### Options

| Flag | Default | Description |
|---|---|---|
| `--chrom CHR [CHR ...]` | all | Restrict liftover to specific contigs (e.g. `--chrom chr1 chr22`) |
| `--no-realign` | off | Disable haplotype realignment near reference differences |
| `--realign-distance` | `50` | Max distance (bp) to search for nearby ref diffs during realignment |
| `--realign-flank` | `20` | Flanking bases added to each side of the realignment window |
| `--realign-max-window` | `200` | Maximum total realignment window size (bp) |
| `--threads` | `2` | Threads for VCF/BCF reading |
| `--debug` | off | Enable verbose debug logging to stderr |
| `--quiet` | off | Suppress progress bars |

### Rust implementation

`liftover_indels.py` is also implemented in Rust, in `src/`. It takes the same flags,
reads and writes the same files, and is the faster of the two; the Python script
remains in the tree and is the reference the Rust is checked against.

#### Building

```
cargo build --release
```

The binary is written to `target/release/liftover_indels`.

The `rust-htslib` dependency builds htslib through `bindgen`, which needs a working
`libclang`. Where the system `libclang` is broken or missing, point the build at
another LLVM installation, for example:

```
export LIBCLANG_PATH=/path/to/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(/path/to/llvm/bin/clang -print-resource-dir)/include"
cargo build --release
```

#### Running

```
target/release/liftover_indels \
    --input-vcf vcf_to_lift.bcf \
    --ref-diffs-vcf vcf_of_assembly_differences.bcf \
    --output-vcf lifted_over_output.bcf \
    --chain chainfile.chain \
    --target-fasta target_fasta.fasta \
    [options]
```

#### Agreement with the Python

Both were run over chr21 of a 107-sample HPRC/HGSVC callset, 619,323 biallelic
variants, lifted from CHM13v2 to GRCh38. The three sidecar files are identical
record for record (37,500 unliftable, 714 multiple-overlap, 32,601 ref-seq
mismatch). The main output agrees on 548,503 of 548,508 records, the five that differ
landing at three positions. The per-decision `--debug` traces agree on 682,997 of the
683,006 lines the Python emits; nine Python lines have no counterpart and the Rust
emits seven the Python does not, all of them realignment decisions. With `--no-realign`
the two agree completely.

The five differing records all come from one place. When haplotype realignment looks
for the nearest assembly difference, two differences can be exactly the same distance
from the variant, and `min()` in Python then returns whichever the interval tree's
result set happens to iterate first. Across chr21, 13,307 queries reach that choice
and 45 of them are tied; Python takes the earlier difference in 29 and the later one
in 16, so the choice is not a rule that can be reimplemented. The Rust always takes
the earliest difference by `(start, end)`, which is at least reproducible across
machines; it therefore agrees on 29 of the 45, and only three of the remaining
sixteen change a written record. Adding `indels.sort(key=lambda d: (d.start, d.end))`
before the `min()` call in `attempt_haplotype_realignment` would make the Python
deterministic and bring the two into exact agreement.

Two further differences, neither affecting output records:

- The Rust does not make the extra counting pass over the input that the Python makes
  to size its progress bars, so progress is reported without a known total.
- For a record with no ALT allele, the Python logs two stderr lines (a summary and the
  full record) before dropping it; the Rust logs only the summary line.

Where the Python would raise an uncaught exception the Rust exits with a message
instead: a target contig missing from the assembly-differences VCF, and an
assembly-difference record with no ALT allele.

### Output Files

In addition to the main lifted output, three sidecar files are written (using the output path as a base name):

- `<base>.unliftable.bcf` -- variants that could not be lifted because they lacked start and/or end coordinates in the target assembly.
- `<base>.multiple_overlaps.bcf` -- variants that overlapped multiple reference differences and could not be unambiguously lifted.
- `<base>.ref_seq_mismatches.bcf` -- variants that failed post-liftover reference sequence validation.

### INFO Tags

The following INFO tags are added to each successfully lifted variant:

| Tag | Description |
|---|---|
| `SRC_CHROM` | Original contig before liftover |
| `SRC_POS` | Original position before liftover |
| `Original_REF` | Original REF allele before liftover |
| `Original_ALT` | Original ALT allele before liftover |
| `SRC_REF_ALT` | Original REF,ALT combined string |
| `Original_ID` | Original variant ID before liftover |
| `Flipped_during_liftover` | Set to `Flipped` when REF/ALT were swapped (genotypes adjusted accordingly) |
| `Realigned_during_liftover` | Set to `Realigned` when haplotype realignment was used to resolve the variant |

### Assembly Differences VCF

Assembly differences vcf can either be generated by you, or in the CHM13/GRCh38 liftover context it can be obtained from the [HPRC AWS bucket](https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=T2T/CHM13/assemblies/chain/v1_nflo/).
<br>
[GRCh38-CHM13](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chain/v1_nflo/grch38-chm13v2.sort.vcf.gz) vcf.gz file (GRCh38 coordinates). Use when lifting CHM13 -> GRCh38 coordinates.
<br>
[CHM13-GRCh38](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chain/v1_nflo/chm13v2-grch38.sort.vcf.gz) vcf.gz file (CHM13 coordinates). Use when lifting GRCh38 -> CHM13 coordinates.