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

`liftover_indels.py` is also implemented in Rust, in `src/`. It takes the same flags
and reads and writes the same files; the Python script remains in the tree and is
the reference the Rust is checked against. See [Performance](#performance) for
measured timings and [C and C++ API](#c-and-c-api) for using it from other
languages.

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

#### Performance

Measured on chr21 of the 107-sample callset, CHM13v2 to GRCh38, with identical
inputs and flags. The host is a shared 2x AMD EPYC 7713 with other work running
(load average around 65), so each condition was repeated and the median is given
with the observed spread. Both implementations default to `--threads 2`.

| Workload | Python | Rust | Speedup | Python peak RSS | Rust peak RSS |
|---|---|---|---|---|---|
| Load only (1 variant) | 9.83 s | 0.24 s | 41x | 663 MB | 82 MB |
| 29,020 variants | 15.63 s | 1.07 s | 15x | 668 MB | 84 MB |
| 619,323 variants | 82.91 s | 9.85 s | 8.4x | 841 MB | 134 MB |
| 619,323, `--no-realign` | 46.27 s | 7.58 s | 6.1x | 835 MB | 136 MB |

Spread over repetitions: Python 80.9-88.7 s and Rust 8.5-11.3 s on the full
chromosome (n=3); Python 9.3-14.4 s and Rust 0.24-0.26 s on load only (n=5).

The end-to-end ratio understates the difference on small jobs and overstates it on
large ones, because a fixed cost of loading the chain, the assembly differences and
the target reference sits in front of every run. Subtracting the load-only time
leaves the marginal cost of lifting one variant:

| | Load | Per variant | Throughput |
|---|---|---|---|
| `liftover_indels.py` | 9.83 s | 118 us | 8,500 variants/s |
| Rust, command line | 0.24 s | 15.5 us | 64,000 variants/s |
| Rust, Python API | 0.26 s | 7.3 us | 137,000 variants/s |
| Rust, C API | 0.25 s | 5.6 us | 177,000 variants/s |

The C API is faster than the command line tool because it does no VCF work: it
neither parses records nor decodes and rewrites genotypes for 107 samples, which is
most of the command line tool's remaining 10 us. The Python API reaches the same
engine through ctypes and pays 1.7 us per call for the crossing, which leaves it
16 times faster per variant than the Python implementation it replaces.

Haplotype realignment is the most expensive stage and the two implementations pay
very differently for it. It adds 59 us per variant to the Python, roughly doubling
its per-variant cost, against 3.7 us to the Rust, about a third more.

### C and C++ API

The library exposes a C ABI, so C and C++ callers can lift variants without going
through a VCF. The header is `include/liftover_indels.h` and is safe to include
from C++.

`cargo build --release` produces `target/release/libliftover_indels.a` and
`libliftover_indels.so` alongside the binary.

```c++
#include "liftover_indels.h"

liftover_indels_options opts;
liftover_indels_options_init(&opts);

char *error = nullptr;
const char *contigs[] = {"chr21"};
liftover_indels_engine *engine = liftover_indels_open(
    "chm13v2-grch38.chain", "grch38-chm13v2.sort.bcf", "GRCh38.fasta",
    contigs, 1, &opts, &error);
if (!engine) { /* report error, then liftover_indels_string_free(error) */ }

liftover_indels_result r;
liftover_indels_result_init(&r);   // so dispose is safe even if no lift runs
if (liftover_indels_lift(engine, "chr21", pos0, "AAAT", "A", 0, &r)
        == LIFTOVER_INDELS_STATUS_OK) {
    // r.chrom, r.pos (0-based), r.ref_allele, r.alt_allele, r.flipped, r.realigned
}
liftover_indels_result_dispose(&r);   // frees all of the strings together
liftover_indels_close(engine);
```

Linking statically:

```
c++ -std=c++17 -O2 -I include app.cpp \
    target/release/libliftover_indels.a -lpthread -ldl -lm -lz -lbz2 -llzma \
    -o app
```

Notes:

- Loading dominates the cost of a lift, so build one engine and reuse it. Naming
  only the contigs you need keeps the rest of the target reference out of memory;
  passing `NULL`, or `n_chroms == 0`, loads every contig.
- `liftover_indels_lift` takes a `const` engine and does not mutate it, so one
  engine can be shared by several threads lifting concurrently. A unit test
  asserts `Engine: Send + Sync` so this cannot regress silently.
- A result owns its strings and they are freed **only** by
  `liftover_indels_result_dispose`, which releases all of them together. Never
  free an individual field; `liftover_indels_string_free` is for one thing, the
  error string `liftover_indels_open` writes on failure.
- `dispose` frees whatever the struct holds and cannot tell an uninitialised
  pointer from `NULL`, so initialise a result with `liftover_indels_result_init`
  (or `= {0}`) before any path that might dispose it without lifting.
- `lift` overwrites a result without freeing what was there, so reusing one across
  lifts leaks unless you dispose between them. Declaring the result inside the loop,
  as the example does, avoids the question.
- Do not edit a returned string in place: they are freed by recomputing their
  length, so truncating one makes the later free wrong. Copy it first.
- The API works at the allele level. A `flipped` result means REF and ALT were
  swapped and the caller must rewrite sample genotypes; that rewrite needs every
  record at the position, which only the caller has. `flipped` is meaningful only
  when the status is OK -- a variant that flipped and then failed its reference
  check reports `REF_MISMATCH` with `flipped == 0`.
- `already_flipped` rejects a *second* flip at a position; it does not make every
  call fail. A variant that needs no flip lifts normally whatever you pass.
- Panics are caught at the boundary and returned as `LIFTOVER_INDELS_STATUS_ERROR`
  rather than unwinding into foreign frames, and a `NULL` engine or string argument
  is reported the same way rather than crashing.

`examples/cpp/liftover_example.cpp` is a worked client and doubles as the API's
integration test: over 29,020 chr21 variants it reproduces the command line tool's
partition and values exactly (25,550 lifted, 1,753 unliftable, 1,665 reference
mismatches, 52 multiple-overlap), and valgrind reports every heap block freed.

### Python API

`python/liftover_indels` binds the same engine through `ctypes`. There is nothing
to compile beyond the library itself, and no third-party dependency.

```
cargo build --release          # produces target/release/libliftover_indels.so
pip install ./python           # or just put python/ on PYTHONPATH
```

```python
from liftover_indels import LiftOver

with LiftOver("chm13v2-grch38.chain", "grch38-chm13v2.sort.bcf",
              "GRCh38.fasta", contigs=["chr21"]) as lo:
    r = lo.lift("chr21", 20000049, "C", "T")   # 0-based
    if r.ok:
        print(r.chrom, r.pos, r.ref, r.alt, r.flipped, r.realigned)
    else:
        print(r.status.name, r.message)
```

Positions are **0-based**, matching the C API and cyvcf2's `variant.start`, and the
alleles passed in are the source assembly's.

A variant that does not lift is returned, not raised: `r.status` is one of
`UNLIFTABLE`, `MULTIPLE_OVERLAPS` or `REF_MISMATCH`, the same three buckets the
command line tool writes as sidecar files. `LiftoverError` is reserved for a bad
argument or an unreadable input.

The shared library is searched for in this order, and `LIFTOVER_INDELS_LIB`
overrides it: next to the package, `target/release/` relative to a source checkout,
`sys.prefix/lib`, `~/usr/local/lib`, `/usr/local/lib`, then the system loader.
`liftover_indels.library_path()` reports which one would be used.

Notes:

- Loading dominates the cost, so build one `LiftOver` and reuse it. Naming only the
  contigs you need keeps the rest of the target reference out of memory.
- A `LiftOver` is safe to share between threads: the engine is immutable once
  loaded and each thread gets its own result buffer.
- The binding works at the allele level, like the C API. `r.flipped` means REF and
  ALT were swapped and sample genotypes must be rewritten by the caller; it is
  meaningful only when the status is OK. `already_flipped` rejects a *second* flip
  at a position.

Verified over the same 29,020 chr21 variants as the C++ client: identical results,
zero mismatches. RSS is flat across 290,200 lifts, so the per-call strings are
released as they should be.

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