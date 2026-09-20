# Changelog

## Unreleased

- Expose a C ABI (`include/liftover_indels.h`) so C and C++ callers can lift
  variants directly. The library now builds as a static and a shared library
  alongside the Rust one, and `examples/cpp/liftover_example.cpp` is a worked
  client. Over 29,020 chr21 variants the C API reproduces the command line tool's
  partition and values exactly, and valgrind reports every heap block freed.
- Move the per-variant liftover into `src/engine.rs`, leaving the binary with
  record I/O, position grouping and genotype rewriting. Full-chromosome output is
  unchanged by the move, byte for byte.
- Measured against the Python on chr21 (619,323 variants, 107 samples): 82.9 s and
  841 MB for the Python against 9.9 s and 134 MB for the Rust. Excluding the fixed
  load, the marginal cost per variant is 118 us for the Python, 15.5 us through the
  Rust command line, and 5.6 us through the C API.

- Add a Rust implementation of the liftover in `src/`, built with `cargo build --release`.
  It takes the same flags as `liftover_indels.py` and writes the same main output and
  sidecar files. The Python script stays in the tree as the reference implementation.
- Verified against the Python over chr21 of a 107-sample callset, 619,323 biallelic
  variants lifted from CHM13v2 to GRCh38: the three sidecar files match record for
  record, the main output matches on 548,503 of 548,508 records, and 682,997 of the
  683,006 `--debug` trace lines the Python emits are reproduced exactly. With
  `--no-realign` the two agree completely.
- The remaining five records come from `attempt_haplotype_realignment` choosing between
  two assembly differences that are equally distant from the variant. Python's `min()`
  resolves the tie by interval-tree set iteration order: of the 13,307 chr21 queries
  that reach the choice, 45 are tied, and Python picked the earlier difference in 29
  and the later one in 16; the Rust always takes the earliest by `(start, end)`. Sorting the candidates before the `min()` call
  in the Python would make both deterministic and identical.

## v1.0.1

- Preprocess non-normalized equal-length alleles by trimming their shared suffix before liftover, preventing reverse-strand `REF==ALT` and missed REF/ALT swaps.
- Add regression coverage for `AC/CC` and `CAT/TAT` inputs.
