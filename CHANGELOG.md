# Changelog

## Unreleased

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
