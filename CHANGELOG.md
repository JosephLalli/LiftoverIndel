# Changelog

## Unreleased

- Remove scratch build artifacts committed by mistake: a compiled binary, a
  throwaway C file and a symlink to an absolute host path, none of which anything
  referenced. Ignore `tmp_*/` so it cannot recur.
- Add `liftover_indels_result_init`, so a result declared on the stack can be
  safely disposed on a path that never reaches a lift. Disposing an uninitialised
  result previously freed whatever the stack held.
- Narrow `liftover_indels_string_free` to its one purpose, the error string from
  `liftover_indels_open`. Using it on a field of a result and then disposing that
  result was a use-after-free and a double free; the header now says so, and the
  C++ example expresses the rule as RAII.
- Report a lifted contig or allele carrying an interior NUL as an error rather
  than returning status OK with NULL strings, which contradicted the documented
  invariant and silently corrupted callers that trusted it.
- Name the offending argument when `open` or `lift` is given a NULL or non-UTF-8
  string, instead of blaming the whole set.
- Correct the documented behaviour of `already_flipped`: it rejects a *second*
  flip at a position, and does not make every call at that position fail. Also
  document that `flipped` is meaningful only when the status is OK, that reusing a
  result across lifts leaks unless it is disposed between them, that returned
  strings must not be edited in place, that `n_chroms == 0` loads every contig,
  that the `threads` default is 2 and a value below 1 is taken as 1, and that a
  NULL engine or string argument is reported rather than fatal.
- Drop `-lcurl` from the header's link line; the library is built without htslib's
  remote-file support and never referenced it.
- Assert `Engine: Send + Sync` in a test, so the header's promise that one engine
  may be lifted from concurrently cannot regress unnoticed.
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
