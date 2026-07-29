# Changelog

## v1.0.1

- Preprocess non-normalized equal-length alleles by trimming their shared suffix before liftover, preventing reverse-strand `REF==ALT` and missed REF/ALT swaps.
- Add regression coverage for `AC/CC` and `CAT/TAT` inputs.
