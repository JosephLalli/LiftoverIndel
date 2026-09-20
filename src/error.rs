//! How a variant can fail to lift.
//!
//! The two kinds map onto the two Python exception paths, and so onto two
//! different sidecar files: `Unliftable` is raised deliberately by the liftover
//! rules, while `Mismatch` stands for the `ValueError`/`AssertionError` raised
//! when a lifted allele disagrees with the target reference.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantError {
    /// Written to `<base>.unliftable.bcf`.
    Unliftable(&'static str),
    /// Written to `<base>.ref_seq_mismatches.bcf`.
    Mismatch(&'static str),
}

pub type Result<T> = std::result::Result<T, VariantError>;
