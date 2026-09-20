//! C ABI for the liftover engine, so C and C++ callers can use it.
//!
//! The matching header is `include/liftover_indels.h`.
//!
//! Lifetime and ownership rules, which the header repeats:
//!
//! * An engine is created by [`liftover_indels_open`] and must be released with
//!   [`liftover_indels_close`].
//! * [`liftover_indels_lift`] takes a `const` engine and does not mutate it, so a
//!   single engine may be shared across threads and lifted from concurrently.
//! * Every `char *` handed back — the strings inside a result, and the error string
//!   from `open` — is owned by the caller. Release a result with
//!   [`liftover_indels_result_dispose`] and an error string with
//!   [`liftover_indels_string_free`].
//!
//! Panics are caught at the boundary and reported as
//! `LIFTOVER_INDELS_STATUS_ERROR` rather than unwinding into foreign frames,
//! which would be undefined behaviour.

// The exported types carry their C names on purpose, so they read the same in the
// header and in this file.
#![allow(non_camel_case_types)]

use std::collections::HashSet;
use std::ffi::{c_char, c_int, c_longlong, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::engine::{Engine, Outcome};
use crate::realign::RealignConfig;

pub const LIFTOVER_INDELS_STATUS_OK: c_int = 0;
pub const LIFTOVER_INDELS_STATUS_UNLIFTABLE: c_int = 1;
pub const LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS: c_int = 2;
pub const LIFTOVER_INDELS_STATUS_REF_MISMATCH: c_int = 3;
pub const LIFTOVER_INDELS_STATUS_ERROR: c_int = -1;

/// Tuning for haplotype realignment. Initialise with
/// [`liftover_indels_options_init`] so new fields keep their defaults.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct liftover_indels_options {
    /// Non-zero to enable realignment near assembly differences (default 1).
    pub realign_enabled: c_int,
    pub realign_distance: c_longlong,
    pub realign_flank: c_longlong,
    pub realign_max_window: c_longlong,
    /// Reader threads for the assembly-differences file.
    pub threads: c_int,
}

impl Default for liftover_indels_options {
    fn default() -> Self {
        let d = RealignConfig::default();
        liftover_indels_options {
            realign_enabled: c_int::from(d.enabled),
            realign_distance: d.distance as c_longlong,
            realign_flank: d.flank as c_longlong,
            realign_max_window: d.max_window as c_longlong,
            threads: 2,
        }
    }
}

/// The outcome of lifting one variant. All `char *` fields are owned by the
/// caller and released by [`liftover_indels_result_dispose`].
#[repr(C)]
#[derive(Debug)]
pub struct liftover_indels_result {
    /// One of the `LIFTOVER_INDELS_STATUS_*` values.
    pub status: c_int,
    /// Target contig, or NULL unless `status` is OK.
    pub chrom: *mut c_char,
    /// 0-based target position, or -1 unless `status` is OK.
    pub pos: c_longlong,
    pub ref_allele: *mut c_char,
    pub alt_allele: *mut c_char,
    /// Non-zero when REF and ALT were swapped. The caller must rewrite genotypes.
    pub flipped: c_int,
    /// Non-zero when haplotype realignment produced the representation.
    pub realigned: c_int,
    /// Reason for a non-OK status, else NULL.
    pub message: *mut c_char,
}

impl liftover_indels_result {
    fn empty() -> Self {
        liftover_indels_result {
            status: LIFTOVER_INDELS_STATUS_ERROR,
            chrom: ptr::null_mut(),
            pos: -1,
            ref_allele: ptr::null_mut(),
            alt_allele: ptr::null_mut(),
            flipped: 0,
            realigned: 0,
            message: ptr::null_mut(),
        }
    }
}

/// Opaque loaded liftover.
pub struct liftover_indels_engine {
    inner: Engine,
}

fn to_c_string(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `p` must be NULL or a valid NUL-terminated string.
unsafe fn borrow_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

unsafe fn free_c_string(p: *mut c_char) {
    if !p.is_null() {
        drop(CString::from_raw(p));
    }
}

/// Library version, a static string the caller must not free.
#[no_mangle]
pub extern "C" fn liftover_indels_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Fill `opts` with defaults. Does nothing if `opts` is NULL.
///
/// # Safety
/// `opts` must be NULL or point to a writable `liftover_indels_options`.
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_options_init(opts: *mut liftover_indels_options) {
    if opts.is_null() {
        return;
    }
    ptr::write(opts, liftover_indels_options::default());
}

/// Load a chain file, an assembly-differences VCF/BCF in *target* coordinates and
/// the target reference FASTA.
///
/// `chroms` may be NULL (load every contig) or an array of `n_chroms` contig names
/// to restrict to, which saves a great deal of memory for a single-chromosome job.
/// `opts` may be NULL for defaults.
///
/// Returns NULL on failure; if `error` is non-NULL it receives an owned message
/// that must be released with [`liftover_indels_string_free`].
///
/// # Safety
/// All pointer arguments must be NULL or valid as described above.
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_open(
    chain_path: *const c_char,
    ref_diffs_path: *const c_char,
    target_fasta_path: *const c_char,
    chroms: *const *const c_char,
    n_chroms: usize,
    opts: *const liftover_indels_options,
    error: *mut *mut c_char,
) -> *mut liftover_indels_engine {
    if !error.is_null() {
        *error = ptr::null_mut();
    }
    let set_error = |msg: String| {
        if !error.is_null() {
            *error = to_c_string(&msg);
        }
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let (Some(chain), Some(diffs), Some(fasta)) = (
            borrow_str(chain_path),
            borrow_str(ref_diffs_path),
            borrow_str(target_fasta_path),
        ) else {
            return Err("chain, ref-diffs and target FASTA paths are required".to_string());
        };

        let filter: Option<HashSet<String>> = if chroms.is_null() || n_chroms == 0 {
            None
        } else {
            let mut set = HashSet::with_capacity(n_chroms);
            for i in 0..n_chroms {
                match borrow_str(*chroms.add(i)) {
                    Some(name) => {
                        set.insert(name.to_string());
                    }
                    None => return Err(format!("contig name {i} is not valid UTF-8")),
                }
            }
            Some(set)
        };

        let o = if opts.is_null() {
            liftover_indels_options::default()
        } else {
            *opts
        };
        let realign = RealignConfig {
            enabled: o.realign_enabled != 0,
            debug: false,
            distance: o.realign_distance as i64,
            flank: o.realign_flank as i64,
            max_window: o.realign_max_window as i64,
        };
        let threads = if o.threads > 0 { o.threads as usize } else { 1 };

        Engine::load(chain, diffs, fasta, filter.as_ref(), realign, threads)
    }));

    match result {
        Ok(Ok(inner)) => Box::into_raw(Box::new(liftover_indels_engine { inner })),
        Ok(Err(msg)) => {
            set_error(msg);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("liftover_indels_open panicked".to_string());
            ptr::null_mut()
        }
    }
}

/// Release an engine. Safe to call with NULL.
///
/// # Safety
/// `engine` must be NULL or a pointer from [`liftover_indels_open`] not yet closed.
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_close(engine: *mut liftover_indels_engine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Lift one variant. `pos` is 0-based and the alleles are the source assembly's.
///
/// Pass a non-zero `already_flipped` when another variant at this same position has
/// already had REF and ALT swapped; the engine then reports a reference mismatch,
/// matching the reference implementation's one-flip-per-site rule.
///
/// Writes `out` and returns its status. `out` is overwritten wholesale, so dispose
/// of any previous contents first.
///
/// # Safety
/// `engine` must be a live engine, `out` must be writable, and the string
/// arguments must be valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_lift(
    engine: *const liftover_indels_engine,
    chrom: *const c_char,
    pos: c_longlong,
    ref_allele: *const c_char,
    alt_allele: *const c_char,
    already_flipped: c_int,
    out: *mut liftover_indels_result,
) -> c_int {
    if out.is_null() {
        return LIFTOVER_INDELS_STATUS_ERROR;
    }
    ptr::write(out, liftover_indels_result::empty());
    if engine.is_null() {
        (*out).message = to_c_string("engine is NULL");
        return LIFTOVER_INDELS_STATUS_ERROR;
    }

    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let (Some(c), Some(r), Some(a)) = (
            borrow_str(chrom),
            borrow_str(ref_allele),
            borrow_str(alt_allele),
        ) else {
            return Err("chrom, ref and alt are required".to_string());
        };
        Ok((*engine)
            .inner
            .lift(c, pos as i64, r, a, already_flipped != 0))
    }));

    match outcome {
        Ok(Ok(Outcome::Lifted(l))) => {
            (*out).status = LIFTOVER_INDELS_STATUS_OK;
            (*out).chrom = to_c_string(&l.chrom);
            (*out).pos = l.start as c_longlong;
            (*out).ref_allele = to_c_string(&l.ref_allele);
            (*out).alt_allele = to_c_string(&l.alt_allele);
            (*out).flipped = c_int::from(l.flipped);
            (*out).realigned = c_int::from(l.realigned);
        }
        Ok(Ok(Outcome::Unliftable(m))) => {
            (*out).status = LIFTOVER_INDELS_STATUS_UNLIFTABLE;
            (*out).message = to_c_string(m);
        }
        Ok(Ok(Outcome::MultipleOverlaps)) => {
            (*out).status = LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS;
            (*out).message = to_c_string("lifted span covers multiple assembly differences");
        }
        Ok(Ok(Outcome::Mismatch(m))) => {
            (*out).status = LIFTOVER_INDELS_STATUS_REF_MISMATCH;
            (*out).message = to_c_string(m);
        }
        Ok(Err(msg)) => {
            (*out).status = LIFTOVER_INDELS_STATUS_ERROR;
            (*out).message = to_c_string(&msg);
        }
        Err(_) => {
            (*out).status = LIFTOVER_INDELS_STATUS_ERROR;
            (*out).message = to_c_string("liftover_indels_lift panicked");
        }
    }
    (*out).status
}

/// Free the strings inside a result and reset it. Safe to call twice.
///
/// # Safety
/// `result` must be NULL or point to a result written by [`liftover_indels_lift`].
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_result_dispose(result: *mut liftover_indels_result) {
    if result.is_null() {
        return;
    }
    free_c_string((*result).chrom);
    free_c_string((*result).ref_allele);
    free_c_string((*result).alt_allele);
    free_c_string((*result).message);
    ptr::write(result, liftover_indels_result::empty());
}

/// Free a string returned by this library. Safe to call with NULL.
///
/// # Safety
/// `s` must be NULL or a string this library returned and that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn liftover_indels_string_free(s: *mut c_char) {
    free_c_string(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_rust_configuration() {
        let mut o = liftover_indels_options::default();
        let d = RealignConfig::default();
        assert_eq!(o.realign_enabled, 1);
        assert_eq!(o.realign_distance, d.distance as c_longlong);
        assert_eq!(o.realign_flank, d.flank as c_longlong);
        assert_eq!(o.realign_max_window, d.max_window as c_longlong);
        // options_init writes the same thing.
        let mut z = liftover_indels_options {
            realign_enabled: 9,
            realign_distance: 9,
            realign_flank: 9,
            realign_max_window: 9,
            threads: 9,
        };
        unsafe { liftover_indels_options_init(&mut z) };
        o.threads = z.threads;
        assert_eq!(format!("{z:?}"), format!("{o:?}"));
    }

    #[test]
    fn null_arguments_are_reported_not_crashed() {
        let mut res = liftover_indels_result::empty();
        // NULL engine
        let rc = unsafe {
            liftover_indels_lift(
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                0,
                &mut res,
            )
        };
        assert_eq!(rc, LIFTOVER_INDELS_STATUS_ERROR);
        assert!(!res.message.is_null());
        unsafe { liftover_indels_result_dispose(&mut res) };
        assert!(res.message.is_null());
        // disposing twice is harmless
        unsafe { liftover_indels_result_dispose(&mut res) };
    }

    #[test]
    fn open_reports_missing_paths_without_panicking() {
        let mut err: *mut c_char = ptr::null_mut();
        let e = unsafe {
            liftover_indels_open(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                &mut err,
            )
        };
        assert!(e.is_null());
        assert!(!err.is_null());
        unsafe { liftover_indels_string_free(err) };
        // closing NULL is a no-op
        unsafe { liftover_indels_close(ptr::null_mut()) };
    }

    #[test]
    fn version_is_the_crate_version() {
        let v = unsafe { CStr::from_ptr(liftover_indels_version()) };
        assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
