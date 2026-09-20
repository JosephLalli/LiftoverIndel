// Temporary ABI layout probe: prints the Rust-side repr(C) layout of the FFI
// structs so it can be diffed against the C header's computed layout.
use std::mem::{align_of, offset_of, size_of};

use liftover_indels::ffi::{
    liftover_indels_options, liftover_indels_result, LIFTOVER_INDELS_STATUS_ERROR,
    LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS, LIFTOVER_INDELS_STATUS_OK,
    LIFTOVER_INDELS_STATUS_REF_MISMATCH, LIFTOVER_INDELS_STATUS_UNLIFTABLE,
};

macro_rules! opt {
    ($f:ident) => {
        println!("  {:<18} off={}", stringify!($f), offset_of!(liftover_indels_options, $f))
    };
}
macro_rules! res {
    ($f:ident) => {
        println!("  {:<18} off={}", stringify!($f), offset_of!(liftover_indels_result, $f))
    };
}

fn main() {
    println!(
        "options size={} align={}",
        size_of::<liftover_indels_options>(),
        align_of::<liftover_indels_options>()
    );
    opt!(realign_enabled);
    opt!(realign_distance);
    opt!(realign_flank);
    opt!(realign_max_window);
    opt!(threads);
    println!(
        "result size={} align={}",
        size_of::<liftover_indels_result>(),
        align_of::<liftover_indels_result>()
    );
    res!(status);
    res!(chrom);
    res!(pos);
    res!(ref_allele);
    res!(alt_allele);
    res!(flipped);
    res!(realigned);
    res!(message);
    println!(
        "STATUS OK={} UNLIFTABLE={} MULT={} REFMM={} ERROR={}",
        LIFTOVER_INDELS_STATUS_OK,
        LIFTOVER_INDELS_STATUS_UNLIFTABLE,
        LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS,
        LIFTOVER_INDELS_STATUS_REF_MISMATCH,
        LIFTOVER_INDELS_STATUS_ERROR
    );
    println!(
        "c_int={} c_longlong={} usize={} ptr={}",
        size_of::<std::ffi::c_int>(),
        size_of::<std::ffi::c_longlong>(),
        size_of::<usize>(),
        size_of::<*const u8>()
    );
}
