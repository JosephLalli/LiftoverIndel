//! Print the Rust side's C-struct layout, to diff against the C++ side.
//!
//! Paired with examples/cpp/abi_layout.cpp; if the two outputs differ, the header
//! and the library disagree about the ABI.

use liftover_indels::ffi::{liftover_indels_options, liftover_indels_result};

macro_rules! field {
    ($t:ty, $f:ident) => {{
        let base = std::mem::MaybeUninit::<$t>::uninit();
        let p = base.as_ptr();
        let off = unsafe { std::ptr::addr_of!((*p).$f) as usize - p as usize };
        off
    }};
}

fn main() {
    println!(
        "options size={} align={}",
        std::mem::size_of::<liftover_indels_options>(),
        std::mem::align_of::<liftover_indels_options>()
    );
    println!("options.realign_enabled={}", field!(liftover_indels_options, realign_enabled));
    println!("options.realign_distance={}", field!(liftover_indels_options, realign_distance));
    println!("options.realign_flank={}", field!(liftover_indels_options, realign_flank));
    println!("options.realign_max_window={}", field!(liftover_indels_options, realign_max_window));
    println!("options.threads={}", field!(liftover_indels_options, threads));

    println!(
        "result size={} align={}",
        std::mem::size_of::<liftover_indels_result>(),
        std::mem::align_of::<liftover_indels_result>()
    );
    println!("result.status={}", field!(liftover_indels_result, status));
    println!("result.chrom={}", field!(liftover_indels_result, chrom));
    println!("result.pos={}", field!(liftover_indels_result, pos));
    println!("result.ref_allele={}", field!(liftover_indels_result, ref_allele));
    println!("result.alt_allele={}", field!(liftover_indels_result, alt_allele));
    println!("result.flipped={}", field!(liftover_indels_result, flipped));
    println!("result.realigned={}", field!(liftover_indels_result, realigned));
    println!("result.message={}", field!(liftover_indels_result, message));

    println!("status.ok={}", liftover_indels::ffi::LIFTOVER_INDELS_STATUS_OK);
    println!("status.unliftable={}", liftover_indels::ffi::LIFTOVER_INDELS_STATUS_UNLIFTABLE);
    println!("status.multiple={}", liftover_indels::ffi::LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS);
    println!("status.mismatch={}", liftover_indels::ffi::LIFTOVER_INDELS_STATUS_REF_MISMATCH);
    println!("status.error={}", liftover_indels::ffi::LIFTOVER_INDELS_STATUS_ERROR);
}
