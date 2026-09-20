// Print the C side's struct layout, to diff against examples/abi_layout.rs.
//
// If the two outputs differ, include/liftover_indels.h and the compiled library
// disagree about the ABI and every call through the header is suspect.
//
// Build:
//   c++ -std=c++17 -I ../../include abi_layout.cpp -o abi_layout

#include "liftover_indels.h"

#include <cstddef>
#include <cstdio>

int main() {
    std::printf("options size=%zu align=%zu\n", sizeof(liftover_indels_options),
                alignof(liftover_indels_options));
    std::printf("options.realign_enabled=%zu\n", offsetof(liftover_indels_options, realign_enabled));
    std::printf("options.realign_distance=%zu\n", offsetof(liftover_indels_options, realign_distance));
    std::printf("options.realign_flank=%zu\n", offsetof(liftover_indels_options, realign_flank));
    std::printf("options.realign_max_window=%zu\n", offsetof(liftover_indels_options, realign_max_window));
    std::printf("options.threads=%zu\n", offsetof(liftover_indels_options, threads));

    std::printf("result size=%zu align=%zu\n", sizeof(liftover_indels_result),
                alignof(liftover_indels_result));
    std::printf("result.status=%zu\n", offsetof(liftover_indels_result, status));
    std::printf("result.chrom=%zu\n", offsetof(liftover_indels_result, chrom));
    std::printf("result.pos=%zu\n", offsetof(liftover_indels_result, pos));
    std::printf("result.ref_allele=%zu\n", offsetof(liftover_indels_result, ref_allele));
    std::printf("result.alt_allele=%zu\n", offsetof(liftover_indels_result, alt_allele));
    std::printf("result.flipped=%zu\n", offsetof(liftover_indels_result, flipped));
    std::printf("result.realigned=%zu\n", offsetof(liftover_indels_result, realigned));
    std::printf("result.message=%zu\n", offsetof(liftover_indels_result, message));

    std::printf("status.ok=%d\n", LIFTOVER_INDELS_STATUS_OK);
    std::printf("status.unliftable=%d\n", LIFTOVER_INDELS_STATUS_UNLIFTABLE);
    std::printf("status.multiple=%d\n", LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS);
    std::printf("status.mismatch=%d\n", LIFTOVER_INDELS_STATUS_REF_MISMATCH);
    std::printf("status.error=%d\n", LIFTOVER_INDELS_STATUS_ERROR);
    return 0;
}
