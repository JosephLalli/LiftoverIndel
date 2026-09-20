#include <stdio.h>
#include <stddef.h>
#include "liftover_indels.h"

#define OPT_F(f) printf("  %-18s off=%zu sz=%zu\n", #f, offsetof(liftover_indels_options,f), sizeof(((liftover_indels_options*)0)->f))
#define RES_F(f) printf("  %-18s off=%zu sz=%zu\n", #f, offsetof(liftover_indels_result,f), sizeof(((liftover_indels_result*)0)->f))

int main(void) {
    printf("options size=%zu align=%zu\n", sizeof(liftover_indels_options), _Alignof(liftover_indels_options));
    OPT_F(realign_enabled);
    OPT_F(realign_distance);
    OPT_F(realign_flank);
    OPT_F(realign_max_window);
    OPT_F(threads);
    printf("result size=%zu align=%zu\n", sizeof(liftover_indels_result), _Alignof(liftover_indels_result));
    RES_F(status);
    RES_F(chrom);
    RES_F(pos);
    RES_F(ref_allele);
    RES_F(alt_allele);
    RES_F(flipped);
    RES_F(realigned);
    RES_F(message);
    printf("STATUS OK=%d UNLIFTABLE=%d MULT=%d REFMM=%d ERROR=%d\n",
           LIFTOVER_INDELS_STATUS_OK, LIFTOVER_INDELS_STATUS_UNLIFTABLE,
           LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS, LIFTOVER_INDELS_STATUS_REF_MISMATCH,
           LIFTOVER_INDELS_STATUS_ERROR);
    printf("int=%zu longlong=%zu size_t=%zu ptr=%zu\n",
           sizeof(int), sizeof(long long), sizeof(size_t), sizeof(void*));
    return 0;
}
