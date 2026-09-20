/*
 * liftover_indels -- indel-aware liftover of variants between genome references.
 *
 * C API. Safe to include from C++; the declarations are wrapped in extern "C".
 *
 * Build the library with:
 *     cargo build --release
 * which produces target/release/libliftover_indels.a and .so.
 *
 * Link (static):
 *     c++ app.cpp -I include -L target/release -lliftover_indels \
 *         -lpthread -ldl -lm -lz -lbz2 -llzma -lcurl
 *
 * Ownership
 * ---------
 * Create an engine with liftover_indels_open() and release it with
 * liftover_indels_close(). Loading dominates the cost of a lift, so create one
 * engine and reuse it.
 *
 * liftover_indels_lift() takes a const engine and does not mutate it, so one
 * engine may be shared by several threads lifting concurrently.
 *
 * Every char* this library returns is owned by the caller. Release a result with
 * liftover_indels_result_dispose() and the error string from open() with
 * liftover_indels_string_free().
 *
 * Genotypes
 * ---------
 * This API works at the allele level. When a lift reports flipped != 0 the REF and
 * ALT alleles were swapped, and the caller must rewrite sample genotypes to match.
 * That rewrite needs every record at the position, which only the caller has.
 * Relatedly, a position may only flip once: pass already_flipped != 0 for later
 * variants at a position where an earlier one flipped, and the engine reports a
 * reference mismatch, as the reference implementation does.
 */

#ifndef LIFTOVER_INDELS_H
#define LIFTOVER_INDELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Values for liftover_indels_result.status. */
#define LIFTOVER_INDELS_STATUS_OK 0
/* No unique target coordinate, or a liftover rule declined. */
#define LIFTOVER_INDELS_STATUS_UNLIFTABLE 1
/* The lifted span covered more than one assembly difference. */
#define LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS 2
/* The lifted allele disagreed with the target reference. */
#define LIFTOVER_INDELS_STATUS_REF_MISMATCH 3
/* Bad arguments or an internal failure; see result.message. */
#define LIFTOVER_INDELS_STATUS_ERROR (-1)

/* Tuning for haplotype realignment near assembly differences.
 * Always initialise with liftover_indels_options_init(). */
typedef struct {
    int realign_enabled;          /* non-zero to enable realignment (default 1) */
    long long realign_distance;   /* bp to search for a nearby difference (50)  */
    long long realign_flank;      /* bp added each side of the window (20)      */
    long long realign_max_window; /* hard cap on the window in bp (200)         */
    int threads;                  /* reader threads for the differences file    */
} liftover_indels_options;

typedef struct {
    int status;            /* LIFTOVER_INDELS_STATUS_*                         */
    char *chrom;           /* target contig, NULL unless status is OK          */
    long long pos;         /* 0-based target position, -1 unless status is OK  */
    char *ref_allele;      /* NULL unless status is OK                         */
    char *alt_allele;      /* NULL unless status is OK                         */
    int flipped;           /* non-zero if REF/ALT swapped; rewrite genotypes   */
    int realigned;         /* non-zero if haplotype realignment was used       */
    char *message;         /* reason when status is not OK, else NULL          */
} liftover_indels_result;

/* Opaque loaded liftover. */
typedef struct liftover_indels_engine liftover_indels_engine;

/* Library version. Static storage; do not free. */
const char *liftover_indels_version(void);

/* Fill opts with defaults. No-op if opts is NULL. */
void liftover_indels_options_init(liftover_indels_options *opts);

/*
 * Load a chain file, an assembly-differences VCF/BCF and the target reference.
 *
 * The differences file must be in TARGET assembly coordinates.
 *
 * chroms may be NULL (load every contig) or an array of n_chroms contig names to
 * restrict to, which saves a great deal of memory for a single-chromosome job.
 * opts may be NULL for defaults.
 *
 * Returns NULL on failure. If error is non-NULL it receives an owned message to
 * release with liftover_indels_string_free().
 */
liftover_indels_engine *liftover_indels_open(const char *chain_path,
                                             const char *ref_diffs_path,
                                             const char *target_fasta_path,
                                             const char *const *chroms,
                                             size_t n_chroms,
                                             const liftover_indels_options *opts,
                                             char **error);

/* Release an engine. Safe with NULL. */
void liftover_indels_close(liftover_indels_engine *engine);

/*
 * Lift one variant. pos is 0-based; the alleles are the source assembly's.
 *
 * Writes out and returns its status. out is overwritten wholesale, so dispose of
 * any previous contents first.
 */
int liftover_indels_lift(const liftover_indels_engine *engine,
                         const char *chrom,
                         long long pos,
                         const char *ref_allele,
                         const char *alt_allele,
                         int already_flipped,
                         liftover_indels_result *out);

/* Free the strings in a result and reset it. Safe with NULL and safe twice. */
void liftover_indels_result_dispose(liftover_indels_result *result);

/* Free a string returned by this library. Safe with NULL. */
void liftover_indels_string_free(char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIFTOVER_INDELS_H */
