/*
 * liftover_indels -- indel-aware liftover of variants between genome references.
 *
 * C API. Safe to include from C++; the declarations are wrapped in extern "C".
 *
 * Build the library with:
 *     cargo build --release
 * which produces target/release/libliftover_indels.a and .so.
 *
 * Link (static). The archive carries htslib, which needs the compression
 * libraries; it is built without remote-file support, so curl is NOT required:
 *     c++ app.cpp -I include target/release/libliftover_indels.a \
 *         -lpthread -ldl -lm -lz -lbz2 -llzma
 *
 * Engines
 * -------
 * Create an engine with liftover_indels_open() and release it with
 * liftover_indels_close(). Loading dominates the cost of a lift, so create one
 * engine and reuse it.
 *
 * liftover_indels_lift() takes a const engine and does not mutate it, so one
 * engine may be shared by several threads lifting concurrently.
 *
 * Results and ownership
 * ---------------------
 * A result owns its strings, and they are released ONLY by
 * liftover_indels_result_dispose(), which frees all of them together.
 *
 *   - Never free an individual field of a result. dispose() would then free it a
 *     second time. liftover_indels_string_free() is for exactly one thing: the
 *     error string that liftover_indels_open() writes on failure.
 *
 *   - dispose() frees whatever pointers the struct holds and cannot tell an
 *     uninitialised one from NULL. A result must therefore be initialised before
 *     it is disposed: either by a completed liftover_indels_lift(), or by
 *     liftover_indels_result_init() (or `= {0}`). This matters for the ordinary
 *     goto-cleanup shape, where an early jump can reach dispose() before any lift.
 *
 *   - lift() overwrites the result without freeing what was there, so reusing one
 *     result across lifts LEAKS unless you dispose between them. Either declare
 *     the result inside the loop, or dispose at the end of each iteration.
 *
 *   - Do not modify a returned string in place. They are freed by recomputing
 *     their length, so truncating one (writing a NUL into it, strtok, ...) makes
 *     the later free wrong. Copy first if you need to edit.
 *
 * Genotypes
 * ---------
 * This API works at the allele level. When a lift reports flipped != 0 the REF and
 * ALT alleles were swapped, and the caller must rewrite sample genotypes to match.
 * That rewrite needs every record at the position, which only the caller has.
 *
 * flipped is only meaningful when status is LIFTOVER_INDELS_STATUS_OK. A variant
 * that flipped and then failed its reference check reports REF_MISMATCH with
 * flipped == 0, so a caller tracking flips per position sees only successful ones.
 *
 * already_flipped rejects a SECOND flip at a position: pass it non-zero for later
 * variants at a position where an earlier one flipped, and a variant that would
 * itself flip is reported as REF_MISMATCH. It does not force every call to fail --
 * a variant that needs no flip lifts normally whatever you pass.
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
                                  /* (default 2; a value below 1 is taken as 1) */
} liftover_indels_options;

typedef struct {
    int status;            /* LIFTOVER_INDELS_STATUS_*                         */
    char *chrom;           /* target contig, NULL unless status is OK          */
    long long pos;         /* 0-based target position, -1 unless status is OK  */
    char *ref_allele;      /* NULL unless status is OK                         */
    char *alt_allele;      /* NULL unless status is OK                         */
    int flipped;           /* REF/ALT swapped; only meaningful when status OK  */
    int realigned;         /* haplotype realignment was used; OK only          */
    char *message;         /* reason when status is not OK, else NULL          */
} liftover_indels_result;

/* Opaque loaded liftover. */
typedef struct liftover_indels_engine liftover_indels_engine;

/* Library version. Static storage; do not free. */
const char *liftover_indels_version(void);

/* Fill opts with defaults. No-op if opts is NULL. */
void liftover_indels_options_init(liftover_indels_options *opts);

/* Put a result into the initialised empty state, so it is safe to dispose before
 * any lift has written it. No-op if result is NULL. */
void liftover_indels_result_init(liftover_indels_result *result);

/*
 * Load a chain file, an assembly-differences VCF/BCF and the target reference.
 *
 * The differences file must be in TARGET assembly coordinates.
 *
 * chroms restricts the load, which saves a great deal of memory for a
 * single-chromosome job. Pass NULL, or n_chroms == 0, to load every contig.
 * opts may be NULL for defaults.
 *
 * Returns NULL on failure. If error is non-NULL it receives an owned message to
 * release with liftover_indels_string_free(). Paths must be valid UTF-8.
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
 * Writes out and returns its status. out is overwritten without freeing what it
 * held, so dispose of a reused result first (see "Results and ownership").
 *
 * A NULL engine or a NULL string argument returns LIFTOVER_INDELS_STATUS_ERROR
 * with an explanatory message rather than crashing; a NULL out returns
 * LIFTOVER_INDELS_STATUS_ERROR and writes nothing.
 */
int liftover_indels_lift(const liftover_indels_engine *engine,
                         const char *chrom,
                         long long pos,
                         const char *ref_allele,
                         const char *alt_allele,
                         int already_flipped,
                         liftover_indels_result *out);

/*
 * Free all four strings in a result and reset it. Safe with NULL, and safe to
 * call twice. The result must have been initialised first, by a lift or by
 * liftover_indels_result_init().
 *
 * Resetting sets status back to LIFTOVER_INDELS_STATUS_ERROR and pos to -1, so
 * read what you need from a result before disposing it.
 */
void liftover_indels_result_dispose(liftover_indels_result *result);

/*
 * Free the error string returned by liftover_indels_open(). Safe with NULL.
 * Do not pass a field of a liftover_indels_result: use
 * liftover_indels_result_dispose() for those.
 */
void liftover_indels_string_free(char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIFTOVER_INDELS_H */
