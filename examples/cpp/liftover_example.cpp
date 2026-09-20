// Example C++ client for the liftover_indels C API.
//
// Reads tab-separated variants on stdin (chrom, 1-based pos, ref, alt) and writes
// the lifted result for each. Used as the integration test for the C API: its
// output is cross-checked against the command line tool's.
//
// Build (static, no runtime library path needed):
//   c++ -std=c++17 -O2 -I ../../include liftover_example.cpp
//       ../../target/release/libliftover_indels.a -lpthread -ldl -lm
//       -o liftover_example
//
// Run:
//   ./liftover_example <chain> <ref_diffs.bcf> <target.fasta> [contig ...] < variants.tsv

#include "liftover_indels.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// RAII wrappers so a thrown exception or an early return cannot leak.
struct EngineDeleter {
    void operator()(liftover_indels_engine *e) const { liftover_indels_close(e); }
};
using EnginePtr = std::unique_ptr<liftover_indels_engine, EngineDeleter>;

// Owns the strings inside a result for exactly as long as the scope needs them.
class Result {
public:
    Result() { result_.status = LIFTOVER_INDELS_STATUS_ERROR; result_.message = nullptr;
               result_.chrom = nullptr; result_.ref_allele = nullptr;
               result_.alt_allele = nullptr; result_.pos = -1;
               result_.flipped = 0; result_.realigned = 0; }
    ~Result() { liftover_indels_result_dispose(&result_); }
    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;

    liftover_indels_result *get() { return &result_; }
    const liftover_indels_result &operator*() const { return result_; }

    static const char *str(const char *s) { return s ? s : ""; }

private:
    liftover_indels_result result_;
};

const char *status_name(int status) {
    switch (status) {
    case LIFTOVER_INDELS_STATUS_OK: return "OK";
    case LIFTOVER_INDELS_STATUS_UNLIFTABLE: return "UNLIFTABLE";
    case LIFTOVER_INDELS_STATUS_MULTIPLE_OVERLAPS: return "MULTIPLE_OVERLAPS";
    case LIFTOVER_INDELS_STATUS_REF_MISMATCH: return "REF_MISMATCH";
    default: return "ERROR";
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " <chain> <ref_diffs.bcf> <target.fasta> [contig ...] < variants.tsv\n";
        return 2;
    }

    std::vector<std::string> contigs;
    for (int i = 4; i < argc; ++i) contigs.emplace_back(argv[i]);
    std::vector<const char *> contig_ptrs;
    for (const auto &c : contigs) contig_ptrs.push_back(c.c_str());

    liftover_indels_options opts;
    liftover_indels_options_init(&opts);

    char *error = nullptr;
    EnginePtr engine(liftover_indels_open(argv[1], argv[2], argv[3],
                                          contig_ptrs.empty() ? nullptr : contig_ptrs.data(),
                                          contig_ptrs.size(), &opts, &error));
    if (!engine) {
        std::cerr << "liftover_indels_open failed: " << (error ? error : "unknown") << "\n";
        liftover_indels_string_free(error);
        return 1;
    }

    std::cerr << "liftover_indels " << liftover_indels_version() << " loaded\n";

    std::string line;
    long long n = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string chrom, pos_s, ref, alt;
        if (!std::getline(fields, chrom, '\t') || !std::getline(fields, pos_s, '\t') ||
            !std::getline(fields, ref, '\t') || !std::getline(fields, alt, '\t')) {
            std::cerr << "malformed line: " << line << "\n";
            return 1;
        }
        // The file is 1-based; the API is 0-based.
        const long long pos0 = std::strtoll(pos_s.c_str(), nullptr, 10) - 1;

        Result result;
        const int status = liftover_indels_lift(engine.get(), chrom.c_str(), pos0,
                                                ref.c_str(), alt.c_str(),
                                                /*already_flipped=*/0, result.get());

        const liftover_indels_result &r = *result;
        if (status == LIFTOVER_INDELS_STATUS_OK) {
            // Print 1-based to line up with the VCF the tool writes.
            std::printf("OK\t%s\t%lld\t%s\t%s\t%d\t%d\n", Result::str(r.chrom), r.pos + 1,
                        Result::str(r.ref_allele), Result::str(r.alt_allele),
                        r.flipped, r.realigned);
        } else {
            std::printf("%s\t.\t.\t.\t.\t0\t0\t%s\n", status_name(status),
                        Result::str(r.message));
        }
        ++n;
    }

    std::cerr << "lifted " << n << " variants\n";
    return 0;
}
