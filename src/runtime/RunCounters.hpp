#pragma once

/**
 * @file RunCounters.hpp
 * @brief Lightweight event counters for pipeline runs.
 *
 * Tracks: steps, stats samples, snapshots, flushes, bytes written.
 * Zero overhead: counters are plain integers, incremented inline.
 * Summary only printed when MACROFLOW3D_ENABLE_DIAGNOSTICS is ON.
 */

#include <cstdio>
#include <cstdint>
#include "diagnostics.cuh"

namespace macroflow3d {
namespace runtime {

struct RunCounters {
    int64_t total_steps       = 0;
    int64_t total_realizations = 0;
    int64_t stats_samples     = 0;
    int64_t snapshots_written = 0;
    int64_t flushes           = 0;
    int64_t bytes_written     = 0;  // approximate

    void reset() { *this = RunCounters{}; }

    void add_step()       { ++total_steps; }
    void add_realization() { ++total_realizations; }
    void add_stats()      { ++stats_samples; }
    void add_snapshot(int64_t approx_bytes = 0) {
        ++snapshots_written;
        bytes_written += approx_bytes;
    }
    void add_flush(int64_t approx_bytes = 0) {
        ++flushes;
        bytes_written += approx_bytes;
    }

    /// Print summary to stdout — only when diagnostics enabled.
    void report() const {
        if (!MACROFLOW3D_DIAGNOSTICS_ENABLED) return;

        std::printf("\n┌─────────────────────────────────────────────┐\n");
        std::printf("│          RUN EVENT COUNTERS                 │\n");
        std::printf("├─────────────────────────────────────────────┤\n");
        std::printf("│  Realizations:    %10lld                │\n", (long long)total_realizations);
        std::printf("│  Total steps:     %10lld                │\n", (long long)total_steps);
        std::printf("│  Stats samples:   %10lld                │\n", (long long)stats_samples);
        std::printf("│  Snapshots:       %10lld                │\n", (long long)snapshots_written);
        std::printf("│  Flushes:         %10lld                │\n", (long long)flushes);
        std::printf("│  Bytes written:   %10lld  (~%.1f MB)    │\n",
                    (long long)bytes_written,
                    bytes_written / (1024.0 * 1024.0));
        std::printf("└─────────────────────────────────────────────┘\n\n");
    }
};

} // namespace runtime
} // namespace macroflow3d
