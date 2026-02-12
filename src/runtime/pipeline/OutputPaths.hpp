#pragma once

/**
 * @file OutputPaths.hpp
 * @brief Output path helpers and printing utilities for the RWPT pipeline
 */

#include <cstdio>
#include <string>
#include <filesystem>

namespace macroflow3d {
namespace pipeline {

// ============================================================================
// Directory helpers
// ============================================================================

inline void ensure_dir(const std::string& path) {
    std::filesystem::create_directories(path);
}

// ============================================================================
// Consistent output paths
// ============================================================================

struct OutputPaths {
    std::string base_dir;

    explicit OutputPaths(const std::string& dir) : base_dir(dir) {}

    std::string stats_csv(int r) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/stats_r%03d.csv", base_dir.c_str(), r);
        return buf;
    }

    std::string r0_snapshot(int step) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/particles_r0_s%06d.csv", base_dir.c_str(), step);
        return buf;
    }

    std::string snapshot(int r, int step, int NR) const {
        char buf[512];
        if (NR > 1)
            std::snprintf(buf, sizeof(buf), "%s/snapshot_r%03d_s%06d.csv", base_dir.c_str(), r, step);
        else
            std::snprintf(buf, sizeof(buf), "%s/snapshot_s%06d.csv", base_dir.c_str(), step);
        return buf;
    }

    std::string ensemble_mean() const {
        return base_dir + "/stats_mean_over_NR.csv";
    }

    std::string alpha_csv() const {
        return base_dir + "/alpha_over_NR.csv";
    }
};

// ============================================================================
// Printing helpers
// ============================================================================

inline void print_separator() {
    std::printf("────────────────────────────────────────────────────────\n");
}

} // namespace pipeline
} // namespace macroflow3d
