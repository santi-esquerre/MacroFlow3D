#pragma once

/**
 * @file output_layout.hpp
 * @brief Stable output directory layout for RWPT runs.
 *
 * All paths are relative to a run's base output directory.
 * Provides consistent naming conventions and format versioning.
 */

#include <cstdio>
#include <filesystem>
#include <string>

namespace macroflow3d {
namespace io {

/// Format version for the output layout. Bump when CSV schema changes.
inline constexpr int kOutputFormatVersion = 1;

/**
 * @brief Stable output path generator.
 *
 * Layout:
 *   <base>/manifest.json
 *   <base>/stats/realization_XXXX_timeseries.csv
 *   <base>/snapshots/step_XXXXXXXX.csv  (r=0 only)
 *   <base>/snapshots/r_XXXX/step_XXXXXXXX.csv  (r>0)
 *   <base>/ensemble/ensemble_timeseries.csv
 *   <base>/analysis/macrodispersion.csv
 */
struct OutputLayout {
    std::string base;

    explicit OutputLayout(const std::string& base_dir) : base(base_dir) {}

    // ── Sub-directories ──────────────────────────────────────────────
    std::string stats_dir() const { return base + "/stats"; }
    std::string snap_dir() const { return base + "/snapshots"; }
    std::string ensemble_dir() const { return base + "/ensemble"; }
    std::string analysis_dir() const { return base + "/analysis"; }

    // ── File paths ───────────────────────────────────────────────────

    std::string manifest() const { return base + "/manifest.json"; }

    std::string effective_config() const { return base + "/effective_config.yaml"; }

    std::string realization_timeseries(int r) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/stats/realization_%04d_timeseries.csv", base.c_str(),
                      r);
        return buf;
    }

    std::string snapshot(int r, int step) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/snapshots/r_%04d/step_%08d.csv", base.c_str(), r, step);
        return buf;
    }

    std::string r0_snapshot(int step) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/snapshots/step_%08d.csv", base.c_str(), step);
        return buf;
    }

    std::string ensemble_timeseries() const { return base + "/ensemble/ensemble_timeseries.csv"; }

    std::string macrodispersion_csv() const { return base + "/analysis/macrodispersion.csv"; }

    // ── Directory creation ───────────────────────────────────────────

    void ensure_all_dirs() const {
        std::filesystem::create_directories(stats_dir());
        std::filesystem::create_directories(snap_dir());
        std::filesystem::create_directories(ensemble_dir());
        std::filesystem::create_directories(analysis_dir());
    }

    void ensure_snapshot_dir(int r) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/snapshots/r_%04d", base.c_str(), r);
        std::filesystem::create_directories(buf);
    }
};

} // namespace io
} // namespace macroflow3d
