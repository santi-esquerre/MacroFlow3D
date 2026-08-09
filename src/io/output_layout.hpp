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
 *   <base>/streamfunctions/r_XXXX/grid_XXXX/{iteration_history,trial_history}.csv
 *   <base>/streamfunctions/r_XXXX/grid_XXXX/stage_history.csv  (SF-17 T03, continuation only)
 *   <base>/streamfunctions/r_XXXX/grid_XXXX/summary.json
 *   <base>/streamfunctions/r_XXXX/grid_XXXX/{u1,u2}.raw
 *   <base>/streamfunctions/r_XXXX/grid_XXXX/fields_meta.json
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

    // ── Streamfunction solver (SF-16 T02) ──────────────────────────────
    // One directory per realization × fine-grid nx, so a run that changes
    // grid resolution across invocations never collides output.
    std::string streamfunction_dir(int r, int nx) const {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/streamfunctions/r_%04d/grid_%04d", base.c_str(), r, nx);
        return buf;
    }

    std::string streamfunction_iteration_history_csv(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/iteration_history.csv";
    }

    std::string streamfunction_trial_history_csv(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/trial_history.csv";
    }

    // Continuation stage history (SF-17 T03): one row per
    // streamfunctions::ContinuationStageRecord, gated by the same
    // `export.iteration_history` switch as `iteration_history.csv`.
    std::string streamfunction_stage_history_csv(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/stage_history.csv";
    }

    std::string streamfunction_summary_json(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/summary.json";
    }

    std::string streamfunction_u1_raw(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/u1.raw";
    }

    std::string streamfunction_u2_raw(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/u2.raw";
    }

    std::string streamfunction_fields_meta_json(int r, int nx) const {
        return streamfunction_dir(r, nx) + "/fields_meta.json";
    }

    void ensure_streamfunction_dir(int r, int nx) const {
        std::filesystem::create_directories(streamfunction_dir(r, nx));
    }

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
