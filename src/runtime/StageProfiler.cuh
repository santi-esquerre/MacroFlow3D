/**
 * @file StageProfiler.cuh
 * @brief Lightweight profiler for measuring stage times in the physics pipeline
 *
 * ## Usage
 *
 *   #include "runtime/StageProfiler.cuh"
 *
 *   StageProfiler profiler(ctx);
 *
 *   profiler.start("K_generation");
 *   generate_K_field(...);
 *   profiler.stop();
 *
 *   profiler.start("solve_head");
 *   solve_head(...);
 *   profiler.stop();
 *
 *   profiler.report();  // Prints timing summary
 *
 * ## Overhead
 *
 * When MACROFLOW3D_ENABLE_PROFILING=0, all operations are no-ops (zero overhead).
 * When enabled, each start/stop pair has ~1μs overhead from event recording.
 *
 * ## Design
 *
 * - Uses CUDA events for accurate GPU timing (not wall-clock)
 * - Stores stage names and accumulated times
 * - Thread-safe within a single stream (not across streams)
 * - No dynamic allocation in start/stop (only in constructor/report)
 */

#pragma once

#include "cuda_check.cuh"
#include "CudaContext.cuh"
#include "diagnostics.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace macroflow3d {

/**
 * @brief Stage timing record
 */
struct StageRecord {
    std::string name;
    float total_ms = 0.0f;
    int count = 0;
};

/**
 * @brief Simple GPU profiler using CUDA events
 *
 * Measures time spent in named stages. All timing is GPU-side (via events).
 */
class StageProfiler {
  public:
    explicit StageProfiler(CudaContext& ctx) : ctx_(ctx), enabled_(MACROFLOW3D_PROFILING_ENABLED) {
#if MACROFLOW3D_ENABLE_PROFILING
        MACROFLOW3D_CUDA_CHECK(cudaEventCreate(&start_event_));
        MACROFLOW3D_CUDA_CHECK(cudaEventCreate(&stop_event_));
#endif
    }

    ~StageProfiler() {
#if MACROFLOW3D_ENABLE_PROFILING
        if (start_event_)
            cudaEventDestroy(start_event_);
        if (stop_event_)
            cudaEventDestroy(stop_event_);
#endif
    }

    // Non-copyable, non-movable (holds reference to CudaContext)
    StageProfiler(const StageProfiler&) = delete;
    StageProfiler& operator=(const StageProfiler&) = delete;
    StageProfiler(StageProfiler&&) = delete;
    StageProfiler& operator=(StageProfiler&&) = delete;

    /**
     * @brief Start timing a named stage
     *
     * Must be followed by stop() before next start().
     *
     * @param name Stage name (stored for reporting)
     */
    void start(const char* name) {
#if MACROFLOW3D_ENABLE_PROFILING
        if (!enabled_)
            return;
        current_stage_ = name;
        MACROFLOW3D_CUDA_CHECK(cudaEventRecord(start_event_, ctx_.cuda_stream()));
#else
        (void)name;
#endif
    }

    /**
     * @brief Stop timing the current stage
     *
     * Records elapsed time since last start().
     */
    void stop() {
#if MACROFLOW3D_ENABLE_PROFILING
        if (!enabled_ || current_stage_.empty())
            return;

        MACROFLOW3D_CUDA_CHECK(cudaEventRecord(stop_event_, ctx_.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float elapsed_ms = 0.0f;
        MACROFLOW3D_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_));

        // Find or create stage record
        auto it = stage_index_.find(current_stage_);
        if (it == stage_index_.end()) {
            stage_index_[current_stage_] = stages_.size();
            stages_.push_back({current_stage_, elapsed_ms, 1});
        } else {
            stages_[it->second].total_ms += elapsed_ms;
            stages_[it->second].count += 1;
        }

        current_stage_.clear();
#endif
    }

    /**
     * @brief Print timing report to stdout
     *
     * Shows:
     * - Each stage: name, total time, call count, avg time
     * - Total time across all stages
     * - Percentage breakdown
     */
    void report() const {
#if MACROFLOW3D_ENABLE_PROFILING
        if (!enabled_ || stages_.empty())
            return;

        // Calculate total time
        float total_ms = 0.0f;
        for (const auto& s : stages_) {
            total_ms += s.total_ms;
        }

        // Get GPU memory info
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        float used_mb = static_cast<float>(total_mem - free_mem) / (1024.0f * 1024.0f);
        float total_mb = static_cast<float>(total_mem) / (1024.0f * 1024.0f);

        // Print header
        std::printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
        std::printf("║                   STAGE PROFILING REPORT                      ║\n");
        std::printf("╠═══════════════════════════════════════════════════════════════╣\n");
        std::printf("║ %-25s │ %8s │ %5s │ %8s │ %5s ║\n", "Stage", "Total(ms)", "Count", "Avg(ms)",
                    "%");
        std::printf("╠═══════════════════════════════════════════════════════════════╣\n");

        // Print each stage
        for (const auto& s : stages_) {
            float avg_ms = s.total_ms / std::max(1, s.count);
            float pct = (total_ms > 0.0f) ? (s.total_ms / total_ms * 100.0f) : 0.0f;
            std::printf("║ %-25s │ %8.2f │ %5d │ %8.2f │ %5.1f ║\n", s.name.c_str(), s.total_ms,
                        s.count, avg_ms, pct);
        }

        std::printf("╠═══════════════════════════════════════════════════════════════╣\n");
        std::printf("║ %-25s │ %8.2f │ %5s │ %8s │ %5s ║\n", "TOTAL", total_ms, "-", "-", "100%");
        std::printf("╠═══════════════════════════════════════════════════════════════╣\n");
        std::printf("║ GPU Memory: %.1f MB used / %.1f MB total (%.1f%% used)      ║\n", used_mb,
                    total_mb, (used_mb / total_mb * 100.0f));
        std::printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
#endif
    }

    /**
     * @brief Get total time for a specific stage (ms)
     */
    float get_stage_time(const char* name) const {
#if MACROFLOW3D_ENABLE_PROFILING
        auto it = stage_index_.find(name);
        if (it != stage_index_.end()) {
            return stages_[it->second].total_ms;
        }
#else
        (void)name;
#endif
        return 0.0f;
    }

    /**
     * @brief Reset all accumulated timings
     */
    void reset() {
#if MACROFLOW3D_ENABLE_PROFILING
        stages_.clear();
        stage_index_.clear();
        current_stage_.clear();
#endif
    }

    /**
     * @brief Enable/disable profiling at runtime
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }

    bool is_enabled() const { return enabled_; }

  private:
    CudaContext& ctx_;
    bool enabled_;

#if MACROFLOW3D_ENABLE_PROFILING
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
    std::string current_stage_;
    std::vector<StageRecord> stages_;
    std::unordered_map<std::string, size_t> stage_index_;
#endif
};

/**
 * @brief RAII scope guard for stage timing
 *
 * Usage:
 *   {
 *       ScopedStage stage(profiler, "compute_velocity");
 *       compute_velocity_from_head(...);
 *   }  // automatically calls profiler.stop()
 */
class ScopedStage {
  public:
    ScopedStage(StageProfiler& profiler, const char* name) : profiler_(profiler) {
        profiler_.start(name);
    }

    ~ScopedStage() { profiler_.stop(); }

    ScopedStage(const ScopedStage&) = delete;
    ScopedStage& operator=(const ScopedStage&) = delete;

  private:
    StageProfiler& profiler_;
};

// Token pasting helpers (guard against redefinition)
#ifndef MACROFLOW3D_CAT
#define MACROFLOW3D_CAT_INNER(a, b) a##b
#define MACROFLOW3D_CAT(a, b) MACROFLOW3D_CAT_INNER(a, b)
#endif

// Convenience macro for scoped profiling
#if MACROFLOW3D_ENABLE_PROFILING
#define MACROFLOW3D_PROFILE_STAGE(profiler, name)                                                  \
    ::macroflow3d::ScopedStage MACROFLOW3D_CAT(__stage_, __LINE__)(profiler, name)
#else
#define MACROFLOW3D_PROFILE_STAGE(profiler, name)                                                  \
    do {                                                                                           \
    } while (0)
#endif

} // namespace macroflow3d
