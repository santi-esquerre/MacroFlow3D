#pragma once

/**
 * @file IOScheduler.hpp
 * @brief Single point of decision for when and how to write I/O.
 *
 * The scheduler owns:
 *   - Cadence config (snapshot_every, stats_every, flush_every)
 *   - A Par2SnapshotAdapter for format-parity snapshot output
 *   - Pending write queue (fixed-capacity, no hot-loop allocs)
 *
 * API:
 *   1) Construct with config → creates Par2-backed snapshot adapter.
 *   2) Per step: call on_step() → scheduler decides if anything is needed.
 *   3) After realization: call end_realization() → flush stats.
 *
 * HPC contract:
 *   - ZERO allocations inside on_step().
 *   - D2H copies delegated to Par2SnapshotAdapter (pinned mem + stream).
 *   - Disk I/O only happens on snapshot/stats write events.
 */

#include "PinnedHostBuffer.hpp"
#include "../../io/output_layout.hpp"
#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../physics/particles/par2_adapter/Par2SnapshotAdapter.hpp"
#include "../../physics/particles/par2_adapter/par2_views.hpp"
#include "../../core/Scalar.hpp"

#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>

namespace macroflow3d {
namespace runtime {

using namespace macroflow3d::physics::particles;

/**
 * @brief Cadence + formatting configuration for the I/O scheduler.
 *
 * snap_writer mirrors SnapshotWriterConfig from par2_views.hpp, ensuring
 * identical snapshot options are available for both Par2 and PSPTA paths.
 */
struct SchedulerConfig {
    // Stats (moments) sampling cadence (transport steps)
    int stats_every        = 0;   // 0 = disabled

    // Snapshot cadence
    int snapshot_every     = 0;   // 0 = disabled

    // First-realization dense snapshots (r=0 only)
    int r0_snapshot_every  = 0;   // 0 = disabled

    // Total step count — used by maybe_write_final() to detect coverage gaps
    int n_steps            = 0;

    // Number of particles
    int n_particles        = 0;

    // Does the domain have periodic BCs?
    bool has_periodic      = false;

    // Per-snapshot formatting/column options (Par2-compatible superset)
    SnapshotWriterConfig snap_writer;
};

/**
 * @brief Pending snapshot write request (host-side data already staged).
 *
 * The staging buffers are owned; the request holds an index into a
 * pre-allocated ring buffer, so no allocation happens.
 */
struct PendingSnapshot {
    std::string filename;   // pre-formatted path
    real time;
    int  step;
    bool has_unwrapped;
    int  n;                 // number of particles
    int  stride;
    int  precision;
    // Data is in the staging buffer at the time of flush
};

/**
 * @brief Central I/O scheduler.
 *
 * Typical usage:
 *   IOScheduler sched(sched_cfg, layout);
 *   sched.begin_realization(r);
 *   for (step ...) {
 *       engine.step(dt);
 *       sched.on_step(step, dt, particles_view, unwrap_view,
 *                     compute_unwrapped_fn, stream);
 *   }
 *   sched.end_realization();
 */
class IOScheduler {
public:
    IOScheduler(const SchedulerConfig& cfg,
                const io::OutputLayout& layout)
        : cfg_(cfg), layout_(layout)
    {
        // Construct Par2-backed snapshot adapter ONCE (owns its own
        // pinned staging buffers inside the PIMPL; no hot-loop allocs).
        if (cfg_.n_particles > 0 &&
            (cfg_.snapshot_every > 0 || cfg_.r0_snapshot_every > 0))
        {
            snap_adapter_ = std::make_unique<Par2SnapshotAdapter>(
                cfg_.n_particles, cfg_.snap_writer);
        }

        // Pre-reserve stats series (no realloc in hot loop)
        if (cfg_.stats_every > 0) {
            stats_series_.reserve(4096);
        }
    }

    /// Signal start of a realization — reset per-realization state
    void begin_realization(int r) {
        current_r_ = r;
        stats_series_.clear();   // .clear() does NOT free, capacity retained
        pending_snaps_.clear();
    }

    // ── Cadence helpers (side-effect-free) ──────────────────────────────────

    /// True if a regular snapshot should be written at this step.
    bool snapshot_due(int step) const {
        return (cfg_.snapshot_every > 0) && (step % cfg_.snapshot_every == 0);
    }

    /// True if an r=0 dense snapshot should be written at this step.
    bool r0_snapshot_due(int step) const {
        return (current_r_ == 0) &&
               (cfg_.r0_snapshot_every > 0) &&
               (step % cfg_.r0_snapshot_every == 0);
    }

    // ── Pre-sync staging (call BEFORE the stream sync) ───────────────────────

    /**
     * @brief No-op.  Retained for API compatibility.
     *
     * Par2SnapshotAdapter handles its own D2H staging internally,
     * so pre-staging from the caller is no longer necessary.
     */
    void stage_snapshot_async(int /*step*/,
                              const ConstParticlesSoA<real>& /*parts*/,
                              const UnwrappedSoA<real>& /*unwrap*/,
                              cudaStream_t /*stream*/)
    {
        // No-op: Par2SnapshotAdapter stages D2H inside write_snapshot().
    }

    // ── Main hot-loop entry point ────────────────────────────────────────────

    /**
     * @brief Called once per transport step.  Decides whether to sample stats
     *        or write a snapshot.
     *
     * @param pre_synced  If true, the caller has already issued a
     *                    cudaStreamSynchronize AND called stage_snapshot_async().
     *                    on_step() will skip re-staging and skip re-sync —
     *                    this gives a single sync per event step (Task 3).
     *                    When false (default), legacy path: on_step stages and
     *                    syncs internally (correct but two-sync on event steps).
     *
     * HPC contract: NO allocations in hot path.
     */
    void on_step(int step, real dt,
                 const ConstParticlesSoA<real>& particles,
                 const UnwrappedSoA<real>& unwrap,
                 const io::TimeSeriesPoint<real>* stats_sample,
                 cudaStream_t stream,
                 bool pre_synced = false)
    {
        // ── Stats ────────────────────────────────────────────────────
        if (cfg_.stats_every > 0 && (step % cfg_.stats_every == 0) && stats_sample) {
            stats_series_.push_back(*stats_sample);
        }

        // ── Snapshots ───────────────────────────────────────────────
        const bool snap_now = snapshot_due(step);
        const bool r0_snap  = r0_snapshot_due(step);

        if (snap_now || r0_snap) {
            if (!pre_synced) {
                // Ensure any prior kernel work is complete before the
                // adapter launches its own D2H copies.
                cudaStreamSynchronize(stream);
            }
            // Write via Par2SnapshotAdapter (format parity — Task 1)
            if (r0_snap) {
                write_snapshot_now(layout_.r0_snapshot(step),
                                   particles, unwrap, step * dt,
                                   false, stream);  // r0: wrapped only
            }
            if (snap_now) {
                write_snapshot_now(layout_.snapshot(current_r_, step),
                                   particles, unwrap, step * dt,
                                   cfg_.snap_writer.include_unwrapped,
                                   stream);
            }
        }
    }

    /**
     * @brief Write a final snapshot if the last step was not already covered
     *        by the regular cadence.
     *
     * Call AFTER the hot loop and eng.synchronize() so all positions are final.
     * Performs its own sync after staging if a snapshot is needed.
     *
     * No-op if snapshot_every <= 0 or n_steps % snapshot_every == 0.
     */
    void maybe_write_final(real dt,
                           const ConstParticlesSoA<real>& parts,
                           const UnwrappedSoA<real>& unwrap,
                           cudaStream_t stream)
    {
        const int n = cfg_.n_steps;
        if (cfg_.snapshot_every <= 0 || n <= 0) return;
        if (n % cfg_.snapshot_every == 0) return;   // already written by cadence

        write_snapshot_now(layout_.snapshot(current_r_, n),
                           parts, unwrap, static_cast<real>(n) * dt,
                           cfg_.snap_writer.include_unwrapped,
                           stream);
        std::printf("       Wrote final snapshot at step %d (r=%d)\n", n, current_r_);
    }

    /// End of realization — flush stats time-series to disk
    void end_realization() {
        if (cfg_.stats_every > 0 && !stats_series_.empty()) {
            std::string fname = layout_.realization_timeseries(current_r_);
            io::CsvTimeSeriesWriter::write(fname, stats_series_);
            std::printf("       Wrote %s (%d samples)\n",
                        fname.c_str(), (int)stats_series_.size());
        }
    }

    /// Access the current realization's time-series (for post-processing)
    const std::vector<io::TimeSeriesPoint<real>>& stats_series() const {
        return stats_series_;
    }

private:
    /**
     * @brief Write one snapshot via Par2SnapshotAdapter.
     *
     * The adapter takes device pointers and handles D2H + disk-write
     * internally, producing the same CSV layout as par2::io::CsvSnapshotWriter.
     */
    void write_snapshot_now(const std::string& filename,
                            const ConstParticlesSoA<real>& particles,
                            const UnwrappedSoA<real>& unwrap,
                            real time,
                            bool write_unwrapped,
                            cudaStream_t stream)
    {
        if (!snap_adapter_) return;
        // Ensure parent directory exists (par2 writer may not create it)
        auto parent = std::filesystem::path(filename).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        snap_adapter_->write_snapshot(
            particles, filename.c_str(), time, stream,
            write_unwrapped ? &unwrap : nullptr);
    }

    SchedulerConfig cfg_;
    io::OutputLayout layout_;
    int current_r_ = 0;

    // Par2-backed snapshot writer (owns its own pinned staging buffers).
    // Constructed once in the constructor; reused across all realizations.
    std::unique_ptr<Par2SnapshotAdapter> snap_adapter_;

    // Stats series (capacity reserved in constructor, .clear() reuses memory)
    std::vector<io::TimeSeriesPoint<real>> stats_series_;

    // Pending snapshot queue (future batched writes — currently unused)
    std::vector<PendingSnapshot> pending_snaps_;
};

} // namespace runtime
} // namespace macroflow3d
