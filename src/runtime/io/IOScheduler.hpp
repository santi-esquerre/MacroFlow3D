#pragma once

/**
 * @file IOScheduler.hpp
 * @brief Single point of decision for when and how to write I/O.
 *
 * The scheduler owns:
 *   - Cadence config (snapshot_every, stats_every, flush_every)
 *   - Snapshot staging buffers (pinned host memory, preallocated)
 *   - Pending write queue (fixed-capacity, no hot-loop allocs)
 *
 * API:
 *   1) Construct with config → allocates staging buffers.
 *   2) Per step: call on_step() → scheduler decides if anything is needed.
 *   3) After realization: call flush() → drain pending writes.
 *
 * HPC contract:
 *   - ZERO allocations inside on_step().
 *   - D2H copies are async (pinned mem + stream).
 *   - Disk I/O only happens in flush() or when the write queue is full.
 */

#include "PinnedHostBuffer.hpp"
#include "../../io/output_layout.hpp"
#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../io/writers/CsvParticleSnapshotWriter.hpp"
#include "../../physics/particles/par2_adapter/par2_views.hpp"
#include "../../core/Scalar.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdio>

namespace macroflow3d {
namespace runtime {

using namespace macroflow3d::physics::particles;

/**
 * @brief Cadence configuration for the I/O scheduler.
 */
struct SchedulerConfig {
    // Stats (moments) sampling cadence (transport steps)
    int stats_every   = 0;   // 0 = disabled

    // Snapshot cadence
    int snapshot_every = 0;  // 0 = disabled

    // First-realization dense snapshots (r=0 only)
    int r0_snapshot_every = 0;  // 0 = disabled

    // Snapshot options
    bool include_unwrapped = false;
    int  snapshot_stride   = 1;
    int  snapshot_precision = 10;

    // Number of particles
    int n_particles = 0;

    // Does the domain have periodic BCs?
    bool has_periodic = false;
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
        // Pre-allocate staging buffers ONCE
        if (cfg_.n_particles > 0 &&
            (cfg_.snapshot_every > 0 || cfg_.r0_snapshot_every > 0))
        {
            staging_.allocate(cfg_.n_particles, cfg_.include_unwrapped);
        }

        // Pre-reserve stats series (avoid realloc in hot loop)
        // Estimate: max reasonable samples per realization
        if (cfg_.stats_every > 0) {
            stats_series_.reserve(4096);
        }
    }

    /// Signal start of a realization — reset per-realization state
    void begin_realization(int r) {
        current_r_ = r;
        stats_series_.clear();   // .clear() does NOT free, capacity retained
        pending_snaps_.clear();  // same
    }

    /**
     * @brief Main hot-loop entry point. Called once per transport step.
     *
     * Decides whether to sample stats or stage a snapshot.
     * NO allocations. Async D2H if snapshot needed.
     *
     * @param step          Current transport step (1-based).
     * @param dt            Timestep size.
     * @param particles     Device-side const view of particle positions.
     * @param unwrap        Device-side unwrapped positions (may be invalid).
     * @param stats_sample  If non-null, the scheduler will read current moments
     *                      from this struct (already computed by the caller).
     * @param stream        CUDA stream for async copies.
     */
    void on_step(int step, real dt,
                 const ConstParticlesSoA<real>& particles,
                 const UnwrappedSoA<real>& unwrap,
                 const io::TimeSeriesPoint<real>* stats_sample,
                 cudaStream_t stream)
    {
        // ── Stats ────────────────────────────────────────────────────
        if (cfg_.stats_every > 0 && (step % cfg_.stats_every == 0) && stats_sample) {
            stats_series_.push_back(*stats_sample);   // capacity pre-reserved
        }

        // ── Snapshots ───────────────────────────────────────────────
        const bool snap_now = (cfg_.snapshot_every > 0) && (step % cfg_.snapshot_every == 0);
        const bool r0_snap  = (current_r_ == 0) &&
                              (cfg_.r0_snapshot_every > 0) &&
                              (step % cfg_.r0_snapshot_every == 0);

        if (snap_now || r0_snap) {
            // Stage D2H async (pinned memory — no alloc)
            staging_.stage_wrapped_async(
                particles.x, particles.y, particles.z,
                particles.n, stream);

            if (cfg_.include_unwrapped && unwrap.valid()) {
                staging_.stage_unwrapped_async(
                    unwrap.x_u, unwrap.y_u, unwrap.z_u,
                    unwrap.capacity, stream);
            }

            // Sync to ensure host buffers are ready before disk write
            cudaStreamSynchronize(stream);

            // Write immediately (cold path — only triggered every N steps)
            if (r0_snap) {
                write_snapshot_now(layout_.r0_snapshot(step),
                                  particles.n, step, step * dt,
                                  false);  // r0: wrapped only
            }
            if (snap_now) {
                write_snapshot_now(layout_.snapshot(current_r_, step),
                                  particles.n, step, step * dt,
                                  cfg_.include_unwrapped);
            }
        }
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
    void write_snapshot_now(const std::string& filename,
                            int n, int step, real time,
                            bool write_unwrapped) {
        io::HostParticleSnapshot<real> snap;
        snap.x = staging_.x.data();
        snap.y = staging_.y.data();
        snap.z = staging_.z.data();
        snap.n = n;
        snap.time = time;
        snap.step = step;
        snap.stride = cfg_.snapshot_stride;
        snap.precision = cfg_.snapshot_precision;

        if (write_unwrapped && staging_.has_unwrapped) {
            snap.x_u = staging_.x_u.data();
            snap.y_u = staging_.y_u.data();
            snap.z_u = staging_.z_u.data();
            snap.has_unwrapped = true;
        }

        io::CsvParticleSnapshotWriter::write(filename, snap);
    }

    SchedulerConfig cfg_;
    io::OutputLayout layout_;
    int current_r_ = 0;

    // Staging (pinned host memory, allocated ONCE in constructor)
    SnapshotStaging<real> staging_;

    // Stats series (capacity reserved in constructor, .clear() reuses memory)
    std::vector<io::TimeSeriesPoint<real>> stats_series_;

    // Pending snapshot queue (for future batched writes — currently writes immediately)
    std::vector<PendingSnapshot> pending_snaps_;
};

} // namespace runtime
} // namespace macroflow3d
