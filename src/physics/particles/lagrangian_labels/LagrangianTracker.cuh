#pragma once

/**
 * @file LagrangianTracker.cuh
 * @brief SF-36 N4: forward Lagrangian-label tracker on the cube production stage (public API).
 *
 * Authority: SF-36 specification (items 4-9, thresholds C and I), decision
 * `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md` item 2.
 *
 * Particles are injected on the plane x1 = X0 on a fixed stratified M x M lattice (fixed,
 * grid-independent positions; weight w_p = max(k g1, 0) at the seed, normalized to sum 1 over the
 * non-backflow seeds), traced FORWARD on the spline flow v_s = k_s g (g = G + grad s_h,
 * k_s = exp(s_Y)) with DP5(4) in arclength (the SF-30 step control, error norm on the 3 positions
 * only, Henon landings), and carry passively: the travel time tau, the accumulated divergence
 * I = int div v_s dt and (optionally) the covectors a, b of the labels, a x b = v_s exp(-I).
 *
 * Two deterministic passes per particle (activation reading rule 1): a PLANE pass (landings with
 * x1 independent on a strictly increasing list of unwrapped planes X_m > X0) and a TIME pass
 * (landings with tau independent on a strictly increasing list of snapshot times). Per pass:
 * status (closure_gate::StreamlineStatus values), counters, records (NaN where not reached).
 * Equations and state layouts: LagrangianTrackerDevice.cuh.
 *
 * Moments (host, index order): weighted mean and BIASED weighted variance of x1, x2, x3 per
 * snapshot (io::TimeSeriesPoint), of x2, x3, tau per plane, plus RMS/max of e_v = 1 - exp(-I);
 * weights renormalized over the particles that reached the plane/time.
 *
 * Ownership/determinism: prepare() is the only allocating call (device buffers for Np, P, T; the
 * host result vectors are sized there too); track() launches one thread per particle, writes
 * per-particle outputs only (no atomics) and reduces on the host in index order; results are
 * bitwise independent of the block size and of repeated launches. Every legacy-stream
 * host-to-device cudaMemcpy is followed by cudaDeviceSynchronize before ctx-stream work (SF-33
 * C2). Double precision; nothing clamped or regularized.
 *
 * NUMERICAL CONTRACT: the device code is compiled with --fmad=false (LagrangianTracker.cu).
 * Other fields (analytic test fields) use the header-only template in LagrangianTrackerImpl.cuh
 * from translation units that are themselves compiled with --fmad=false.
 */

#include "../../../runtime/CudaContext.cuh"
#include "../../streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "LagrangianTrackerTypes.hpp"

#include <memory>
#include <string>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

class LagrangianTracker {
  public:
    LagrangianTracker();
    ~LagrangianTracker();
    LagrangianTracker(LagrangianTracker&&) noexcept;
    LagrangianTracker& operator=(LagrangianTracker&&) noexcept;
    LagrangianTracker(const LagrangianTracker&) = delete;
    LagrangianTracker& operator=(const LagrangianTracker&) = delete;

    /// Uploads the spline coefficients, builds the lattice, weights and (if
    /// stage.labels_available and inj.X0 == 0) the D-1 labels at the seeds, and allocates all
    /// device/host buffers. Requires stage.flow_ready(). Throws std::invalid_argument for invalid
    /// options and std::runtime_error if no seed has positive flux.
    void prepare(CudaContext& ctx, const streamfunctions::inlet_slab::ProductionStage& stage,
                 const InjectionSpec& inj, const TrackOptions& opt);

    /// Runs both passes and the host reductions. `block` > 0 overrides TrackOptions::block. The
    /// returned reference is valid until the next track()/destruction. No device allocation.
    const LagrangianTrackResult& track(CudaContext& ctx, int block = 0);

    bool prepared() const;
    /// Data pointers of every owned device buffer (stable between prepare and track).
    std::vector<const void*> device_storage_pointers() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Writes the outputs of the SF-36 specification (item 8) into `dir` (created if missing):
/// seeds.npy, weights.npy, labels.npy, status_plane.npy, status_time.npy (uint8), planes_x23.npy,
/// planes_tau.npy, planes_I.npy, times_x.npy, times_I.npy, [planes_cov.npy, times_cov.npy],
/// moments_time.csv (CsvTimeSeriesWriter format), moments_planes.csv, track.json.
void write_track_outputs(const std::string& dir, const LagrangianTrackResult& result,
                         const TrackMeta& meta);

} // namespace lagrangian_labels
} // namespace macroflow3d
