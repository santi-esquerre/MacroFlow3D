#pragma once

/**
 * @file PollockComparator.cuh
 * @brief SF-37 N3: per-particle Pollock (RT0, cell-by-cell semi-analytic) lattice tracker on the SF-19
 *        finite-volume face fluxes: the independent velocity representation of V3b (contract
 *        C-POLLOCK-LIB). Shared by the ctest `lagrangian_face_flux_repack` and the inlet_slab driver, so
 *        that the driver runs exactly the tested code.
 *
 * Frame (SF-37 V3b): the Lagrangian consumer (SF-36) traces the spline flow v_s; this comparator traces
 * the SAME seed lattice (make_seed_lattice, p = m2 M + m3) with the SF-32 Pollock core on the RT0
 * reconstruction of the SF-19 face fluxes (C^0, solenoidal to the PCG residual, not structure-preserving).
 * The SF-32 core is called unchanged (PollockTracker.cuh: pollock_init_state, pollock_advance_to_x1,
 * pollock_advance_to_time); nothing here is a velocity model of its own.
 *
 * Seeds and weights. Seeds are the injection-plane lattice (X0, x2, x3), unwrapped coordinates
 * (state w = 0: x_u = x). raw weight_p = x1-velocity of the RT0 interpolant at the seed in the cell chosen
 * by pollock_init_state (the core's own evaluation: u_- + A r[0], u_+ exactly when r == D after the core
 * assigned the seed to the lower cell); weight_p = max(raw_p, 0) / sum of the positive raw weights (index
 * order; non-finite or <= 0 raw weights get weight 0 and are counted in seed_nonpos; those particles are
 * still tracked).
 *
 * Two deterministic passes, one thread per particle (no atomics, no reductions in the kernels):
 *   PLANE pass: init, then pollock_advance_to_x1 for each plane X_m (strictly increasing, > X0) in turn;
 *               at each reached plane the unwrapped (x2, x3) and the clock t are recorded.
 *   TIME pass:  fresh init, then pollock_advance_to_time for each t_n in turn; the unwrapped (x1, x2, x3)
 *               is recorded.
 * After the first non-active status of a pass (raw SF-32 codes: 12 sub-step limit, 14 non-finite,
 * 15 stagnation in the plane pass; the time pass tolerates 15 by construction) the status is kept and
 * all later records, including the failing one, are NaN. A pass with an empty list is not run
 * (status kStatusActive = 0, no records).
 *
 * Host moments (index order, replicating finalize_track_result of SF-36): per plane the weighted mean and
 * BIASED weighted variance of x2, x3, t over the particles that reached the plane (weights renormalized
 * over them; the count includes weight-0 particles); per time the same for x1, x2, x3 as
 * io::TimeSeriesPoint<double> with `active` = number of particles that reached the time.
 *
 * Memory (device, per call; allocated once per call, never inside kernels): per particle
 *   24 (seeds) + 8 (raw weight) + 2 (statuses) + 8 (reached counts) + 16 (cell counts)
 *   + 24 P (plane records x2, x3, t) + 24 T (time records x1, x2, x3) bytes,
 * plus P + T doubles for the lists. The face arrays (24 nx ny nz bytes) are owned by the caller.
 * Every legacy-stream host-to-device cudaMemcpy is followed by cudaDeviceSynchronize before ctx-stream
 * work (SF-33 C2); downloads happen after ctx.synchronize(). Double precision; compiled with --fmad=false.
 */

#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "../streamline_tracker/PollockTracker.cuh"
#include "LagrangianTrackerTypes.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

/// Weighted statistics at one plane over the particles that reached it (Pollock plane pass).
struct PollockPlaneMoments {
    double x1 = 0.0;
    int contributing = 0; ///< particles that landed on this plane
    int non_ok = 0;       ///< particles whose plane-pass status != active (any reason)
    double mean_x2 = 0.0, var_x2 = 0.0;
    double mean_x3 = 0.0, var_x3 = 0.0;
    double mean_t = 0.0, var_t = 0.0; ///< clock at the plane
};

/// Number of slots of the status histograms (raw SF-32 codes 0..15).
inline constexpr int kPollockStatusSlots = 16;

struct PollockLatticeResult {
    int Np = 0, P = 0, T = 0;
    InjectionSpec injection;
    std::vector<double> planes, times;
    int max_cells_per_call = 0;

    // seeds and weights
    std::vector<double> seeds;      ///< Np x 3
    std::vector<double> weight_raw; ///< Np (NaN if the seed could not be initialised)
    std::vector<double> weights;    ///< Np, sum 1 over the particles with positive raw weight
    long long seed_nonpos = 0;      ///< seeds with raw weight non-finite or <= 0 (weight 0)

    // per particle
    std::vector<std::uint8_t> status_plane, status_time; ///< raw SF-32 codes
    std::vector<int> reached_plane, reached_time;         ///< planes (times) reached
    std::vector<double> planes_x23;                       ///< Np x P x 2 unwrapped (NaN if not reached)
    std::vector<double> planes_t;                         ///< Np x P clock at the plane
    std::vector<double> times_x;                          ///< Np x T x 3 unwrapped
    std::vector<std::uint64_t> cells;                     ///< Np x 2 cell steps (plane pass, time pass)

    // reductions (host, index order)
    std::vector<PollockPlaneMoments> moments_planes;       ///< P entries
    std::vector<io::TimeSeriesPoint<double>> moments_time; ///< T entries
    std::array<long long, kPollockStatusSlots> status_hist_plane{}, status_hist_time{};
    unsigned long long cells_plane = 0, cells_time = 0; ///< total cell steps per pass

    // SF-41 non-periodic-in-x mode only (empty / 0 for the periodic comparator): particles that left through
    // the inlet (status 8) / outlet (status 9) before reaching plane m (time n); totals per pass. A
    // particle counts for every plane (time) from the first one it did not reach.
    std::vector<long long> exit_inlet_plane, exit_outlet_plane; ///< P entries
    std::vector<long long> exit_inlet_time, exit_outlet_time;   ///< T entries
    long long exit_inlet_plane_total = 0, exit_outlet_plane_total = 0;
    long long exit_inlet_time_total = 0, exit_outlet_time_total = 0;

    // timings [s]
    double seconds_plane_kernel = 0.0, seconds_time_kernel = 0.0, seconds_download = 0.0,
           seconds_moments = 0.0;
};

/**
 * Tracks the M x M injection lattice with the SF-32 Pollock core on the face view f (device pointers).
 * planes: strictly increasing unwrapped x1 > inj.X0; times: strictly increasing > 0; at least one of the
 * two non-empty; prm.max_cells_per_call >= 1; the injection spec as SF-36 (M >= 1, finite windows with
 * a < b). Throws std::invalid_argument on a violation, std::runtime_error if no seed has a positive raw
 * weight. Blocking: everything has completed on return.
 */
PollockLatticeResult track_pollock_lattice(CudaContext& ctx,
                                           const physics::particles::streamline_tracker::PeriodicFaceFluxView& f,
                                           const InjectionSpec& inj,
                                           const std::vector<double>& planes,
                                           const std::vector<double>& times,
                                           const physics::particles::streamline_tracker::PollockParams& prm);

/// Options of the non-periodic overload.
struct PollockNonPeriodicOptions {
    /// TEST ONLY: do not throw when no seed has a positive raw weight (all weights 0, seed_nonpos = Np, the
    /// moments are NaN); the per-particle statuses and records remain valid. Production keeps the throw.
    bool allow_no_positive_weight = false;
};

/**
 * SF-41 N3a: the same lattice tracker on face fluxes non-periodic in x (medium (b)); see PollockTracker.cuh
 * section 7. A particle that leaves through the inlet (status 8) or the outlet (status 9) keeps that status
 * in status_plane / status_time, has NaN records from the exit on, and is counted in the exit_* fields; the
 * histograms keep their 16 slots; the moments are over the particles that reached the plane / time. The
 * periodic overload above is unchanged. Throws as the periodic overload.
 */
PollockLatticeResult track_pollock_lattice(CudaContext& ctx,
                                           const physics::particles::streamline_tracker::NonPeriodicXFaceFluxView& f,
                                           const InjectionSpec& inj,
                                           const std::vector<double>& planes,
                                           const std::vector<double>& times,
                                           const physics::particles::streamline_tracker::PollockParams& prm,
                                           const PollockNonPeriodicOptions& opt = {});

} // namespace lagrangian_labels
} // namespace macroflow3d
