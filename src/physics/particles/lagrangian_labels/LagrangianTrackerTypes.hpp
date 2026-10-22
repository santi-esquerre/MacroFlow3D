#pragma once

/**
 * @file LagrangianTrackerTypes.hpp
 * @brief SF-36 N4: host-safe data types of the forward Lagrangian-label tracker (no CUDA include).
 *
 * See LagrangianTracker.cuh for the module description and LagrangianTrackerDevice.cuh for the
 * equations. Cube units: [0,1]^3 periodic, mean flux e1, porosity 1; coordinates are unwrapped.
 */

#include "../../../io/writers/CsvTimeSeriesWriter.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

/// Per-particle, per-pass statistics (POD, copied from the device as is).
struct PassStats {
    long long nfev = 0; ///< every field evaluation inside the pass (as SF-30 field_evaluations)
    long long accepted_steps = 0, rejected_steps = 0;
    long long landing_steps = 0, landing_rejected = 0, landings_discarded = 0;
    long long recrossings = 0; ///< accepted steps crossing back over an already-landed plane
    double min_g1_hat = 0.0;   ///< min g1/|g| over accepted points (incl. the seed)
    double final_s = 0.0;      ///< arclength of the last accepted/landed state
    int status = 0;            ///< closure_gate::StreamlineStatus value (0 ok ... 6)
    int n_reached = 0;         ///< planes (times) landed
    int backflow_encounter = 0;
    int pad_ = 0;
};

/// Integrator parameters of a pass (device-copyable).
struct PassOptions {
    double tol = 1e-8;
    double h_max = 0.0;
    double min_step = 1e-13;
    double max_arclength = 0.0;
};

/// Injection plane x1 = X0 and the stratified M x M lattice on the window
/// [w2a, w2b) x [w3a, w3b): x2 = w2a + (w2b - w2a) (m2 + 1/2)/M, x3 likewise; particle index
/// p = m2 M + m3.
struct InjectionSpec {
    double X0 = 0.0;
    int M = 16;
    double w2a = 0.0, w2b = 1.0, w3a = 0.0, w3b = 1.0;
};

struct TrackOptions {
    double tol = 1e-8;
    double h_max = 0.0;    ///< REQUIRED (0 < h_max < 1), as the SF-30 integrator
    double min_step = 1e-13;
    /// Cap on the total arclength from the seed per pass; <= 0: 4 max(X_last - X0, t_last) + 10.
    double max_arclength = 0.0;
    std::vector<double> planes; ///< strictly increasing unwrapped x1 values > X0
    std::vector<double> times;  ///< strictly increasing t_n > 0
    bool covectors = false;     ///< transport the covectors a, b (D = 11) in both passes
    int block = 64;             ///< threads per block (results are independent of it)
    /// SF-40 L6 (opt-in, bitwise-identical outputs): evaluate the potential and ln k splines over one
    /// interleaved double2 coefficient array (FusedSplineLagrangianField).
    bool fused_splines = false;
    /// SF-40 L6 (opt-in, bitwise-identical outputs): one kernel launch runs the plane pass and then the
    /// time pass of each particle; timings: plane/time kernel NaN, seconds_single_kernel set.
    bool single_launch = false;
};

/// Weighted statistics at one plane over the particles that reached it.
struct PlaneMoments {
    double x1 = 0.0;
    int contributing = 0; ///< particles whose plane pass landed on this plane
    int non_ok = 0;       ///< particles whose plane pass status != ok (any reason, incl. backflow seed)
    double mean_x2 = 0.0, var_x2 = 0.0;
    double mean_x3 = 0.0, var_x3 = 0.0;
    double mean_tau = 0.0, var_tau = 0.0;
    double rms_ev = 0.0; ///< sqrt(sum w e_v^2 / sum w), e_v = 1 - exp(-I)
    double max_ev = 0.0; ///< max |e_v|
};

/// Totals of one pass over all particles.
struct PassTotals {
    std::array<long long, 7> status_hist{};
    long long nfev = 0, accepted_steps = 0, rejected_steps = 0, landing_steps = 0,
              landing_rejected = 0, landings_discarded = 0, recrossings = 0;
    long long backflow_encounters = 0; ///< particles with a backflow encounter
    double min_g1_hat = 0.0;
};

struct LagrangianTrackResult {
    int Np = 0, P = 0, T = 0;
    bool covectors = false;
    // options actually used (after defaults)
    double tol = 0.0, h_max = 0.0, min_step = 0.0, max_arclength = 0.0;
    InjectionSpec injection;
    std::vector<double> planes, times;

    // injection (fixed after prepare)
    std::vector<double> seeds;   ///< Np x 3
    std::vector<double> weights; ///< Np, sum 1 over the non-backflow seeds
    std::vector<double> labels;  ///< Np x 2 (psi1, psi2 D-1 at the seed), NaN if unavailable
    long long seed_backflow = 0; ///< seeds with g1 <= 0 (weight 0)

    // per-particle
    std::vector<PassStats> stats_plane, stats_time;
    std::vector<std::uint8_t> status_plane, status_time;
    std::vector<double> planes_x23; ///< Np x P x 2 (NaN where not reached)
    std::vector<double> planes_tau; ///< Np x P
    std::vector<double> planes_I;   ///< Np x P
    std::vector<double> planes_cov; ///< Np x P x 6 (a, b) iff covectors
    std::vector<double> times_x;    ///< Np x T x 3
    std::vector<double> times_I;    ///< Np x T
    std::vector<double> times_cov;  ///< Np x T x 6 iff covectors

    // reductions (host, index order)
    PassTotals totals_plane, totals_time;
    std::vector<io::TimeSeriesPoint<double>> moments_time; ///< T entries
    std::vector<PlaneMoments> moments_planes;              ///< P entries

    // timings [s]
    double seconds_plane_kernel = 0.0, seconds_time_kernel = 0.0, seconds_download = 0.0,
           seconds_moments = 0.0;
    bool single_launch = false;         ///< SF-40 L6: the passes ran in one launch
    double seconds_single_kernel = 0.0; ///< SF-40 L6: launch time iff single_launch (else 0)
};

/// Run metadata for track.json (everything the tracker itself does not know).
struct TrackMeta {
    std::string field;  ///< e.g. "gaussian", "generic3d", "uniform"
    double eps = 0.0;
    int N = 0;
    double h = 0.0;     ///< grid spacing of the stage
    std::string commit; ///< git commit of the build, if available
};

// ---- host helpers (LagrangianTracker.cu) -------------------------------------------------------

/// Validates injection/options; throws std::invalid_argument naming the offending field.
void validate_track_setup(const InjectionSpec& inj, const TrackOptions& opt);
/// Cap used when TrackOptions::max_arclength <= 0.
double default_max_arclength(const InjectionSpec& inj, const TrackOptions& opt);
/// Np x 3 seeds, particle index p = m2 M + m3.
std::vector<double> make_seed_lattice(const InjectionSpec& inj);
/// Normalizes weights = max(flux, 0)/sum over flux > 0 (finite), index order; counts non-positive
/// (or non-finite) fluxes as seed_backflow. Throws std::runtime_error if no seed has flux > 0.
void normalize_flux_weights(const std::vector<double>& flux, std::vector<double>& weights,
                            long long& seed_backflow);
/// Fills statuses, totals and moments of `r` from the per-particle stats and records.
void finalize_track_result(LagrangianTrackResult& r);

} // namespace lagrangian_labels
} // namespace macroflow3d
