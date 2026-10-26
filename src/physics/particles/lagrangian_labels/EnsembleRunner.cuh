#pragma once

/**
 * @file EnsembleRunner.cuh
 * @brief SF-38 N3: Lagrangian ensemble runner on the long box, medium (a) (contract C-RUNNER of
 *        docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/preregistration.md).
 *
 * One realization r (seed seed0 + r) is a pure function of (command line, seed):
 *   SF-18 field -> LongBoxStage (lean, SF-40) -> LagrangianTracker (plane + time passes) ->
 *   twin series (weights 1/N_p) -> Pollock bridge (RT0 comparator on the SF-19 faces of the same
 *   realization, flux-weighted and twin) -> outputs.
 *
 * Output tree under --out DIR (io::OutputLayout names for stats/, ensemble/, analysis/):
 *   stats/realization_%04d_timeseries.csv        flux-weighted moments_time (CsvTimeSeriesWriter)
 *   stats_twin/realization_%04d_timeseries.csv   twin (uniform weights) moments_time
 *   realizations/r_%04d/                         write_track_outputs files (track.json WITHOUT
 *                                                timings_s), twin/{moments_planes,moments_time}.csv,
 *                                                pollock/ (bridge), field.json, status.json, timings.json, DONE
 *   ensemble/ensemble_timeseries.csv, analysis/macrodispersion.csv, analysis_config.yaml
 *
 * Wall-clock values (timings, commit hash of the build) go ONLY to timings.json and stdout; every
 * other file is byte-reproducible. Atomicity: a realization is built in realizations/r_%04d.tmp/
 * and stats/*.tmp, renamed into place, and DONE is written last; on start realizations with DONE
 * are skipped, stale *.tmp and r_%04d directories without DONE are deleted. DONE holds the
 * canonical option signature, so a restart under different options is refused (exit 7).
 *
 * SF-41 N5 (contract C-RUNNER-B): `--medium b` runs the same realization on the lean medium (b) stage
 * (`DirichletLongBoxStage`: Dirichlet in x1, periodic y/z, K = exp(eps Y), G1 = 1, non-periodic-in-x1 splines),
 * tracks with the non-periodic tracker entry (particles leaving through x1 = 0 / x1 = L1 get the exit statuses
 * 7 / 8 and are counted per plane and per time) and bridges to the non-periodic Pollock mode (codes 8 / 9).
 * Medium (b) differences (all additive; medium (a) with ONE --hmax-div keeps its file set, stdout and bytes):
 *   - defaults: --x0 = 5 lambda; --planes auto = x0 + m lambda (m = 1, 2, 4, .. while <= L1 - 5 lambda) plus
 *     x0 + L1 - 10 lambda and L1 - 5 lambda; --times auto = n lambda / 2 for n = 1, 2, .. while <=
 *     (L1 - 10 lambda) / qbar (at least 2), qbar = the realization's measured inlet mean flux (mean of U[0]), so
 *     the times are resolved PER REALIZATION (printed as `R=<r> RESOLVED_TIMES`; every list is a prefix of the same
 *     arithmetic progression, the ensemble analysis uses the common prefix);
 *   - per realization additionally: flow_stats.json (x-face u <= 0 fraction, percentiles 0 / 0.1 / 1 / 50 % of the
 *     cell-centre |v| / qbar, interior mean face-flux vector over [5 lambda, L1 - 5 lambda] and its tilt theta_2,
 *     theta_3, qbar; `%.17g`), solver_b.json (the stage's B_SOLVER record), `B_SOLVER`, `MEM_PHASE` and `TEXIT`
 *     lines, track.json exit keys and exits_planes.csv / exits_time.csv (tracker), pollock/pollock_exits_planes.csv
 *     and pollock_exits_time.csv (bridge), exit totals in status.json;
 *   - the analysis uses vmean_norm = the mean over the realizations of qbar (flow_stats.json), written in
 *     analysis_config.yaml;
 *   - --head-mapped-host: the PCG solution lives in pinned mapped host memory (bitwise, pre-authorized memory fallback).
 * Multi-pass (`--hmax-div 8,16,32`, either medium): ONE stage build per realization, the tracker re-prepared and
 * re-run for each h_max = h / d; Pollock bridge and flow statistics once. Layout: realizations/r_%04d/hmax_<d>/
 * holds the per-pass tracker files (write_track_outputs, twin/, status.json); the realization directory keeps
 * field.json, status.json (Np, seed_backflow, hmax_div list, pollock), pollock/, flow_stats.json, solver_b.json,
 * timings.json, DONE; the per-pass series live under <out>/hmax_<d>/{stats,stats_twin,ensemble,analysis,
 * analysis_config.yaml} (each <out>/hmax_<d> is an analysis_only directory). A single value uses the layout above.
 * The in-process EnsembleRunSummary refers to the first --hmax-div value.
 *
 * Exit codes: 0 ok (every realization completed; particles with a non-ok status are RECORDED in
 * status.json and the STATUS line, never an exit condition); 1 unexpected exception; 2 usage error
 * (unknown/contradictory/invalid option); 3 seed-face assertion (RR12); 4 stage failure (SF-19 not
 * converged / splines not built; medium (b): the solve is not usable, i.e. not converged beyond the 1e-8 fallback or
 * balance above 1e-8); 7 DONE realization produced under different options; 99 test-only
 * abort (--test-abort-after). Codes 5 and 6 of the first version were removed.
 */

#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "LagrangianTrackerTypes.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

enum EnsembleExit : int {
    kExitOk = 0,
    kExitException = 1,
    kExitUsage = 2,
    kExitSeedFace = 3,
    kExitStageFailed = 4,
    kExitConfigMismatch = 7,
    kExitTestAbort = 99
};

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct EnsembleConfig {
    int nx = 0, ny = 0, nz = 0; ///< ny == nz; h = 1/ny; L1 = nx/ny box units
    double sigma2 = 1.0, ell = 0.0, eps = 0.0;
    unsigned long long seed0 = 0;
    int nr = 1;
    double x0 = 0.0;
    double window[4] = {0.1, 0.9, 0.1, 0.9}; ///< w2a,w2b,w3a,w3b
    int np_side = 100;
    std::vector<double> planes, times; ///< resolved (auto rules applied by parse_ensemble_args)
    bool planes_auto = true, times_auto = true;
    double tol = 1e-8;
    double max_arclength = 0.0; ///< 0: tracker default
    int hmax_div = 8;           ///< first (or only) --hmax-div value
    std::vector<int> hmax_list{8}; ///< SF-41 N5: all --hmax-div values (multi-pass), distinct, >= 1
    bool medium_b = false;      ///< SF-41 N5: --medium b (DirichletLongBoxStage, non-periodic x1, exit-aware)
    bool head_mapped_host = false; ///< (b) only: PCG solution in pinned mapped host memory
    int cg_max_iter_b = 2000;   ///< (b) only (the A/B application's value)
    double pcg_rtol = 1e-10;
    bool fused_splines = false, single_launch = false;
    bool twin = true, pollock_bridge = false, field_only = false;
    std::string out;
    std::string command_line;
    // test-only (hidden option): _Exit(99) at `phase` of realization r. Phases: stage, partial, tmp, rename.
    int test_abort_r = -1;
    std::string test_abort_phase;
};

/// Parses and validates the command line (argv[0] skipped). Throws UsageError whose message names
/// the offending option. `--help` throws UsageError with the usage text (caller prints, exit 0).
EnsembleConfig parse_ensemble_args(int argc, const char* const* argv);
std::string ensemble_usage();

/// RR3 auto rules (lambda = ell, L1 in box units).
std::vector<double> auto_planes(double x0, double L1, double lambda);
std::vector<double> auto_times(double L1, double lambda);
/// SF-41 medium (b) auto rules: planes x0 + m lambda (m = 1, 2, 4, .. while <= L1 - 5 lambda) plus x0 + L1 - 10 lambda
/// and L1 - 5 lambda (sorted, near-equal values merged as auto_planes); times n lambda / 2, n = 1.. while
/// n lambda / 2 <= (L1 - 10 lambda) / qbar, at least 2 times.
std::vector<double> auto_planes_b(double x0, double L1, double lambda);
std::vector<double> auto_times_b(double L1, double lambda, double qbar);

/// SF-41 flow statistics of a medium (b) realization (CompactMAC faces U (nx+1) ny nz, V nx (ny+1) nz,
/// W nx ny (nz+1), x fastest; h = 1/ny, L1 = nx h; reductions in index order, double).
struct FlowStatsB {
    double qbar = 0.0, h = 0.0, L1 = 0.0, lambda = 0.0;
    long long x_faces = 0, x_faces_nonpositive = 0; ///< all (nx+1) ny nz x-faces, boundary planes included
    double x_nonpositive_fraction = 0.0;
    /// percentiles 0, 0.1, 1, 50 % of |v_c| / qbar over the n cells, v_c = mean of the two opposite face velocities
    /// per axis; linear interpolation between order statistics at (p/100)(n-1)
    double speed_pct[4] = {0, 0, 0, 0};
    int plane_lo = 0, plane_hi = -1, layer_lo = 0, layer_hi = -1; ///< interior x-face planes / cell layers
    double mean_u = 0.0, mean_v = 0.0, mean_w = 0.0;               ///< interior mean face fluxes
    double theta_2 = 0.0, theta_3 = 0.0;                            ///< atan(<v>/<u>), atan(<w>/<u>)
};
/// Interior = x-face planes i with i h in [5 lambda, L1 - 5 lambda] and cell layers with (i + 1/2) h in the same
/// interval (closed, 1e-9 h tolerance, as EulerianFaceStats). Throws std::invalid_argument on wrong sizes,
/// non-finite or non-positive h / lambda / qbar, or an empty interior.
FlowStatsB compute_flow_stats_b(const std::vector<real>& U, const std::vector<real>& V,
                                const std::vector<real>& W, int nx, int ny, int nz, double lambda,
                                double qbar);
/// The flow_stats.json text of a FlowStatsB (every double `%.17g`).
std::string flow_stats_b_json(const FlowStatsB& s);

/// RR12: true iff some seed has |n_T x - round(n_T x)| <= 1e-6 (or non-finite) in x2 or x3; `what`
/// names the first offending seed.
bool find_seed_on_face(const InjectionSpec& inj, int n_transverse, std::string* what);

/// Canonical signature of everything that determines a realization (excludes --out, --nr and the
/// test hook); stored in DONE.
std::string ensemble_signature(const EnsembleConfig& c);

/// In-memory results of the realizations computed by THIS invocation (index r; skipped ones empty).
struct EnsembleRunSummary {
    std::vector<std::vector<io::TimeSeriesPoint<double>>> flux_series, twin_series;
    std::vector<std::vector<double>> twin_weights;
    std::vector<bool> computed;
};

/// Runs the ensemble (see the file header). Returns an EnsembleExit code; never throws for the
/// documented failure classes (an unexpected exception is rethrown).
int run_ensemble(CudaContext& ctx, const EnsembleConfig& cfg, EnsembleRunSummary* summary = nullptr);

} // namespace lagrangian_labels
} // namespace macroflow3d
