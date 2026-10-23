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
 *                                                pollock/ (bridge), field.json, timings.json, DONE
 *   ensemble/ensemble_timeseries.csv, analysis/macrodispersion.csv, analysis_config.yaml
 *
 * Wall-clock values (timings, commit hash of the build) go ONLY to timings.json and stdout; every
 * other file is byte-reproducible. Atomicity: a realization is built in realizations/r_%04d.tmp/
 * and stats/*.tmp, renamed into place, and DONE is written last; on start realizations with DONE
 * are skipped, stale *.tmp and r_%04d directories without DONE are deleted. DONE holds the
 * canonical option signature, so a restart under different options is refused (exit 7).
 *
 * Exit codes: 0 ok; 1 unexpected exception; 2 usage error (unknown/contradictory/invalid option);
 * 3 seed-face assertion (RR12); 4 stage failure (SF-19 not converged / splines not built); 5 track
 * failure (some particle with status >= 2 in a pass); 6 comparator failure (a positive-weight
 * Pollock particle not active); 7 DONE realization produced under different options;
 * 99 test-only abort (--test-abort-after).
 */

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
    kExitTrackFailed = 5,
    kExitComparatorFailed = 6,
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
    int hmax_div = 8;
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
