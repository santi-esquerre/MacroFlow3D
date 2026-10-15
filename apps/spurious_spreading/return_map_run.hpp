#pragma once

/**
 * @file return_map_run.hpp
 * @brief SF-32 N2b: the `return-map` subcommand of the `spurious_spreading`
 *        instrument as library code (options, one in-memory run, the
 *        seeds.csv / summary.json texts and the files).
 *
 * Factored out of `spurious_spreading_main.cu` (N2a) so that the fast ctest
 * `spurious_spreading_controls16` drives exactly the code the executable
 * runs, with in-memory results, without spawning the executable. The
 * executable's `return-map` is `run_return_map(argc, argv)`, i.e.
 * parse_run_options -> read_labels -> compute_return_map -> write_run_dir.
 *
 * Behaviour and schema: see the file header of spurious_spreading_main.cu
 * (CLI, exit codes, seeds.csv, summary.json key order) and return_map.cuh
 * (the per-particle cores). Nothing here prints except run_return_map.
 */

#include "apps/spurious_spreading/json_writer.hpp"
#include "apps/spurious_spreading/label_routes.hpp"

#include "src/runtime/CudaContext.cuh"

#include <cstdint>
#include <string>
#include <vector>

namespace spurious_spreading {

enum class TrackerKind { pseudo_symplectic, rk, pollock };

const char* tracker_name(TrackerKind k);

/// Every CLI value of `return-map` (defaults = the pre-registered ones).
struct RunOptions {
    std::string labels;
    TrackerKind tracker = TrackerKind::pseudo_symplectic;
    double tol_psi = 1e-10;
    double ds_ratio = 0.5;
    double tol = 1e-6;
    double dt_max = 0.25;      ///< RK step cap, ABSOLUTE time (decision D-2); chunk = dt_max
    long long delta_ratio = 1; ///< Pollock grid Delta = delta_ratio h (integer, divides N)
    long long seeds = 8192;
    unsigned long long seed = 20261006ULL;
    long long max_panels = 10000000LL;
    long long max_chunks = 10000000LL;
    long long max_cells = 10000000LL; ///< Pollock max_cells_per_call
    bool no_timing = false;
    std::string out;
};

/// argv[0] is the subcommand name. Throws UsageError.
RunOptions parse_run_options(int argc, char** argv);

/// The summary.json "config" member (every CLI value plus fixed parameters).
JVal run_config_json(const RunOptions& o);

/// In-memory result of one return map (host arrays in seed order).
struct ReturnMapRun {
    int exit_code = 0; ///< 0 ok; 3 labels refused by the usability guard (nothing else filled)
    double min_abs_c_grid = 0; ///< spline pair, cell centres
    JVal level;                ///< tracker level object of summary.json
    JVal divergence_max_rel;   ///< null unless pollock

    // per seed (n entries each)
    std::vector<double> x2_0, x3_0, c1;
    std::vector<uint8_t> status;
    std::vector<double> x1u, x2u, x3u;
    std::vector<double> delta_x2, delta_x3, delta_psi1, delta_psi2, tau, land_err;
    std::vector<unsigned long long> count;
    std::vector<int> land_iters;

    JVal summary; ///< full summary.json object (wall_seconds set by compute_return_map)
    double wall_seconds = 0.0; ///< measured (written as 0 with no_timing)
};

/**
 * One return map on an already loaded label pair: spline build, usability
 * guard (min_abs_c_grid <= 0.5 -> exit_code 3), seeds (SF-31 inject_box),
 * the tracker, host statistics and the summary object. Throws UsageError
 * for a Pollock delta_ratio that does not divide N; std exceptions otherwise.
 */
ReturnMapRun compute_return_map(const macroflow3d::CudaContext& ctx, const LoadedLabels& labels,
                                const RunOptions& o);

/// seeds.csv text (header + one row per seed, %.17g; nan deltas if status != 0).
std::string seeds_csv_text(const ReturnMapRun& r);

/// summary.json text (r.summary.dump(2) + '\n').
std::string summary_json_text(const ReturnMapRun& r);

/// mkdir -p out_dir; write <out_dir>/seeds.csv and <out_dir>/summary.json.
void write_run_dir(const std::string& out_dir, const ReturnMapRun& r);

/// `spurious_spreading return-map ...` (argv[0] = "return-map"). Exit codes
/// 0 ok, 3 labels refused; throws UsageError (exit 2) / std exceptions (exit 1).
int run_return_map(int argc, char** argv);

} // namespace spurious_spreading
