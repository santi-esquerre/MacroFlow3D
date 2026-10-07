/**
 * @file spurious_spreading_main.cu
 * @brief SF-32 N2a/N2b: `spurious_spreading` -- the measurement instrument of
 *        Lester et al. (2023) eqs. 34-36 (spurious transverse spreading of a
 *        tracker after one period on a surrogate flow with exact invariants).
 *
 * Documented-experiment instrument (like apps/closure_gate/closure_gate), NOT
 * a ctest entry. Nothing existing calls it. Subcommands:
 *
 *   spurious_spreading solve-labels --field
 * <lester2021|lester_brk|control2d|two_mode|generic3d|homogeneous>
 *       --n <N> --out <prefix> [--eps 0.25] [--max-iter 1000] [--tolerance 1e-8]
 *       [--epsilon 1e-6] [--anderson 1] [--newton 0] [--pcg-rtol 1e-10]
 *       [--mg-levels auto] [--no-timing]
 *     Frozen periodic stack (ev_ladder recipe), saves the final state as the
 *     label pair (label_routes.hpp).
 *
 *   spurious_spreading analytic-labels --pair <U|A|B|G> --n <N> --out <prefix>
 *       [--amplitude a] [--amplitude-b b] [--no-timing]
 *     Closed-form pair sampled at the cell centres (analytic_pair_g.hpp);
 *     default amplitudes: G 0.05, A 0.1, B 0.1 (b = 0.08), U 0.
 *
 *   spurious_spreading return-map --labels <prefix> --tracker <pseudo_symplectic|rk|pollock>
 *       --out <run_dir> [--seeds 8192] [--seed 20261006] [--tol-psi 1e-10]
 *       [--ds-ratio 0.5] [--tol 1e-6] [--dt-max 0.25] [--delta-ratio 1]
 *       [--max-panels 10000000] [--max-chunks 10000000] [--max-cells 10000000]
 *       [--no-timing]
 *     Seeds: inject_box (SF-31 53-bit hash) on the face x1 = 0, identical for
 *     every tracker and level given --seed. One-period return map
 *     (return_map.cuh), outputs <run_dir>/seeds.csv and <run_dir>/summary.json
 *     (schema below; dag.json output_schema). <run_dir> is created if missing.
 *     Levels: pseudo_symplectic --tol-psi (ds = ds_ratio h); rk --tol with
 *     --dt-max an ABSOLUTE time (default 0.25; decision D-2 of the SF-32
 *     orchestration record: the former cap h/2 left the DP5(4) controller
 *     inactive) and the crossing-detection chunk EQUAL to dt_max; pollock
 *     --delta-ratio m (integer >= 1 dividing N; Pollock grid n = N / m,
 *     Delta = m h; Stokes face fluxes, divergence_max_rel in summary.json;
 *     --max-cells = cells per particle before status 12).
 *     The return map itself lives in return_map_run.{hpp,cu} (library code
 *     shared with the ctest spurious_spreading_controls16).
 *
 * Exit codes: 0 ok; 2 usage (including a --delta-ratio that does not divide
 * N); 3 label pair refused by the usability guard (min_abs_c_grid <= 0.5);
 * 4 Darcy PCG did not converge in solve-labels; 1 exception.
 *
 * seeds.csv: header
 *   seed_index,x2_0,x3_0,status,delta_x2,delta_x3,delta_psi1,delta_psi2,tau,count,land_err
 * doubles with %.17g; for status != 0 the four deltas are written as `nan`
 * (tau, count and land_err are then those of the last accepted state).
 *
 * summary.json (key order fixed): field (copy of the labels metadata object),
 * tracker, level (pseudo_symplectic: {tol_psi, ds}; rk: {tol, dt_max};
 * pollock: {delta_ratio, n_cells, delta}),
 * n_seeds, n_ok, status_counts ({"code": count}, codes ascending), stats
 * (over status == 0; population variance: delta_x2, delta_x3, delta_psi1,
 * delta_psi2: {mean, var, rms, max_abs}; tau: {mean, var, min, max}),
 * flux_weighted_tau_mean (weights c1(seed) of the spline velocity),
 * D22_uniform = var(delta_x2) / (2 mean tau), D33_uniform, D22_flux, D33_flux
 * (flux-weighted variance about the flux-weighted mean over twice the
 * flux-weighted mean tau; eq. 36, protocol numbers, not coefficients),
 * divergence_max_rel (null except pollock: max |div| / max |u| of the face
 * fluxes, flux form), min_abs_c_grid, landing
 * {max_err, max_iterations} (over status == 0), config (every CLI value),
 * wall_seconds (LAST; 0 with --no-timing, so two runs with the same inputs
 * are byte-identical).
 *
 * Determinism: one thread per particle, no atomics or reductions on the
 * device; host statistics are sequential sums in seed order.
 */

#include "apps/spurious_spreading/label_routes.hpp"
#include "apps/spurious_spreading/return_map_run.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace {

using namespace spurious_spreading;

void print_usage() {
    std::fprintf(
        stderr,
        "usage:\n"
        "  spurious_spreading solve-labels --field <lester2021|lester_brk|control2d|two_mode|"
        "generic3d|homogeneous>\n"
        "      --n <N (power of 2)> --out <prefix> [--eps 0.25] [--max-iter 1000] [--tolerance "
        "1e-8]\n"
        "      [--epsilon 1e-6] [--anderson 1] [--newton 0] [--pcg-rtol 1e-10] [--mg-levels auto]\n"
        "      [--no-timing]\n"
        "  spurious_spreading analytic-labels --pair <U|A|B|G> --n <N (power of 2)> --out "
        "<prefix>\n"
        "      [--amplitude a] [--amplitude-b b] [--no-timing]\n"
        "  spurious_spreading return-map --labels <prefix> --tracker "
        "<pseudo_symplectic|rk|pollock>\n"
        "      --out <run_dir> [--seeds 8192] [--seed 20261006] [--tol-psi 1e-10] [--ds-ratio "
        "0.5]\n"
        "      [--tol 1e-6] [--dt-max 0.25 (absolute; rk chunk = dt_max)]\n"
        "      [--delta-ratio 1 (integer dividing N)] [--max-panels 10000000]\n"
        "      [--max-chunks 10000000] [--max-cells 10000000] [--no-timing]\n"
        "exit codes: 0 ok, 2 usage, 3 labels refused (min_abs_c_grid <= 0.5),\n"
        "            4 Darcy PCG not converged (solve-labels), 1 exception\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "solve-labels")
            return run_solve_labels(argc - 1, argv + 1);
        if (cmd == "analytic-labels")
            return run_analytic_labels(argc - 1, argv + 1);
        if (cmd == "return-map")
            return run_return_map(argc - 1, argv + 1);
        if (cmd == "--help" || cmd == "-h") {
            print_usage();
            return 0;
        }
        std::fprintf(stderr, "error: unknown subcommand '%s'\n", cmd.c_str());
        print_usage();
        return 2;
    } catch (const UsageError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "spurious_spreading: exception: %s\n", e.what());
        return 1;
    }
}
