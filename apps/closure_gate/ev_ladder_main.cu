/**
 * @file ev_ladder_main.cu
 * @brief SF-30 N3: `streamfunction_ev_ladder` -- measures the SF-11 velocity
 *        reconstruction error `e_v` of the FROZEN periodic streamfunction
 *        stack (SF-02..SF-26) on one field and one grid.
 *
 * Documented-experiment tool (not a ctest entry). It calls the frozen public
 * API only and changes nothing in it:
 *
 *   - `physics::generate_periodic_gaussian_field` (SF-18) for `--field gaussian`,
 *     or `closure_gate::fill_analytic_log_conductivity` (N1) for the analytic
 *     controls; `homogeneous` is `Y = 0`;
 *   - `physics::solve_affine_periodic_flow` (SF-19) with `qbar = (1, 0, 0)` for
 *     the Darcy reference velocity, on `K = exp(Y)`;
 *   - `streamfunctions::solve_streamfunctions` (SF-13..SF-24), problem view
 *     built exactly like `run_streamfunction_heterogeneity_continuation`
 *     (`ContinuationController.cu`): `Y` in the log-conductivity
 *     representation, triply periodic `BCSpec`, the SF-19 velocity as
 *     `darcy_velocity`, `AffineGauge::benchmark(1)` (with `qbar = e1` the
 *     achieved mean flux is 1, so `vbar = 1` and `psi1 = x2 + u1`,
 *     `psi2 = x3 + u2`, `grad psi1 x grad psi2 = e1` for the affine parts);
 *   - every residual/`e_v`/invariance/divergence/`|c|` number is read from the
 *     returned `StreamfunctionSolveReport` (`report.residual` is the head
 *     evaluation at the final accepted state, `report.diagnostics` the SF-11
 *     re-evaluation at the final state, `residual_histogram_percentile` on
 *     `report.residual` gives the `|c|` percentiles). The only separate call is
 *     one extra `enqueue_streamfunction_physical_diagnostics` on the final
 *     state with degeneracy thresholds configured, because the solver's own
 *     diagnostics config keeps the library default (no thresholds; thresholds
 *     would activate the adaptive trial guards and change the solve). That
 *     call is post-solve only and its `e_v` is printed next to the report's
 *     as a consistency check.
 *
 * Whatever state the solver leaves (converged, stagnated, budget exhausted,
 * omega floor, linear failure) is "the stack's answer": no retry, no
 * fallback, `r_F` is always printed next to `e_v`, and `tolerance_met` is true
 * only if the final `r_F <= tolerance`.
 *
 * `--lambda-steps k > 1` is a plain warm-start ladder K_j = exp((j/k) Y),
 * j = 1..k, with NO acceptance gating between stages (every stage runs to its
 * own stop, the next stage starts from whatever state the previous one left).
 * This is deliberately NOT `run_streamfunction_heterogeneity_continuation`:
 * that controller accepts a stage only at `r_F <= tolerance` and cannot pass
 * the documented eta = 1 residual floor on Gaussian fields (SF-26 note), so it
 * would stop at lambda = 0 instead of measuring.
 *
 * Exit codes: 0 run completed and record produced (any solver status);
 * 2 usage error; 3 a Darcy corrector PCG did not converge (record still
 * written, the streamfunction solve of that stage is not run); 1 exception.
 */

#include "apps/closure_gate/closure_fields.hpp"
#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/Diagnostics.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace macroflow3d;
using json = nlohmann::ordered_json;
namespace sf = macroflow3d::streamfunctions;

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Options {
    std::string field;
    int n = 0;
    double eps = 0.25;
    bool have_sigma2 = false, have_ell = false, have_seed = false;
    double sigma2 = 0.0;
    double ell = 0.0;
    unsigned long long seed = 0ULL;
    double epsilon = 1e-6;
    double tolerance = 1e-8;
    int max_iter = 500;
    bool anderson = true;
    bool newton = false;
    int lambda_steps = 1;
    double pcg_rtol = 1e-10;
    bool mg_levels_auto = true;
    int mg_levels = 0;
    std::string out;
};

void print_usage() {
    std::fprintf(
        stderr,
        "usage: streamfunction_ev_ladder --field "
        "<homogeneous|lester2021|lester_brk|control2d|two_mode|generic3d|gaussian>\n"
        "                                --n <N>        cubic grid N^3 on the unit cube, h = 1/N (N even)\n"
        "                                [--eps <e>]    analytic fields: Y = eps * f (default 0.25)\n"
        "                                [--sigma2 <s> --ell <l> --seed <u64>]   gaussian (all three required)\n"
        "                                [--epsilon <1e-6>] [--tolerance <1e-8>] [--max-iter <500>]\n"
        "                                [--anderson <1|0>] [--newton <0|1>] [--lambda-steps <1>]\n"
        "                                [--pcg-rtol <1e-10>] [--mg-levels <auto|int>] [--out <file.json>]\n");
}

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double parse_double(const std::string& name, const std::string& v) {
    std::size_t pos = 0;
    double d = 0.0;
    try {
        d = std::stod(v, &pos);
    } catch (const std::exception&) {
        throw UsageError("invalid value for " + name + ": '" + v + "'");
    }
    if (pos != v.size() || !std::isfinite(d)) {
        throw UsageError("invalid value for " + name + ": '" + v + "'");
    }
    return d;
}

long long parse_int(const std::string& name, const std::string& v) {
    std::size_t pos = 0;
    long long i = 0;
    try {
        i = std::stoll(v, &pos);
    } catch (const std::exception&) {
        throw UsageError("invalid integer for " + name + ": '" + v + "'");
    }
    if (pos != v.size()) {
        throw UsageError("invalid integer for " + name + ": '" + v + "'");
    }
    return i;
}

bool parse_bool01(const std::string& name, const std::string& v) {
    if (v == "1") return true;
    if (v == "0") return false;
    throw UsageError(name + " expects 0 or 1, got '" + v + "'");
}

Options parse_options(int argc, char** argv) {
    Options o;
    bool have_field = false, have_n = false;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (a + 1 >= argc) {
            throw UsageError("missing value for " + key);
        }
        const std::string val = argv[++a];
        if (key == "--field") {
            o.field = val;
            have_field = true;
        } else if (key == "--n") {
            const long long n = parse_int(key, val);
            if (n < 4 || n > 4096 || (n % 2) != 0) {
                throw UsageError("--n must be even and in [4, 4096]");
            }
            o.n = static_cast<int>(n);
            have_n = true;
        } else if (key == "--eps") {
            o.eps = parse_double(key, val);
        } else if (key == "--sigma2") {
            o.sigma2 = parse_double(key, val);
            o.have_sigma2 = true;
        } else if (key == "--ell") {
            o.ell = parse_double(key, val);
            o.have_ell = true;
        } else if (key == "--seed") {
            std::size_t pos = 0;
            try {
                o.seed = std::stoull(val, &pos);
            } catch (const std::exception&) {
                throw UsageError("invalid --seed '" + val + "'");
            }
            if (pos != val.size()) throw UsageError("invalid --seed '" + val + "'");
            o.have_seed = true;
        } else if (key == "--epsilon") {
            o.epsilon = parse_double(key, val);
        } else if (key == "--tolerance") {
            o.tolerance = parse_double(key, val);
        } else if (key == "--max-iter") {
            const long long m = parse_int(key, val);
            if (m < 0 || m > 1000000) throw UsageError("--max-iter must be in [0, 1e6]");
            o.max_iter = static_cast<int>(m);
        } else if (key == "--anderson") {
            o.anderson = parse_bool01(key, val);
        } else if (key == "--newton") {
            o.newton = parse_bool01(key, val);
        } else if (key == "--lambda-steps") {
            const long long k = parse_int(key, val);
            if (k < 1 || k > 1000) throw UsageError("--lambda-steps must be in [1, 1000]");
            o.lambda_steps = static_cast<int>(k);
        } else if (key == "--pcg-rtol") {
            o.pcg_rtol = parse_double(key, val);
        } else if (key == "--mg-levels") {
            if (val == "auto") {
                o.mg_levels_auto = true;
            } else {
                const long long l = parse_int(key, val);
                if (l < 1 || l > 32) throw UsageError("--mg-levels must be auto or in [1, 32]");
                o.mg_levels_auto = false;
                o.mg_levels = static_cast<int>(l);
            }
        } else if (key == "--out") {
            o.out = val;
        } else {
            throw UsageError("unknown option " + key);
        }
    }
    if (!have_field) throw UsageError("--field is required");
    if (!have_n) throw UsageError("--n is required");
    const bool known = o.field == "homogeneous" || o.field == "gaussian" || o.field == "lester2021" ||
                       o.field == "lester_brk" || o.field == "control2d" || o.field == "two_mode" ||
                       o.field == "generic3d";
    if (!known) throw UsageError("unknown --field '" + o.field + "'");
    if (o.field == "gaussian" && !(o.have_sigma2 && o.have_ell && o.have_seed)) {
        throw UsageError("--field gaussian requires --sigma2, --ell and --seed");
    }
    if (o.field != "gaussian" && (o.have_sigma2 || o.have_ell || o.have_seed)) {
        throw UsageError("--sigma2/--ell/--seed apply only to --field gaussian");
    }
    return o;
}

// MG depth rule for `--mg-levels auto`: the existing callers use the library
// default (4 levels) at 32^3, i.e. a coarsest level of 4^3; generalized here
// as "halve while the extent stays even and the next extent is >= 4 and
// even", which gives 3/4/5/6 levels at 16/32/64/128 (coarsest 4^3).
int auto_mg_levels(int n) {
    int levels = 1;
    int m = n;
    while (m % 2 == 0 && (m / 2) >= 4 && ((m / 2) % 2) == 0) {
        m /= 2;
        ++levels;
    }
    return levels;
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

const char* status_label(sf::StreamfunctionSolveStatus s) {
    switch (s) {
    case sf::StreamfunctionSolveStatus::not_run: return "not_run";
    case sf::StreamfunctionSolveStatus::converged: return "converged";
    case sf::StreamfunctionSolveStatus::not_converged: return "not_converged";
    case sf::StreamfunctionSolveStatus::invalid_problem: return "invalid_problem";
    }
    return "unknown";
}

const char* exit_label(sf::PicardExitReason r) {
    switch (r) {
    case sf::PicardExitReason::none: return "none";
    case sf::PicardExitReason::converged: return "converged";
    case sf::PicardExitReason::budget_exhausted: return "budget_exhausted";
    case sf::PicardExitReason::linear_block_failure: return "linear_block_failure";
    case sf::PicardExitReason::stagnated: return "stagnated";
    case sf::PicardExitReason::omega_floor_rejected: return "omega_floor_rejected";
    case sf::PicardExitReason::newton_exhausted: return "newton_exhausted";
    case sf::PicardExitReason::newton_budget_exhausted: return "newton_budget_exhausted";
    }
    return "unknown";
}

const char* pcg_label(solvers::ProjectedPCGStatus s) {
    switch (s) {
    case solvers::ProjectedPCGStatus::converged: return "converged";
    case solvers::ProjectedPCGStatus::max_iterations: return "max_iterations";
    case solvers::ProjectedPCGStatus::invalid_configuration: return "invalid_configuration";
    case solvers::ProjectedPCGStatus::size_mismatch: return "size_mismatch";
    case solvers::ProjectedPCGStatus::aliasing: return "aliasing";
    case solvers::ProjectedPCGStatus::breakdown_pAp: return "breakdown_pAp";
    case solvers::ProjectedPCGStatus::breakdown_rz: return "breakdown_rz";
    case solvers::ProjectedPCGStatus::nonfinite_value: return "nonfinite_value";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// JSON helpers (nlohmann serializes doubles with the shortest round-trip
// representation, i.e. lossless; NaN/inf become null and are flagged by the
// corresponding *_finite booleans where they matter).
// ---------------------------------------------------------------------------

json j_pcg(const solvers::ProjectedPCGResult& r) {
    json j;
    j["status"] = pcg_label(r.status);
    j["converged"] = r.converged;
    j["iterations"] = r.iterations;
    j["raw_rhs_mean"] = static_cast<double>(r.raw_rhs_mean);
    j["raw_rhs_l2_norm"] = static_cast<double>(r.raw_rhs_l2_norm);
    j["raw_rhs_compatibility_defect"] = static_cast<double>(r.raw_rhs_compatibility_defect);
    j["initial_projected_residual"] = static_cast<double>(r.initial_projected_residual);
    j["final_projected_residual"] = static_cast<double>(r.final_projected_residual);
    j["relative_projected_residual"] = static_cast<double>(r.relative_projected_residual);
    j["final_field_mean"] = static_cast<double>(r.final_field_mean);
    return j;
}

json j_mg(const multigrid::MGConfig& m) {
    json j;
    j["num_levels"] = m.num_levels;
    j["pre_smooth"] = m.pre_smooth;
    j["post_smooth"] = m.post_smooth;
    j["coarse_solve_iters"] = m.coarse_solve_iters;
    j["check_convergence_every"] = m.check_convergence_every;
    j["omega"] = static_cast<double>(m.omega);
    return j;
}

json j_linear(const solvers::ProjectedPCGConfig& l) {
    json j;
    j["rtol"] = static_cast<double>(l.rtol);
    j["max_iter"] = l.max_iter;
    j["check_every"] = l.check_every;
    return j;
}

json j_flow_config(const physics::AffinePeriodicFlowConfig& c) {
    json j;
    j["qbar"] = {static_cast<double>(c.qbar[0]), static_cast<double>(c.qbar[1]),
                 static_cast<double>(c.qbar[2])};
    j["linear"] = j_linear(c.linear);
    j["mg"] = j_mg(c.mg);
    return j;
}

json j_flow_report(const physics::AffinePeriodicFlowReport& f) {
    json j;
    json keff = json::array();
    for (int i = 0; i < 3; ++i) {
        keff.push_back({static_cast<double>(f.K_eff[i][0]), static_cast<double>(f.K_eff[i][1]),
                        static_cast<double>(f.K_eff[i][2])});
    }
    j["K_eff"] = keff;
    j["symmetry_defect_rel"] = static_cast<double>(f.symmetry_defect_rel);
    j["eigenvalues_symmetric_part"] = {static_cast<double>(f.eigenvalues_symmetric_part[0]),
                                       static_cast<double>(f.eigenvalues_symmetric_part[1]),
                                       static_cast<double>(f.eigenvalues_symmetric_part[2])};
    j["G"] = {static_cast<double>(f.G[0]), static_cast<double>(f.G[1]), static_cast<double>(f.G[2])};
    j["achieved_mean_flux"] = {static_cast<double>(f.achieved_mean_flux[0]),
                               static_cast<double>(f.achieved_mean_flux[1]),
                               static_cast<double>(f.achieved_mean_flux[2])};
    j["div_max_abs"] = static_cast<double>(f.div_max_abs);
    j["div_rms"] = static_cast<double>(f.div_rms);
    j["corrector_results"] = {j_pcg(f.corrector_results[0]), j_pcg(f.corrector_results[1]),
                              j_pcg(f.corrector_results[2])};
    j["memory_total_bytes"] = f.memory.total_bytes;
    return j;
}

json j_solver_config(const sf::StreamfunctionSolverConfig& c) {
    json j;
    j["picard"] = {{"max_iter", c.picard.max_iter},
                   {"tolerance", static_cast<double>(c.picard.tolerance)},
                   {"omega", static_cast<double>(c.picard.omega)}};
    const auto& a = c.adaptive;
    j["adaptive"] = {{"enabled", a.enabled},
                     {"omega_min", static_cast<double>(a.omega_min)},
                     {"backtrack_factor", static_cast<double>(a.backtrack_factor)},
                     {"growth_factor", static_cast<double>(a.growth_factor)},
                     {"omega_max", static_cast<double>(a.omega_max)},
                     {"easy_streak", a.easy_streak},
                     {"armijo_c", static_cast<double>(a.armijo_c)},
                     {"stagnation_window", a.stagnation_window},
                     {"stagnation_min_reduction", static_cast<double>(a.stagnation_min_reduction)},
                     {"max_unexplained_fraction", static_cast<double>(a.max_unexplained_fraction)},
                     {"unexplained_growth_factor", static_cast<double>(a.unexplained_growth_factor)},
                     {"unexplained_growth_offset", static_cast<double>(a.unexplained_growth_offset)},
                     {"percentile_collapse_factor", static_cast<double>(a.percentile_collapse_factor)},
                     {"floor_guard",
                      {{"enabled", a.floor_guard.enabled},
                       {"window", a.floor_guard.window},
                       {"drop_factor", static_cast<double>(a.floor_guard.drop_factor)},
                       {"max_resets", a.floor_guard.max_resets}}}};
    const auto& an = c.anderson;
    j["anderson"] = {{"enabled", an.enabled},
                     {"depth", an.depth},
                     {"start_iteration", an.start_iteration},
                     {"condition_limit", static_cast<double>(an.condition_limit)},
                     {"restart_on_stagnation", an.restart_on_stagnation},
                     {"max_restarts", an.max_restarts}};
    const auto& nw = c.newton;
    j["newton"] = {{"enabled", nw.enabled},
                   {"activation_r_F", static_cast<double>(nw.activation_r_F)},
                   {"stagnation_activation_r_F", static_cast<double>(nw.stagnation_activation_r_F)},
                   {"forcing_coefficient", static_cast<double>(nw.forcing_coefficient)},
                   {"forcing_min", static_cast<double>(nw.forcing_min)},
                   {"forcing_max", static_cast<double>(nw.forcing_max)},
                   {"armijo_c", static_cast<double>(nw.armijo_c)},
                   {"alpha_min", static_cast<double>(nw.alpha_min)},
                   {"backtrack_factor", static_cast<double>(nw.backtrack_factor)},
                   {"max_newton_iterations", nw.max_newton_iterations},
                   {"rescue_picard_steps", nw.rescue_picard_steps},
                   {"gmres",
                    {{"restart", nw.gmres.restart},
                     {"max_iterations", nw.gmres.max_iterations},
                     {"rel_tol", static_cast<double>(nw.gmres.rel_tol)}}},
                   {"delta",
                    {{"delta_min", static_cast<double>(nw.delta.delta_min)},
                     {"delta_max", static_cast<double>(nw.delta.delta_max)}}},
                   {"rescue_resets_omega", nw.rescue_resets_omega}};
    j["eta"] = static_cast<double>(c.eta);
    j["epsilon"] = static_cast<double>(c.epsilon);
    j["linear"] = j_linear(c.linear);
    j["mg"] = j_mg(c.mg);
    j["histogram"] = {{"c_min_rel", static_cast<double>(c.histogram.c_min_rel)},
                      {"c_max_rel", static_cast<double>(c.histogram.c_max_rel)}};
    json thr = json::array();
    for (int t = 0; t < c.diagnostics.num_degeneracy_thresholds; ++t) {
        thr.push_back(static_cast<double>(c.diagnostics.degeneracy_thresholds[t]));
    }
    j["diagnostics"] = {{"angle_exclusion_rel", static_cast<double>(c.diagnostics.angle_exclusion_rel)},
                        {"low_speed_rel", static_cast<double>(c.diagnostics.low_speed_rel)},
                        {"num_degeneracy_thresholds", c.diagnostics.num_degeneracy_thresholds},
                        {"degeneracy_thresholds", thr}};
    json sthr = json::array();
    for (int t = 0; t < c.num_degeneracy_thresholds; ++t) {
        sthr.push_back(static_cast<double>(c.degeneracy_thresholds[t]));
    }
    j["source_num_degeneracy_thresholds"] = c.num_degeneracy_thresholds;
    j["source_degeneracy_thresholds"] = sthr;
    j["initial_state"] =
        c.initial_state == sf::PicardInitialState::zero_source ? "zero_source" : "warm_start";
    j["coefficient_state"] = c.coefficient_state == sf::CoefficientState::rebuild ? "rebuild" : "reuse";
    return j;
}

json j_diagnostics(const sf::PhysicalDiagnosticsReport& d) {
    json j;
    j["e_v"] = static_cast<double>(d.e_v);
    j["rms_u"] = static_cast<double>(d.rms_u);
    j["rms_v"] = static_cast<double>(d.rms_v);
    j["rms_w"] = static_cast<double>(d.rms_w);
    j["linf_u"] = static_cast<double>(d.linf_u);
    j["linf_v"] = static_cast<double>(d.linf_v);
    j["linf_w"] = static_cast<double>(d.linf_w);
    j["rms_u_rel"] = static_cast<double>(d.rms_u_rel);
    j["rms_v_rel"] = static_cast<double>(d.rms_v_rel);
    j["rms_w_rel"] = static_cast<double>(d.rms_w_rel);
    j["linf_u_rel"] = static_cast<double>(d.linf_u_rel);
    j["linf_v_rel"] = static_cast<double>(d.linf_v_rel);
    j["linf_w_rel"] = static_cast<double>(d.linf_w_rel);
    j["corr_u"] = static_cast<double>(d.corr_u);
    j["corr_v"] = static_cast<double>(d.corr_v);
    j["corr_w"] = static_cast<double>(d.corr_w);
    j["rms_magnitude"] = static_cast<double>(d.rms_magnitude);
    j["linf_magnitude"] = static_cast<double>(d.linf_magnitude);
    j["rms_magnitude_rel"] = static_cast<double>(d.rms_magnitude_rel);
    j["linf_magnitude_rel"] = static_cast<double>(d.linf_magnitude_rel);
    j["rms_theta"] = static_cast<double>(d.rms_theta);
    j["max_theta"] = static_cast<double>(d.max_theta);
    j["angle_included_count"] = d.angle_included_count;
    j["angle_excluded_count"] = d.angle_excluded_count;
    j["invariance_raw_rms_psi1"] = static_cast<double>(d.invariance_raw_rms_psi1);
    j["invariance_raw_rms_psi2"] = static_cast<double>(d.invariance_raw_rms_psi2);
    j["invariance_grad_rms_psi1"] = static_cast<double>(d.invariance_grad_rms_psi1);
    j["invariance_grad_rms_psi2"] = static_cast<double>(d.invariance_grad_rms_psi2);
    j["invariance_e_psi1"] = static_cast<double>(d.invariance_e_psi1);
    j["invariance_e_psi2"] = static_cast<double>(d.invariance_e_psi2);
    j["rms_div"] = static_cast<double>(d.rms_div);
    j["linf_div"] = static_cast<double>(d.linf_div);
    j["e_div"] = static_cast<double>(d.e_div);
    j["c_min"] = static_cast<double>(d.c_min);
    j["c_max"] = static_cast<double>(d.c_max);
    j["c_mean"] = static_cast<double>(d.c_mean);
    json thr = json::array(), tot = json::array(), low = json::array(), unx = json::array();
    for (int t = 0; t < d.num_degeneracy_thresholds; ++t) {
        thr.push_back(static_cast<double>(d.degeneracy_thresholds[t]));
        tot.push_back(d.degeneracy_total[t]);
        low.push_back(d.degeneracy_low_speed[t]);
        unx.push_back(d.degeneracy_unexplained[t]);
    }
    j["degeneracy_thresholds_rel"] = thr;
    j["degeneracy_total"] = tot;
    j["degeneracy_low_speed"] = low;
    j["degeneracy_unexplained"] = unx;
    j["v_d_rms"] = static_cast<double>(d.v_d_rms);
    j["L_ref"] = static_cast<double>(d.L_ref);
    j["angle_exclusion_rel"] = static_cast<double>(d.angle_exclusion_rel);
    j["low_speed_rel"] = static_cast<double>(d.low_speed_rel);
    j["n"] = d.n;
    return j;
}

json j_residual(const sf::StreamfunctionResidualReport& r) {
    json j;
    j["r_F"] = static_cast<double>(r.r_F);
    j["r1"] = static_cast<double>(r.r1);
    j["r2"] = static_cast<double>(r.r2);
    j["rms_f1"] = static_cast<double>(r.rms_f1);
    j["rms_f2"] = static_cast<double>(r.rms_f2);
    j["linf_f1"] = static_cast<double>(r.linf_f1);
    j["linf_f2"] = static_cast<double>(r.linf_f2);
    j["q_rms"] = static_cast<double>(r.q_rms);
    j["nonfinite_s1"] = r.nonfinite_s1;
    j["nonfinite_s2"] = r.nonfinite_s2;
    j["histogram_underflow"] = r.histogram_underflow;
    j["histogram_overflow"] = r.histogram_overflow;
    j["histogram_c_min"] = static_cast<double>(r.histogram_c_min);
    j["histogram_c_max"] = static_cast<double>(r.histogram_c_max);
    j["eta"] = static_cast<double>(r.eta);
    j["epsilon"] = static_cast<double>(r.epsilon);
    j["v_rms"] = static_cast<double>(r.v_rms);
    j["n"] = r.n;
    return j;
}

// ---------------------------------------------------------------------------
// Field statistics (host, population variance sum/n)
// ---------------------------------------------------------------------------

struct FieldStats {
    double mean = 0.0, variance = 0.0, min = 0.0, max = 0.0;
};

FieldStats field_stats(const std::vector<double>& y) {
    FieldStats s;
    if (y.empty()) return s;
    double sum = 0.0;
    s.min = y[0];
    s.max = y[0];
    for (double v : y) {
        sum += v;
        s.min = std::min(s.min, v);
        s.max = std::max(s.max, v);
    }
    s.mean = sum / static_cast<double>(y.size());
    double var = 0.0;
    for (double v : y) {
        const double d = v - s.mean;
        var += d * d;
    }
    s.variance = var / static_cast<double>(y.size());
    return s;
}

// ---------------------------------------------------------------------------
// One stage
// ---------------------------------------------------------------------------

constexpr int kNumPostThresholds = 3;
constexpr real kPostThresholds[kNumPostThresholds] = {real{1e-3}, real{1e-2}, real{1e-1}};
constexpr double kPercentiles[4] = {0.001, 0.01, 0.05, 0.5};

struct StageResult {
    double lambda = 0.0;
    physics::AffinePeriodicFlowReport flow{};
    bool darcy_ok = false;
    bool solved = false;
    sf::StreamfunctionSolverConfig config{};
    sf::StreamfunctionSolveReport report{};
    sf::PhysicalDiagnosticsReport post_diag{};
    double c_percentiles[4]{};
    double wall_darcy = 0.0;
    double wall_solve = 0.0;
};

double r_f_initial(const sf::StreamfunctionSolveReport& r) {
    return r.picard_history.empty() ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(r.picard_history.front().r_F);
}

bool tolerance_met(const StageResult& s) {
    return s.solved && std::isfinite(static_cast<double>(s.report.residual.r_F)) &&
           s.report.residual.r_F <= s.config.picard.tolerance;
}

json j_stage_summary(const StageResult& s) {
    json j;
    j["lambda"] = s.lambda;
    j["darcy_pcg_converged"] = s.darcy_ok;
    j["solved"] = s.solved;
    if (s.solved) {
        j["status"] = status_label(s.report.status);
        j["exit_reason"] = exit_label(s.report.exit_reason);
        j["picard_iterations"] = s.report.picard_iterations;
        j["initial_state"] = s.config.initial_state == sf::PicardInitialState::zero_source
                                 ? "zero_source"
                                 : "warm_start";
        j["r_F_initial_state"] = r_f_initial(s.report);
        j["r_F_final"] = static_cast<double>(s.report.residual.r_F);
        j["tolerance_met"] = tolerance_met(s);
        j["e_v"] = static_cast<double>(s.report.diagnostics.e_v);
    }
    j["wall_seconds_darcy"] = s.wall_darcy;
    j["wall_seconds_solve"] = s.wall_solve;
    return j;
}

json j_stage_full(const StageResult& s) {
    json j = j_stage_summary(s);
    j["darcy"] = j_flow_report(s.flow);
    if (!s.solved) return j;
    const auto& r = s.report;
    j["solver_config"] = j_solver_config(s.config);
    j["final_omega"] = static_cast<double>(r.final_omega);
    j["picard_history_size"] = r.picard_history.size();
    json last = json::array();
    const std::size_t h = r.picard_history.size();
    for (std::size_t i = (h > 10 ? h - 10 : 0); i < h; ++i) {
        last.push_back(static_cast<double>(r.picard_history[i].r_F));
    }
    j["r_F_history_last10"] = last;
    j["trial_history_size"] = r.trial_history.size();
    j["anderson_accepted"] = r.anderson_accepted;
    j["anderson_rejected"] = r.anderson_rejected;
    j["anderson_condition_resets"] = r.anderson_condition_resets;
    j["anderson_stagnation_restarts"] = r.anderson_stagnation_restarts;
    j["omega_floor_guard_resets"] = r.omega_floor_guard_resets;
    j["newton_activations"] = r.newton_activations;
    j["newton_steps_accepted"] = r.newton_steps_accepted;
    j["newton_step_failures"] = r.newton_step_failures;
    j["newton_rescue_events"] = r.newton_rescue_events;
    j["newton_jv_evaluations"] = r.newton_jv_evaluations;
    j["psi1_result"] = j_pcg(r.psi1_result);
    j["psi2_result"] = j_pcg(r.psi2_result);
    j["residual_final"] = j_residual(r.residual);
    j["c_percentiles_source"] = "residual_histogram_percentile(report.residual, p)";
    j["c_percentiles_abs"] = {{"p0.001", s.c_percentiles[0]},
                              {"p0.01", s.c_percentiles[1]},
                              {"p0.05", s.c_percentiles[2]},
                              {"p0.5", s.c_percentiles[3]}};
    j["diagnostics_source"] = "StreamfunctionSolveReport::diagnostics (final state)";
    j["diagnostics"] = j_diagnostics(r.diagnostics);
    j["post_diagnostics_source"] =
        "separate enqueue_streamfunction_physical_diagnostics on the final state with degeneracy "
        "thresholds (post-solve only)";
    j["post_diagnostics"] = j_diagnostics(s.post_diag);
    j["memory"] = {{"streamfunction_total_bytes", r.memory.total_bytes},
                   {"streamfunction_fine_grid_equivalent_fields", r.memory.fine_grid_equivalent_fields},
                   {"anderson_history_bytes", r.memory.anderson_history_bytes},
                   {"darcy_workspace_total_bytes", s.flow.memory.total_bytes}};
    return j;
}

void print_stage(const Options& o, int j, int k, const StageResult& s) {
    const auto& f = s.flow;
    std::printf("---- stage %d/%d  lambda = %.17g ----\n", j, k, s.lambda);
    std::printf("darcy (SF-19, qbar=e1): G = (%.17g, %.17g, %.17g)\n", (double)f.G[0], (double)f.G[1],
                (double)f.G[2]);
    std::printf("  K_eff diag = (%.17g, %.17g, %.17g)  offdiag max|.| = %.3e  sym_defect_rel = %.3e\n",
                (double)f.K_eff[0][0], (double)f.K_eff[1][1], (double)f.K_eff[2][2],
                std::max({std::fabs((double)f.K_eff[0][1]), std::fabs((double)f.K_eff[0][2]),
                          std::fabs((double)f.K_eff[1][0]), std::fabs((double)f.K_eff[1][2]),
                          std::fabs((double)f.K_eff[2][0]), std::fabs((double)f.K_eff[2][1])}),
                (double)f.symmetry_defect_rel);
    std::printf("  achieved mean flux = (%.17g, %.17g, %.17g)  div_max_abs = %.3e  div_rms = %.3e\n",
                (double)f.achieved_mean_flux[0], (double)f.achieved_mean_flux[1],
                (double)f.achieved_mean_flux[2], (double)f.div_max_abs, (double)f.div_rms);
    for (int d = 0; d < 3; ++d) {
        const auto& c = f.corrector_results[d];
        std::printf("  corrector %c: %s iters=%d rel_res=%.3e\n", "xyz"[d], pcg_label(c.status),
                    c.iterations, (double)c.relative_projected_residual);
    }
    std::printf("  darcy wall = %.3f s\n", s.wall_darcy);
    if (!s.solved) {
        std::printf("streamfunction solve NOT RUN (Darcy corrector PCG did not converge)\n");
        return;
    }
    const auto& r = s.report;
    const auto& d = r.diagnostics;
    std::printf("solve: status = %s  exit = %s  picard_iterations = %d  final_omega = %.6g  "
                "initial_state = %s\n",
                status_label(r.status), exit_label(r.exit_reason), r.picard_iterations,
                (double)r.final_omega,
                s.config.initial_state == sf::PicardInitialState::zero_source ? "zero_source"
                                                                              : "warm_start");
    std::printf("  anderson acc/rej/cond_resets = %d/%d/%d  newton activations = %d  wall = %.3f s\n",
                r.anderson_accepted, r.anderson_rejected, r.anderson_condition_resets,
                r.newton_activations, s.wall_solve);
    std::printf("  r_F initial = %.17g\n", r_f_initial(r));
    std::printf("  r_F final   = %.17g  (r1 = %.17g, r2 = %.17g)\n", (double)r.residual.r_F,
                (double)r.residual.r1, (double)r.residual.r2);
    std::printf("  tolerance = %.3g  tolerance_met = %s\n", (double)s.config.picard.tolerance,
                tolerance_met(s) ? "true" : "false");
    std::printf("  e_v         = %.17g   (post-call e_v = %.17g)\n", (double)d.e_v,
                (double)s.post_diag.e_v);
    std::printf("  rms_rel u/v/w = %.6e %.6e %.6e  linf_rel u/v/w = %.6e %.6e %.6e\n",
                (double)d.rms_u_rel, (double)d.rms_v_rel, (double)d.rms_w_rel, (double)d.linf_u_rel,
                (double)d.linf_v_rel, (double)d.linf_w_rel);
    std::printf("  corr u/v/w = %.9f %.9f %.9f  magnitude rms_rel/linf_rel = %.6e %.6e\n",
                (double)d.corr_u, (double)d.corr_v, (double)d.corr_w, (double)d.rms_magnitude_rel,
                (double)d.linf_magnitude_rel);
    std::printf("  angle rms/max = %.6e %.6e rad (included %llu, excluded %llu)\n",
                (double)d.rms_theta, (double)d.max_theta, d.angle_included_count,
                d.angle_excluded_count);
    std::printf("  invariance e_psi1 = %.6e  e_psi2 = %.6e\n", (double)d.invariance_e_psi1,
                (double)d.invariance_e_psi2);
    std::printf("  reconstructed divergence e_div = %.6e  (rms %.6e, linf %.6e)\n", (double)d.e_div,
                (double)d.rms_div, (double)d.linf_div);
    std::printf("  |c| min = %.6e  mean = %.6e  max = %.6e  v_d_rms = %.17g\n", (double)d.c_min,
                (double)d.c_mean, (double)d.c_max, (double)d.v_d_rms);
    std::printf("  |c| percentiles (histogram upper edges) 0.1%% = %.6e  1%% = %.6e  5%% = %.6e  "
                "50%% = %.6e\n",
                s.c_percentiles[0], s.c_percentiles[1], s.c_percentiles[2], s.c_percentiles[3]);
    for (int t = 0; t < s.post_diag.num_degeneracy_thresholds; ++t) {
        std::printf("  degeneracy |c| < %.0e v_rms: total %llu  low_speed %llu  unexplained %llu\n",
                    (double)s.post_diag.degeneracy_thresholds[t], s.post_diag.degeneracy_total[t],
                    s.post_diag.degeneracy_low_speed[t], s.post_diag.degeneracy_unexplained[t]);
    }
    (void)o;
}

int run(const Options& o) {
    const int N = o.n;
    const real h = real{1} / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    const std::size_t n = grid.num_cells();
    const int mg_levels = o.mg_levels_auto ? auto_mg_levels(N) : o.mg_levels;

    CudaContext ctx(0);
    const auto t_total0 = std::chrono::steady_clock::now();

    // ---- 1. Y on the host (analytic) or device (SF-18) ----
    std::vector<double> y_host;
    json field_json;
    field_json["name"] = o.field;
    if (o.field == "homogeneous") {
        y_host.assign(n, 0.0);
    } else if (o.field == "gaussian") {
        physics::PeriodicGaussianFieldConfig gcfg;
        gcfg.sigma2 = static_cast<real>(o.sigma2);
        gcfg.corr_length = static_cast<real>(o.ell);
        gcfg.seed = o.seed;
        gcfg.normalize_variance = true;
        DeviceBuffer<real> y_dev(n);
        physics::PeriodicGaussianFieldWorkspace gws;
        const physics::PeriodicGaussianFieldReport grep =
            physics::generate_periodic_gaussian_field(ctx, grid, gcfg, y_dev.span(), gws);
        ctx.synchronize();
        y_host.resize(n);
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(y_host.data(), y_dev.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
        field_json["sf18"] = {{"sigma2", o.sigma2},
                              {"corr_length", o.ell},
                              {"seed", o.seed},
                              {"normalize_variance", true},
                              {"raw_mean", static_cast<double>(grep.raw_mean)},
                              {"raw_variance", static_cast<double>(grep.raw_variance)},
                              {"applied_scale", static_cast<double>(grep.applied_scale)},
                              {"final_variance", static_cast<double>(grep.final_variance)},
                              {"active_mode_count", grep.active_mode_count}};
    } else {
        const closure_gate::AnalyticField af = closure_gate::analytic_field_from_name(o.field);
        closure_gate::fill_analytic_log_conductivity(grid, af, o.eps, y_host);
        field_json["eps"] = o.eps;
    }
    const FieldStats ys = field_stats(y_host);
    field_json["Y_mean"] = ys.mean;
    field_json["Y_variance"] = ys.variance;
    field_json["Y_min"] = ys.min;
    field_json["Y_max"] = ys.max;

    // ---- solver / flow configuration (fixed for every stage) ----
    sf::StreamfunctionSolverConfig cfg{}; // library defaults, then the explicit knobs below
    cfg.picard.max_iter = o.max_iter;
    cfg.picard.tolerance = static_cast<real>(o.tolerance);
    cfg.adaptive.enabled = true;
    cfg.anderson.enabled = o.anderson;
    cfg.anderson.depth = 5;
    cfg.anderson.start_iteration = 5;
    cfg.anderson.condition_limit = real{1e12};
    cfg.newton.enabled = o.newton;
    cfg.eta = real{1};
    cfg.epsilon = static_cast<real>(o.epsilon);
    cfg.mg.num_levels = mg_levels;

    physics::AffinePeriodicFlowConfig flow_cfg{}; // qbar = (1, 0, 0)
    flow_cfg.qbar[0] = real{1};
    flow_cfg.qbar[1] = real{0};
    flow_cfg.qbar[2] = real{0};
    flow_cfg.linear.rtol = static_cast<real>(o.pcg_rtol);
    flow_cfg.mg.num_levels = mg_levels;

    std::printf("==== streamfunction_ev_ladder (SF-30 N3; frozen periodic stack) ====\n");
    std::printf("field = %s  N = %d  h = %.17g  lambda_steps = %d\n", o.field.c_str(), N, (double)h,
                o.lambda_steps);
    if (o.field == "gaussian") {
        std::printf("  SF-18: sigma2 = %.17g  ell = %.17g  seed = %llu  normalize_variance = true\n",
                    o.sigma2, o.ell, o.seed);
    } else if (o.field != "homogeneous") {
        std::printf("  analytic: Y = eps * f, eps = %.17g\n", o.eps);
    }
    std::printf("  Y: mean = %.17g  var = %.17g  min = %.17g  max = %.17g\n", ys.mean, ys.variance,
                ys.min, ys.max);
    std::printf("config: eta = %g  epsilon = %g  tolerance = %g  max_iter = %d  omega0 = %g  "
                "adaptive = %d  anderson = %d (depth %d, start %d, cond %g)  newton = %d\n",
                (double)cfg.eta, (double)cfg.epsilon, (double)cfg.picard.tolerance,
                cfg.picard.max_iter, (double)cfg.picard.omega, cfg.adaptive.enabled ? 1 : 0,
                cfg.anderson.enabled ? 1 : 0, cfg.anderson.depth, cfg.anderson.start_iteration,
                (double)cfg.anderson.condition_limit, cfg.newton.enabled ? 1 : 0);
    std::printf("  streamfunction linear rtol = %g max_iter = %d check_every = %d; mg levels = %d "
                "(%s) pre/post = %d/%d coarse_iters = %d\n",
                (double)cfg.linear.rtol, cfg.linear.max_iter, cfg.linear.check_every,
                cfg.mg.num_levels, o.mg_levels_auto ? "auto" : "explicit", cfg.mg.pre_smooth,
                cfg.mg.post_smooth, cfg.mg.coarse_solve_iters);
    std::printf("  darcy: qbar = (1,0,0)  pcg rtol = %g max_iter = %d  mg levels = %d; gauge = "
                "AffineGauge::benchmark(1)\n",
                (double)flow_cfg.linear.rtol, flow_cfg.linear.max_iter, flow_cfg.mg.num_levels);

    // ---- buffers (allocated once; reused by every stage) ----
    const std::size_t nu = static_cast<std::size_t>(N + 1) * N * N;
    const std::size_t nv = static_cast<std::size_t>(N) * (N + 1) * N;
    const std::size_t nw = static_cast<std::size_t>(N) * N * (N + 1);
    DeviceBuffer<real> y_stage(n), k_stage(n);
    DeviceBuffer<real> flow_u(nu), flow_v(nv), flow_w(nw);
    DeviceBuffer<real> post_u(nu), post_v(nv), post_w(nw);
    std::vector<double> y_scaled(n), k_scaled(n);
    physics::AffinePeriodicFlowWorkspace flow_ws;
    sf::StreamfunctionFields fields;
    sf::StreamfunctionWorkspace workspace;
    sf::StreamfunctionDiagnosticsWorkspace post_ws;
    fields.prepare(grid);
    workspace.prepare(grid, cfg);
    post_ws.prepare(grid);

    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    const sf::AffineGauge gauge = sf::AffineGauge::benchmark(real{1});

    sf::PhysicalDiagnosticsConfig post_cfg = cfg.diagnostics;
    post_cfg.num_degeneracy_thresholds = kNumPostThresholds;
    for (int t = 0; t < kNumPostThresholds; ++t) post_cfg.degeneracy_thresholds[t] = kPostThresholds[t];

    std::vector<StageResult> stages;
    bool darcy_failed = false;
    const int K = o.lambda_steps;
    for (int j = 1; j <= K; ++j) {
        StageResult s;
        s.lambda = (K == 1) ? 1.0 : static_cast<double>(j) / static_cast<double>(K);
        for (std::size_t c = 0; c < n; ++c) {
            y_scaled[c] = (K == 1) ? y_host[c] : s.lambda * y_host[c];
            k_scaled[c] = std::exp(y_scaled[c]);
        }
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(y_stage.data(), y_scaled.data(), n * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(k_stage.data(), k_scaled.data(), n * sizeof(real),
                                          cudaMemcpyHostToDevice));

        // ---- 2. Darcy reference ----
        const auto t0 = std::chrono::steady_clock::now();
        s.flow = physics::solve_affine_periodic_flow(
            ctx, grid, DeviceSpan<const real>(k_stage.span()), flow_cfg,
            physics::AffinePeriodicVelocityView{flow_u.span(), flow_v.span(), flow_w.span()},
            flow_ws);
        ctx.synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        s.wall_darcy = std::chrono::duration<double>(t1 - t0).count();
        s.darcy_ok = s.flow.corrector_results[0].converged && s.flow.corrector_results[1].converged &&
                     s.flow.corrector_results[2].converged;
        if (!s.darcy_ok) {
            darcy_failed = true;
            print_stage(o, j, K, s);
            stages.push_back(s);
            break;
        }

        // ---- 3. problem view (as ContinuationController builds it) ----
        sf::StreamfunctionProblemView view;
        view.grid = grid;
        view.conductivity = DeviceSpan<const real>(y_stage.span());
        view.conductivity_representation = sf::ConductivityRepresentation::log_conductivity_y;
        view.darcy_velocity = sf::CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                              DeviceSpan<const real>(flow_v.span()),
                                                              DeviceSpan<const real>(flow_w.span())};
        view.bc = bc;
        view.gauge = gauge;

        s.config = cfg;
        s.config.initial_state =
            (j == 1) ? sf::PicardInitialState::zero_source : sf::PicardInitialState::warm_start;
        s.config.coefficient_state = sf::CoefficientState::rebuild; // K changes every stage

        // ---- 4. one solve: whatever it leaves is the stack's answer ----
        const auto t2 = std::chrono::steady_clock::now();
        s.report = sf::solve_streamfunctions(ctx, view, s.config, fields, workspace);
        ctx.synchronize();
        const auto t3 = std::chrono::steady_clock::now();
        s.wall_solve = std::chrono::duration<double>(t3 - t2).count();
        s.solved = true;

        // ---- 5. |c| percentiles from the final head residual histogram;
        //         degeneracy counts from one post-solve diagnostics call ----
        for (int p = 0; p < 4; ++p) {
            s.c_percentiles[p] = static_cast<double>(
                sf::residual_histogram_percentile(s.report.residual, static_cast<real>(kPercentiles[p])));
        }
        sf::enqueue_streamfunction_physical_diagnostics(
            ctx, grid, fields.fluctuations(), gauge, view.darcy_velocity, post_cfg,
            sf::CompactMacVelocityView{post_u.span(), post_v.span(), post_w.span()}, post_ws);
        s.post_diag =
            sf::synchronize_streamfunction_physical_diagnostics_report(ctx, grid, post_cfg, post_ws);

        print_stage(o, j, K, s);
        stages.push_back(s);
    }
    const double wall_total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total0).count();

    std::size_t free_b = 0, total_b = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    const StageResult& fin = stages.back();
    std::printf("==== final: lambda = %.17g  status = %s  r_F = %.17g  tolerance_met = %s  e_v = "
                "%.17g  wall_total = %.3f s ====\n",
                fin.lambda, fin.solved ? status_label(fin.report.status) : "not_run",
                fin.solved ? (double)fin.report.residual.r_F : std::nan(""),
                tolerance_met(fin) ? "true" : "false",
                fin.solved ? (double)fin.report.diagnostics.e_v : std::nan(""), wall_total);
    if (K > 1) {
        std::printf("ladder summary (lambda, status, iterations, r_F, e_v):\n");
        for (const auto& s : stages) {
            std::printf("  %.6f  %-14s %5d  %.6e  %.6e\n", s.lambda,
                        s.solved ? status_label(s.report.status) : "not_run",
                        s.solved ? s.report.picard_iterations : -1,
                        s.solved ? (double)s.report.residual.r_F : std::nan(""),
                        s.solved ? (double)s.report.diagnostics.e_v : std::nan(""));
        }
    }

    if (!o.out.empty()) {
        json rec;
        rec["tool"] = "streamfunction_ev_ladder";
        rec["increment"] = "SF-30 N3";
        rec["options"] = {{"field", o.field},
                          {"n", o.n},
                          {"h", static_cast<double>(h)},
                          {"eps", o.eps},
                          {"sigma2", o.field == "gaussian" ? json(o.sigma2) : json(nullptr)},
                          {"ell", o.field == "gaussian" ? json(o.ell) : json(nullptr)},
                          {"seed", o.field == "gaussian" ? json(o.seed) : json(nullptr)},
                          {"epsilon", o.epsilon},
                          {"tolerance", o.tolerance},
                          {"max_iter", o.max_iter},
                          {"anderson", o.anderson},
                          {"newton", o.newton},
                          {"lambda_steps", o.lambda_steps},
                          {"pcg_rtol", o.pcg_rtol},
                          {"mg_levels", mg_levels},
                          {"mg_levels_mode", o.mg_levels_auto ? "auto" : "explicit"},
                          {"out", o.out}};
        rec["field"] = field_json;
        rec["gauge"] = {{"psi1_gradient", {0.0, 1.0, 0.0}}, {"psi2_gradient", {0.0, 0.0, 1.0}},
                        {"rule", "AffineGauge::benchmark(1)"}};
        rec["conductivity_representation"] = "log_conductivity_y";
        rec["darcy_config"] = j_flow_config(flow_cfg);
        rec["solver_config"] = j_solver_config(stages.back().solved ? stages.back().config : cfg);
        json st = json::array();
        for (const auto& s : stages) st.push_back(j_stage_summary(s));
        rec["stages"] = st;
        rec["final_stage"] = j_stage_full(fin);
        rec["tolerance_met"] = tolerance_met(fin);
        rec["darcy_failed"] = darcy_failed;
        rec["wall_seconds_total"] = wall_total;
        rec["device_memory"] = {{"cuda_free_bytes", free_b}, {"cuda_total_bytes", total_b}};
        std::ofstream os(o.out);
        if (!os) throw std::runtime_error("cannot open --out file '" + o.out + "'");
        os << rec.dump(2) << '\n';
        if (!os) throw std::runtime_error("failed writing --out file '" + o.out + "'");
        std::printf("record written to %s\n", o.out.c_str());
    }
    return darcy_failed ? 3 : 0;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse_options(argc, argv);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return 2;
    }
    try {
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "streamfunction_ev_ladder: exception: %s\n", e.what());
        return 1;
    }
}
