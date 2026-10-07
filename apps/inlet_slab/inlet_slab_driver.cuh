#pragma once

/**
 * @file inlet_slab_driver.cuh
 * @brief SF-33 N5: logic of the `inlet_slab` experiment driver (documented-experiment instrument,
 *        not a ctest entry). Three modes:
 *
 *   --proto <case_dir>        claim (a): the GPU solves the SF-29 prototype's discrete problem on
 *                             the exported prototype inputs (N4 `load_proto_case`,
 *                             `ProtoStageProvider`), N2 continuation, prototype-format lines
 *                             (STAGE / NEWTON / LINEAR / STAGE_END / CONTINUATION / PATH / CASE /
 *                             EXTRA / HISTORY), optional FIELDDIFF against a converted prototype
 *                             solution.
 *   --production              claim (b): N3 production stages (SF-18 field or an analytic
 *                             closure field, SF-19 Darcy per continuation amplitude, D-1 inlet
 *                             labels, SF-28 splines), N2 continuation, production oracle (SF-30
 *                             integrator) at the target amplitude, the same report lines.
 *   --sf19-crosscheck <dir>   step 8: SF-19 inlet-face v1 and spline-flow v_perp of the exported
 *                             SF-29 `gauss` field vs the prototype's spectral reference.
 *   --linear-probe <case_dir> SF-33 N7c (discriminating experiment): continuation to --eps-from
 *                             (exactly as the solver: ladder entries below it, or --probe-ladder),
 *                             k = --newton-steps Newton steps of the stage --eps-stage with the
 *                             driver policy (Psi-tc / forcing options; P-A only), then the Jacobian
 *                             is FROZEN at that iterate and (J + mu D) p = -E is solved with GMRES
 *                             (--restart, --max-inner, tol --probe-tol) for every preconditioner of
 *                             --probe-precs (pa | multP | addP, P = coarse profiles) and mu in
 *                             {mu_SER of that iterate, 0} (--probe-mu). PROBE / PROBE_CURVE lines.
 *
 * SF-33 N7c also adds `--coarse off|add|mult --coarse-profiles 1|2` (Galerkin coarse correction
 * on top of P-A in the Newton solves; SlabCoarseCorrection.cuh).
 *
 * SF-33 C3 (driver production defaults): `--coarse mult --coarse-profiles 2 --coarse-assembly
 * colored --coarse-factor banded` (the N7c-validated solver), `--psitc on`, `--forcing ew`,
 * `--restart 100`, `--max-inner 6000`, `--max-newton 120` (40 when `--psitc off` and not given).
 * `--coarse off` restores the N7b solver bitwise. `--gmres-stagnation-factor f` (default 0.9, the
 * prototype rule) sets SlabGmresConfig::stagnation_factor of the Newton solves. Library defaults
 * (SlabNewtonConfig: coarse off) are unchanged.
 *
 * Nothing is clamped or regularized here: the driver only orchestrates N0-N4 and reports. Every
 * terminal status is printed as `STATUS <name>` and mapped to a distinct exit code (ExitCode).
 *
 * Memory accounting: cudaMemGetInfo is polled before / after every phase (device-wide: other
 * processes on the same GPU are included in `peak_device_bytes = total - free_min`); the sum of
 * the solver / metrics workspaces' allocated_bytes() is reported separately (process-local).
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletLabels.cuh"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/physics/streamfunctions/inlet_slab/SlabCoarseCorrection.cuh"
#include "src/physics/streamfunctions/inlet_slab/ProtoCase.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabNewtonKrylov.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabOracle.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabSolverTypes.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace macroflow3d {
namespace inlet_slab_app {

namespace sl = streamfunctions::inlet_slab;
using json = nlohmann::json;

// ================================================================================================
// Exit codes and statuses
// ================================================================================================

/// Distinct exit code per terminal status (spec item 10; nothing is mapped onto another status).
enum ExitCode : int {
    kExitConverged = 0,
    kExitException = 1,
    kExitUsage = 2,
    kExitLinesearchFail = 10,
    kExitStagnation = 11,
    kExitMaxit = 12,
    kExitLinearFailure = 13,
    kExitNanInf = 14,
    kExitContinuationFloor = 15,
    kExitMissingStageInput = 16,
    kExitInletBackflow = 17,
    kExitDarcyFailed = 18,
    kExitOracleRoundtripFail = 19,
    kExitNotRun = 20,
};

inline int exit_code_of(const std::string& status) {
    if (status == "converged" || status == "ok")
        return kExitConverged;
    if (status == "linesearch-fail")
        return kExitLinesearchFail;
    if (status == "stagnation")
        return kExitStagnation;
    if (status == "maxit")
        return kExitMaxit;
    if (status == "linear_failure")
        return kExitLinearFailure;
    if (status == "nan_inf")
        return kExitNanInf;
    if (status == "continuation_floor")
        return kExitContinuationFloor;
    if (status == "missing_stage_input")
        return kExitMissingStageInput;
    if (status == "inlet_backflow")
        return kExitInletBackflow;
    if (status == "darcy_failed")
        return kExitDarcyFailed;
    if (status == "oracle_roundtrip_fail")
        return kExitOracleRoundtripFail;
    return kExitNotRun;
}

class UsageError : public std::runtime_error {
  public:
    explicit UsageError(const std::string& m) : std::runtime_error(m) {}
};

/// A production stage that could not be built (darcy_failed / inlet_backflow): stops the run.
class StageBuildFailure : public std::runtime_error {
  public:
    StageBuildFailure(const std::string& status, real amp, const std::string& m)
        : std::runtime_error(m), status_(status), amp_(amp) {}
    const std::string& status() const noexcept { return status_; }
    real amplitude() const noexcept { return amp_; }

  private:
    std::string status_;
    real amp_;
};

// ================================================================================================
// Configuration
// ================================================================================================

enum class Mode { none, proto, production, crosscheck, linear_probe };

struct DriverConfig {
    Mode mode = Mode::none;
    // proto
    std::string case_dir;
    std::string solution_dir;
    bool has_eps_target = false;
    real eps_target = 0.0;
    // production
    int n = 0;
    bool has_eps = false;
    real eps = 0.0;
    real sigma2 = 1.0;
    real ell = 0.25;
    unsigned long long seed = 3001ULL;
    std::string analytic; ///< empty: SF-18 gaussian field
    int oracle_hmax_div = 8;
    real oracle_tol = 1e-8;
    real oracle_max_roundtrip = 1e-8; ///< acceptance (d): round trip <= 1e-8 on every plane
    bool oracle_ladder = false;
    bool no_oracle = false;
    int threads = 0; ///< 0: min(hardware_concurrency, 32)
    real pcg_rtol = 1e-10;
    // crosscheck
    std::string crosscheck_dir;
    // solver
    real lin_tol = 1e-12;
    int restart = 100;
    int max_inner = 6000;
    real newton_tol = 1e-13;
    int max_newton = 40; ///< effective value after parse_args (default 120 with --psitc on)
    bool max_newton_given = false;
    int bisect = 4;
    std::string prec = "pa";
    std::string forcing = "ew"; ///< SF-33 N7a: fixed | ew (Eisenstat-Walker choice 2)
    real ew_eta_max = 0.1;
    real ew_eta0 = 0.1;
    std::string psitc = "on"; ///< SF-33 N7b: pseudo-transient continuation (SER shift) on | off
    real psitc_mu0 = 1.0;
    real psitc_mu_max = 100.0;
    // SF-33 C3: driver production defaults = the N7c-validated solver (mult, 2 profiles,
    // colored assembly, banded host LU); the library default (SlabNewtonConfig) stays off.
    std::string coarse = "mult"; ///< off | add | mult (Galerkin coarse correction, N7c)
    int coarse_profiles = 2;     ///< 1 (x1-constant) | 2 (+ x1-linear)
    std::string coarse_assembly = "colored"; ///< direct (K applies) | colored (productization)
    std::string coarse_factor = "banded";    ///< dense host LU | banded host LU (needs colored)
    /// SF-33 C3: GMRES restart-stagnation factor of the Newton solves (SlabGmresConfig):
    /// stop when the true residual at a restart > f x the one two restarts earlier.
    real gmres_stagnation_factor = 0.9;
    // linear probe (SF-33 N7c)
    bool has_eps_from = false;
    real eps_from = 0.0;
    bool has_eps_stage = false;
    real eps_stage = 0.0;
    int newton_steps = -1;
    std::string probe_precs = "pa,mult1,mult2,add1";
    real probe_tol = 1e-8;
    std::string probe_mu = "both"; ///< ser | zero | both
    real probe_stagnation = 1.0;   ///< GMRES restart-stagnation factor in the probe (1: cap only)
    std::string probe_ladder;      ///< empty: the prototype ladder (0.25, 0.5, 1)
    // outputs
    std::string save_solution_dir;
    std::string summary_path;
    int device = 0;
    std::string command_line;
};

inline const char* usage_text() {
    return "usage:\n"
           "  inlet_slab --proto <case_dir> [--eps-target E] [--solution <solution_dir>]\n"
           "             [--save-solution <dir>] [--summary <json>] [solver options]\n"
           "  inlet_slab --production --n N --eps E [--sigma2 1 --ell 0.25 --seed 3001 | "
           "--analytic <field>]\n"
           "             [--oracle-hmax-div 8] [--oracle-tol 1e-8] [--oracle-max-roundtrip 1e-8]\n"
           "             [--oracle-ladder] [--no-oracle] [--threads T] [--pcg-rtol 1e-10]\n"
           "             [--save-solution <dir>] [--summary <json>] [solver options]\n"
           "  inlet_slab --sf19-crosscheck <crosscheck_dir> [--pcg-rtol 1e-10] [--summary <json>]\n"
           "  inlet_slab --linear-probe <case_dir> --eps-stage E --newton-steps k [--eps-from A]\n"
           "             [--probe-ladder 0.25,0.375] [--probe-precs pa,mult1,mult2,add1]\n"
           "             [--probe-tol 1e-8] [--probe-mu both|ser|zero] [--probe-stagnation 1]\n"
           "             [--summary <json>] [solver options]\n"
           "solver options: [--lin-tol 1e-12] [--restart 100] [--max-inner 6000] [--newton-tol "
           "1e-13]\n"
           "                [--max-newton 120 (psitc on) | 40 (psitc off)] [--bisect 4]\n"
           "                [--prec pa] [--device 0] [--forcing ew|fixed] [--ew-eta-max 0.1]\n"
           "                [--ew-eta0 0.1] [--psitc on|off] [--psitc-mu0 1] [--psitc-mu-max 100]\n"
           "                [--coarse mult|add|off] [--coarse-profiles 2|1]\n"
           "                [--coarse-assembly colored|direct] [--coarse-factor banded|dense]\n"
           "                [--gmres-stagnation-factor 0.9]\n"
           "exit codes: 0 converged; 1 exception; 2 usage; 10 linesearch-fail; 11 stagnation; 12 "
           "maxit;\n"
           "            13 linear_failure; 14 nan_inf; 15 continuation_floor; 16 "
           "missing_stage_input;\n"
           "            17 inlet_backflow; 18 darcy_failed; 19 oracle_roundtrip_fail\n";
}

namespace detail {

inline double parse_double(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0' || errno == ERANGE || !std::isfinite(v))
        throw UsageError(opt + ": invalid number '" + s + "'");
    return v;
}

inline int parse_int(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno == ERANGE || v < -2147483647LL || v > 2147483647LL)
        throw UsageError(opt + ": invalid integer '" + s + "'");
    return static_cast<int>(v);
}

inline unsigned long long parse_u64(const std::string& opt, const std::string& s) {
    if (s.empty() || s[0] == '-' || s[0] == '+')
        throw UsageError(opt + ": invalid unsigned integer '" + s + "'");
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE)
        throw UsageError(opt + ": invalid unsigned integer '" + s + "'");
    return v;
}

} // namespace detail

inline DriverConfig parse_args(int argc, char** argv) {
    using namespace detail;
    DriverConfig c;
    for (int i = 0; i < argc; ++i) {
        if (i > 0)
            c.command_line += " ";
        c.command_line += argv[i];
    }
    std::vector<std::string> seen;
    auto set_mode = [&](Mode m) {
        if (c.mode != Mode::none)
            throw UsageError("exactly one of --proto, --production, --sf19-crosscheck, "
                             "--linear-probe");
        c.mode = m;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        if (opt.rfind("--", 0) != 0)
            throw UsageError("unexpected argument '" + opt + "'");
        if (std::find(seen.begin(), seen.end(), opt) != seen.end())
            throw UsageError("option given twice: " + opt);
        seen.push_back(opt);
        // flags without a value
        if (opt == "--production") {
            set_mode(Mode::production);
            continue;
        }
        if (opt == "--oracle-ladder") {
            c.oracle_ladder = true;
            continue;
        }
        if (opt == "--no-oracle") {
            c.no_oracle = true;
            continue;
        }
        if (i + 1 >= argc)
            throw UsageError(opt + " requires a value");
        const std::string val = argv[++i];
        if (opt == "--proto") {
            set_mode(Mode::proto);
            c.case_dir = val;
        } else if (opt == "--sf19-crosscheck") {
            set_mode(Mode::crosscheck);
            c.crosscheck_dir = val;
        } else if (opt == "--linear-probe") {
            set_mode(Mode::linear_probe);
            c.case_dir = val;
        } else if (opt == "--eps-from") {
            c.eps_from = parse_double(opt, val);
            c.has_eps_from = true;
        } else if (opt == "--eps-stage") {
            c.eps_stage = parse_double(opt, val);
            c.has_eps_stage = true;
        } else if (opt == "--newton-steps") {
            c.newton_steps = parse_int(opt, val);
        } else if (opt == "--probe-precs") {
            c.probe_precs = val;
        } else if (opt == "--probe-tol") {
            c.probe_tol = parse_double(opt, val);
        } else if (opt == "--probe-mu") {
            c.probe_mu = val;
        } else if (opt == "--probe-stagnation") {
            c.probe_stagnation = parse_double(opt, val);
        } else if (opt == "--probe-ladder") {
            c.probe_ladder = val;
        } else if (opt == "--coarse") {
            c.coarse = val;
        } else if (opt == "--coarse-profiles") {
            c.coarse_profiles = parse_int(opt, val);
        } else if (opt == "--coarse-assembly") {
            c.coarse_assembly = val;
        } else if (opt == "--coarse-factor") {
            c.coarse_factor = val;
        } else if (opt == "--gmres-stagnation-factor") {
            c.gmres_stagnation_factor = parse_double(opt, val);
        } else if (opt == "--eps-target") {
            c.eps_target = parse_double(opt, val);
            c.has_eps_target = true;
        } else if (opt == "--solution") {
            c.solution_dir = val;
        } else if (opt == "--save-solution") {
            c.save_solution_dir = val;
        } else if (opt == "--summary") {
            c.summary_path = val;
        } else if (opt == "--n") {
            c.n = parse_int(opt, val);
        } else if (opt == "--eps") {
            c.eps = parse_double(opt, val);
            c.has_eps = true;
        } else if (opt == "--sigma2") {
            c.sigma2 = parse_double(opt, val);
        } else if (opt == "--ell") {
            c.ell = parse_double(opt, val);
        } else if (opt == "--seed") {
            c.seed = parse_u64(opt, val);
        } else if (opt == "--analytic") {
            c.analytic = val;
        } else if (opt == "--oracle-hmax-div") {
            c.oracle_hmax_div = parse_int(opt, val);
        } else if (opt == "--oracle-tol") {
            c.oracle_tol = parse_double(opt, val);
        } else if (opt == "--oracle-max-roundtrip") {
            c.oracle_max_roundtrip = parse_double(opt, val);
        } else if (opt == "--threads") {
            c.threads = parse_int(opt, val);
        } else if (opt == "--pcg-rtol") {
            c.pcg_rtol = parse_double(opt, val);
        } else if (opt == "--lin-tol") {
            c.lin_tol = parse_double(opt, val);
        } else if (opt == "--restart") {
            c.restart = parse_int(opt, val);
        } else if (opt == "--max-inner") {
            c.max_inner = parse_int(opt, val);
        } else if (opt == "--newton-tol") {
            c.newton_tol = parse_double(opt, val);
        } else if (opt == "--max-newton") {
            c.max_newton = parse_int(opt, val);
            c.max_newton_given = true;
        } else if (opt == "--bisect") {
            c.bisect = parse_int(opt, val);
        } else if (opt == "--prec") {
            c.prec = val;
        } else if (opt == "--forcing") {
            c.forcing = val;
        } else if (opt == "--ew-eta-max") {
            c.ew_eta_max = parse_double(opt, val);
        } else if (opt == "--ew-eta0") {
            c.ew_eta0 = parse_double(opt, val);
        } else if (opt == "--psitc") {
            c.psitc = val;
        } else if (opt == "--psitc-mu0") {
            c.psitc_mu0 = parse_double(opt, val);
        } else if (opt == "--psitc-mu-max") {
            c.psitc_mu_max = parse_double(opt, val);
        } else if (opt == "--device") {
            c.device = parse_int(opt, val);
        } else {
            throw UsageError("unknown option " + opt);
        }
    }
    if (c.mode == Mode::none)
        throw UsageError("one of --proto, --production, --sf19-crosscheck, --linear-probe is "
                         "required");
    if (c.prec != "pa")
        throw UsageError("--prec: only 'pa' (per-mode plane-averaged preconditioner P-A) exists");
    if (c.forcing != "ew" && c.forcing != "fixed")
        throw UsageError("--forcing: 'ew' (Eisenstat-Walker, default) or 'fixed' (lin_tol)");
    if (!(c.ew_eta_max > 0.0 && c.ew_eta_max < 1.0) || !(c.ew_eta0 > 0.0 && c.ew_eta0 < 1.0))
        throw UsageError("--ew-eta-max / --ew-eta0 must lie in (0, 1)");
    if (c.forcing == "ew" && c.lin_tol > c.ew_eta_max)
        throw UsageError("--forcing ew requires --lin-tol (eta_min) <= --ew-eta-max");
    if (c.psitc != "on" && c.psitc != "off")
        throw UsageError("--psitc: 'on' (pseudo-transient continuation, default) or 'off'");
    if (!(c.psitc_mu0 >= 0.0) || !(c.psitc_mu_max >= c.psitc_mu0))
        throw UsageError("--psitc-mu0 / --psitc-mu-max: 0 <= mu0 <= mu_max");
    if (c.coarse != "off" && c.coarse != "add" && c.coarse != "mult")
        throw UsageError("--coarse: mult (default) | add | off");
    if (c.coarse_profiles != 1 && c.coarse_profiles != 2)
        throw UsageError("--coarse-profiles: 1 | 2");
    if (c.coarse_assembly != "direct" && c.coarse_assembly != "colored")
        throw UsageError("--coarse-assembly: colored (default) | direct");
    if (c.coarse_factor != "dense" && c.coarse_factor != "banded")
        throw UsageError("--coarse-factor: banded (default) | dense");
    if (c.coarse_factor == "banded" && c.coarse_assembly != "colored")
        throw UsageError("--coarse-factor banded (default) requires --coarse-assembly colored "
                         "(use --coarse-factor dense with --coarse-assembly direct)");
    if (!(c.gmres_stagnation_factor > 0.0))
        throw UsageError("--gmres-stagnation-factor must be > 0");
    if (c.mode == Mode::linear_probe) {
        if (!c.has_eps_stage || !(c.eps_stage > 0.0))
            throw UsageError("--linear-probe requires --eps-stage E > 0");
        if (c.newton_steps < 0)
            throw UsageError("--linear-probe requires --newton-steps k >= 0");
        if (c.has_eps_from && !(c.eps_from >= 0.0 && c.eps_from < c.eps_stage))
            throw UsageError("--eps-from must satisfy 0 <= A < --eps-stage");
        if (c.probe_mu != "both" && c.probe_mu != "ser" && c.probe_mu != "zero")
            throw UsageError("--probe-mu: both | ser | zero");
        if (!(c.probe_tol > 0.0) || !(c.probe_stagnation > 0.0))
            throw UsageError("--probe-tol / --probe-stagnation must be > 0");
    }
    // SF-33 N7b: SER-damped steps converge linearly while mu is large; Psi-tc default 120.
    if (!c.max_newton_given)
        c.max_newton = c.psitc == "on" ? 120 : 40;
    if (c.lin_tol <= 0.0 || c.newton_tol <= 0.0 || c.restart < 1 || c.max_inner < 1 ||
        c.max_newton < 1 || c.bisect < 0)
        throw UsageError("solver options must be positive (bisect >= 0)");
    if (c.mode == Mode::production) {
        if (c.n < 8 || c.n % 2 != 0)
            throw UsageError("--production requires --n N (even, >= 8)");
        if (!c.has_eps || !(c.eps > 0.0))
            throw UsageError("--production requires --eps E > 0");
        if (c.oracle_hmax_div < 1)
            throw UsageError("--oracle-hmax-div must be >= 1");
        if (!(c.oracle_tol > 0.0) || !(c.oracle_max_roundtrip > 0.0))
            throw UsageError("--oracle-tol / --oracle-max-roundtrip must be > 0");
        if (!(c.sigma2 > 0.0) || !(c.ell > 0.0))
            throw UsageError("--sigma2 / --ell must be > 0");
    } else if (c.mode == Mode::proto) {
        if (c.has_eps_target && !(c.eps_target > 0.0))
            throw UsageError("--eps-target must be > 0");
    }
    if (c.threads == 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        c.threads = static_cast<int>(std::max(1u, std::min(hc == 0 ? 1u : hc, 32u)));
    }
    if (c.threads < 1)
        throw UsageError("--threads must be >= 1");
    return c;
}

// ================================================================================================
// Small utilities
// ================================================================================================

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

inline void out_line(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

/// Prints every line of a multi-line block with a prefix.
inline void out_block(const std::string& prefix, const std::string& block) {
    std::size_t a = 0;
    while (a < block.size()) {
        std::size_t b = block.find('\n', a);
        if (b == std::string::npos)
            b = block.size();
        if (b > a)
            out_line(prefix + block.substr(a, b - a));
        a = b + 1;
    }
}

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) {
    return DeviceSpan<const real>(b.data(), b.size());
}
inline DeviceSpan<real> mspan(DeviceBuffer<real>& b) {
    return DeviceSpan<real>(b.data(), b.size());
}

inline std::vector<real> download(const real* d, std::size_t n) {
    // SF-33 C2: ctx-stream work must land before a legacy-stream D2H copy
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<real> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

/// JSON number or null (NaN / inf are not representable in JSON).
inline json jnum(double v) {
    return std::isfinite(v) ? json(v) : json(nullptr);
}

inline json jvec(const std::vector<real>& v) {
    json a = json::array();
    for (real x : v)
        a.push_back(jnum(x));
    return a;
}

inline void make_dir(const std::string& d) {
    if (d.empty())
        return;
    std::string acc;
    for (std::size_t i = 0; i <= d.size(); ++i) {
        if (i == d.size() || d[i] == '/') {
            if (!acc.empty())
                ::mkdir(acc.c_str(), 0755); // EEXIST is fine; failures surface on write
        }
        if (i < d.size())
            acc.push_back(d[i]);
    }
}

inline real median_of(std::vector<int> v) {
    if (v.empty())
        return std::nan("");
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 == 1 ? static_cast<real>(v[n / 2]) : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ================================================================================================
// Phase timing and memory polling
// ================================================================================================

class PhaseLog {
  public:
    PhaseLog() { poll(); }
    void begin(const std::string& name) {
        poll();
        cur_ = name;
        free_before_ = last_free_;
        t0_ = std::chrono::steady_clock::now();
    }
    void end() {
        const double s = seconds_since(t0_);
        poll();
        json p;
        p["phase"] = cur_;
        p["seconds"] = s;
        p["free_before"] = free_before_;
        p["free_after"] = last_free_;
        phases_.push_back(p);
        secs_[cur_] += s;
    }
    void poll() {
        std::size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
            total_ = tot;
            last_free_ = fr;
            free_min_ = std::min(free_min_, fr);
        }
    }
    double seconds(const std::string& name) const {
        auto it = secs_.find(name);
        return it == secs_.end() ? 0.0 : it->second;
    }
    std::size_t total() const { return total_; }
    std::size_t free_min() const { return free_min_; }
    std::size_t peak_used() const { return free_min_ <= total_ ? total_ - free_min_ : 0; }
    const json& phases() const { return phases_; }
    const std::map<std::string, double>& by_name() const { return secs_; }

  private:
    std::string cur_;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
    std::size_t free_before_ = 0, last_free_ = 0, total_ = 0;
    std::size_t free_min_ = std::numeric_limits<std::size_t>::max();
    json phases_ = json::array();
    std::map<std::string, double> secs_;
};

// ================================================================================================
// Report pieces
// ================================================================================================

inline json metrics_json(const sl::SlabMetrics& m) {
    json j;
    j["e_v"] = jnum(m.e_v);
    j["e_i1"] = jnum(m.e_i1);
    j["e_i2"] = jnum(m.e_i2);
    j["e_div"] = jnum(m.e_div);
    j["min_c"] = jnum(m.min_c);
    j["p0.1"] = jnum(m.p0_1);
    j["p1"] = jnum(m.p1);
    j["p5"] = jnum(m.p5);
    j["p50"] = jnum(m.p50);
    j["vD_min"] = jnum(m.vD_min);
    j["vD_p"] = {jnum(m.vD_p[0]), jnum(m.vD_p[1]), jnum(m.vD_p[2]), jnum(m.vD_p[3])};
    j["v_rms"] = jnum(m.v_rms);
    j["nonfinite"] = m.nonfinite;
    if (m.has_psi) {
        j["e_psi"] = jnum(m.e_psi);
        j["e_psi1"] = jnum(m.e_psi1);
        j["e_psi2"] = jnum(m.e_psi2);
        j["a_psi1"] = jnum(m.a_psi1);
        j["a_psi2"] = jnum(m.a_psi2);
        j["den_psi"] = {jnum(m.den_used1), jnum(m.den_used2)};
    }
    return j;
}

/// Per-stage GMRES statistics of a Newton report (the preconditioner gate report).
struct LinearStats {
    int steps = 0, its_max = 0, its_total = 0;
    real its_median = std::nan("");
    real rel_max = 0.0;
    int failures = 0;
};

inline LinearStats linear_stats(const sl::SlabNewtonReport& r) {
    LinearStats s;
    std::vector<int> its;
    for (const auto& st : r.steps) {
        ++s.steps;
        // per linear SOLVE (SF-33 N7b: a Psi-tc line-search retry is an extra solve of the step;
        // without retries this is one entry per step, as before)
        for (int li : st.linear_its_solves) {
            its.push_back(li);
            s.its_max = std::max(s.its_max, li);
        }
        if (st.linear_its_solves.empty()) { // defensive: reports built without the per-solve list
            its.push_back(st.linear.iterations);
            s.its_max = std::max(s.its_max, st.linear.iterations);
        }
        s.its_total += std::max(st.linear_iterations_all, st.linear.iterations);
        if (std::isfinite(st.linear.rel_residual))
            s.rel_max = std::max(s.rel_max, st.linear.rel_residual);
        if (st.linear.status != sl::SlabLinearStatus::converged)
            ++s.failures;
    }
    s.its_median = median_of(its);
    return s;
}

inline json newton_json(const sl::SlabNewtonReport& r) {
    json j;
    j["status"] = sl::to_string(r.status);
    j["forcing"] = sl::to_string(r.forcing);
    j["psitc"] = r.psitc;
    j["linesearch_retries"] = r.linesearch_retries;
    j["its"] = r.its;
    j["r_F"] = jnum(r.r_F);
    j["r_out"] = jnum(r.r_out);
    j["hist_r_F"] = jvec(r.hist_r_F);
    j["hist_r_out"] = jvec(r.hist_r_out);
    j["seconds"] = r.seconds;
    j["linear_iterations_total"] = r.linear_iterations_total;
    j["linear_iterations_max"] = r.linear_iterations_max;
    json steps = json::array();
    for (const auto& st : r.steps) {
        json s;
        s["r_F"] = jnum(st.r_F);
        s["r_out"] = jnum(st.r_out);
        s["lambda"] = jnum(st.lambda);
        s["eta"] = jnum(st.eta);
        s["mu"] = jnum(st.mu);
        s["mu_ser"] = jnum(st.mu_ser);
        s["mu_retries"] = jvec(st.mu_retries);
        s["lin_its_all"] = st.linear_iterations_all;
        s["lin_its_solves"] = st.linear_its_solves;
        s["dx_max"] = jnum(st.dx_max);
        s["dx_l2"] = jnum(st.dx_l2);
        s["lin_status"] = sl::to_string(st.linear.status);
        s["lin_its"] = st.linear.iterations;
        s["lin_cycles"] = st.linear.cycles;
        s["lin_rel"] = jnum(st.linear.rel_residual);
        s["lin_rec"] = jnum(st.linear.rel_recurrence);
        s["lin_cycle_true"] = jvec(st.linear.cycle_true);      // per restart cycle (N7a)
        s["lin_cycle_rec"] = jvec(st.linear.cycle_recurrence); // per restart cycle (N7a)
        s["t_lin"] = st.t_lin;
        s["t_fact"] = st.t_fact;
        s["prec_singular_modes"] = st.prec_singular_modes;
        steps.push_back(s);
    }
    j["steps"] = steps;
    const LinearStats ls = linear_stats(r);
    j["gmres"] = {{"steps", ls.steps},
                  {"its_max", ls.its_max},
                  {"its_median", jnum(ls.its_median)},
                  {"its_total", ls.its_total},
                  {"rel_max", jnum(ls.rel_max)},
                  {"failures", ls.failures}};
    return j;
}

inline json continuation_json(const sl::SlabContinuationReport& r) {
    json j;
    j["status"] = sl::to_string(r.status);
    j["path"] = r.path;
    j["bisections"] = r.bisections;
    j["floor_reached"] = r.floor_reached;
    j["eps_accepted"] = r.eps_accepted;
    j["seconds"] = r.seconds;
    json st = json::array();
    for (const auto& s : r.stages) {
        json e;
        e["eps"] = s.eps;
        e["from_eps"] = s.from_eps;
        e["warm_start"] = s.warm_start;
        e["final_attempt"] = s.final_attempt;
        e["accepted"] = s.accepted;
        e["newton"] = newton_json(s.newton);
        st.push_back(e);
    }
    j["stages"] = st;
    return j;
}

/// GMRES_STATS line per stage (the preconditioner gate report, parsed by compare_proto.py).
inline void print_gmres_stats(const std::string& field, real eps, int N,
                              const sl::SlabContinuationReport& r) {
    for (const auto& s : r.stages) {
        const LinearStats ls = linear_stats(s.newton);
        std::string etas, mus;
        for (const auto& st : s.newton.steps) {
            etas += (etas.empty() ? "" : ",") + fmt("%.2e", st.eta);
            mus += (mus.empty() ? "" : ",") + fmt("%.2e", st.mu_ser);
            for (real mr : st.mu_retries)
                mus += fmt("^%.2e", mr); // ^ = line-search retry of the same step
        }
        if (etas.empty())
            etas = "-";
        if (mus.empty())
            mus = "-";
        out_line(fmt("GMRES_STATS field=%s eps=%g N=%d stage_eps=%g%s status=%s newton_its=%d "
                     "steps=%d its_max=%d its_median=%.1f its_total=%d rel_max=%.1e "
                     "linear_failures=%d forcing=%s etas=%s psitc=%s mus=%s",
                     field.c_str(), eps, N, s.eps, s.final_attempt ? "(final)" : "",
                     sl::to_string(s.newton.status), s.newton.its, ls.steps, ls.its_max,
                     ls.its_median, ls.its_total, ls.rel_max, ls.failures,
                     sl::to_string(s.newton.forcing), etas.c_str(),
                     s.newton.psitc ? "on" : "off", mus.c_str()));
    }
}

/// c2, c3 of the last evaluate_metrics on planes 0, 1 and N (host copies).
struct CrossPlanes {
    std::vector<real> c2[3], c3[3]; ///< index 0: plane 0, 1: plane 1, 2: plane N
};

inline CrossPlanes grab_cross_planes(const sl::InletSlabGrid& g, const sl::SlabMetricsWorkspace& ws) {
    CrossPlanes cp;
    const std::size_t n2 = g.plane_size();
    const int planes[3] = {0, 1, g.n};
    for (int k = 0; k < 3; ++k) {
        const std::size_t off = static_cast<std::size_t>(planes[k]) * n2;
        cp.c2[k] = download(ws.cross_component(1).data() + off, n2);
        cp.c3[k] = download(ws.cross_component(2).data() + off, n2);
    }
    return cp;
}

/// The prototype's `report()` EXTRA quantities for one label set: outlet defect, inlet oblique
/// defects on plane 0 (one-sided) and plane 1, plane 1 vs vD(j = 1); all / v_rms (ctx.v_rms).
struct ExtraDefects {
    real out = std::nan(""), in0 = std::nan(""), in1 = std::nan(""), in1v = std::nan("");
};

inline ExtraDefects extra_defects(const CrossPlanes& cp, const std::vector<real>& vp2,
                                  const std::vector<real>& vp3, const std::vector<real>& vD2_1,
                                  const std::vector<real>& vD3_1, real v_rms) {
    auto rmsdef = [&](const std::vector<real>& a2, const std::vector<real>& a3,
                      const std::vector<real>& b2, const std::vector<real>& b3) {
        real s = 0.0;
        for (std::size_t i = 0; i < a2.size(); ++i) {
            const real d2 = a2[i] - b2[i], d3 = a3[i] - b3[i];
            s += d2 * d2 + d3 * d3;
        }
        return std::sqrt(s / static_cast<real>(a2.size())) / v_rms;
    };
    ExtraDefects e;
    e.out = rmsdef(cp.c2[2], cp.c3[2], vp2, vp3);
    e.in0 = rmsdef(cp.c2[0], cp.c3[0], vp2, vp3);
    e.in1 = rmsdef(cp.c2[1], cp.c3[1], vp2, vp3);
    e.in1v = rmsdef(cp.c2[1], cp.c3[1], vD2_1, vD3_1);
    return e;
}

// ================================================================================================
// Solve + report (shared by --proto and --production)
// ================================================================================================

struct SolveContext {
    std::string field;
    real eps = 0.0;
    sl::InletSlabGrid grid;
    sl::StageInputProvider provider;
};

struct SolveOutcome {
    std::string status = "not_run"; ///< continuation status, or missing_stage_input / stage
                                    ///< build failure
    bool solved = false;            ///< the continuation returned (x is a state at eps)
    sl::SlabContinuationReport rep;
    DeviceBuffer<real> x;
    double seconds = 0.0;
    std::size_t solver_bytes = 0;
};

inline sl::SlabCoarseAssembly coarse_assembly_of(const DriverConfig& c) {
    return c.coarse_assembly == "colored" ? sl::SlabCoarseAssembly::colored
                                          : sl::SlabCoarseAssembly::direct;
}
inline sl::SlabCoarseFactor coarse_factor_of(const DriverConfig& c) {
    return c.coarse_factor == "banded" ? sl::SlabCoarseFactor::banded
                                       : sl::SlabCoarseFactor::dense;
}

inline sl::SlabNewtonConfig newton_config(const DriverConfig& c) {
    sl::SlabNewtonConfig n;
    n.tol = c.newton_tol;
    n.max_iterations = c.max_newton;
    n.gmres.tol = c.lin_tol;
    n.gmres.restart = c.restart;
    n.gmres.max_iterations = c.max_inner;
    n.gmres.stagnation_factor = c.gmres_stagnation_factor;
    n.prec_name = "P-A";
    n.forcing = c.forcing == "fixed" ? sl::SlabForcing::fixed : sl::SlabForcing::ew;
    n.ew.eta_max = c.ew_eta_max;
    n.ew.eta0 = c.ew_eta0;
    n.psitc.enabled = c.psitc == "on";
    n.psitc.mu0 = c.psitc_mu0;
    n.psitc.mu_max = c.psitc_mu_max;
    if (c.coarse != "off") {
        n.coarse = c.coarse == "add" ? sl::SlabCoarseMode::add : sl::SlabCoarseMode::mult;
        n.prec_name = fmt("P-A+CC(%s,%d)", c.coarse.c_str(), c.coarse_profiles);
    }
    return n;
}

/// Runs the continuation; stage-input exceptions (missing stage, stage build failure) become
/// distinct statuses (rethrown as nothing: the outcome records them).
inline void run_solve(CudaContext& ctx, const DriverConfig& c, SolveContext& sc,
                      sl::SlabNewtonKrylov& nk, SolveOutcome& out, json& J) {
    const sl::SlabNewtonConfig ncfg = newton_config(c);
    sl::SlabContinuationConfig ccfg;
    ccfg.max_bisections = c.bisect;
    ccfg.field = sc.field;
    ccfg.cand = "i1o4";
    out.x.resize(sc.grid.unknown_size());
    out_line(fmt("SOLVER N=%d eps=%g cand=i1o4 newton_tol=%.1e max_newton=%d lin_tol=%.1e "
                 "restart=%d max_inner=%d bisect=%d prec=P-A ladder=(0.25,0.5,1) stage_ok=%.0e "
                 "forcing=%s ew_gamma=%g ew_alpha=%g ew_eta0=%g ew_eta_max=%g ew_eta_min=%.1e "
                 "psitc=%s psitc_mu0=%g psitc_mu_max=%g psitc_norm=merit psitc_retries=%d "
                 "psitc_retry_factor=%g stagnation_window=%d stagnation_factor=%g "
                 "gmres_stagnation_factor=%g coarse=%s coarse_profiles=%d coarse_assembly=%s "
                 "coarse_factor=%s",
                 sc.grid.n, sc.eps, ncfg.tol, ncfg.max_iterations, ncfg.gmres.tol,
                 ncfg.gmres.restart, ncfg.gmres.max_iterations, ccfg.max_bisections,
                 ccfg.stage_ok, sl::to_string(ncfg.forcing), ncfg.ew.gamma, ncfg.ew.alpha,
                 ncfg.ew.eta0, ncfg.ew.eta_max, ncfg.gmres.tol, ncfg.psitc.enabled ? "on" : "off",
                 ncfg.psitc.mu0, ncfg.psitc.mu_max, ncfg.psitc.max_retries,
                 ncfg.psitc.retry_factor, ncfg.stagnation_window, ncfg.stagnation_factor,
                 ncfg.gmres.stagnation_factor, sl::to_string(ncfg.coarse), c.coarse_profiles,
                 c.coarse_assembly.c_str(), c.coarse_factor.c_str()));
    if (ncfg.coarse != sl::SlabCoarseMode::off)
        out_line(fmt("SOLVER_COARSE mode=%s profiles=%d K=%d assembly=%s factor=%s (SF-33 N7c "
                     "probe: Galerkin coarse correction on the x1-constant%s column subspace, "
                     "rebuilt with P-A at every factor)",
                     sl::to_string(ncfg.coarse), c.coarse_profiles,
                     2 * c.coarse_profiles * sc.grid.n * sc.grid.n, c.coarse_assembly.c_str(),
                     c.coarse_factor.c_str(), c.coarse_profiles == 2 ? " + x1-linear" : ""));
    J["solver_config_coarse"] = {{"coarse", sl::to_string(ncfg.coarse)},
                                 {"coarse_profiles", c.coarse_profiles},
                                 {"coarse_assembly", c.coarse_assembly},
                                 {"coarse_factor", c.coarse_factor}};
    J["solver_config"] = {{"forcing", sl::to_string(ncfg.forcing)},
                          {"ew_gamma", ncfg.ew.gamma},
                          {"ew_alpha", ncfg.ew.alpha},
                          {"ew_eta0", ncfg.ew.eta0},
                          {"ew_eta_max", ncfg.ew.eta_max},
                          {"ew_eta_min", ncfg.gmres.tol},
                          {"lin_tol", ncfg.gmres.tol},
                          {"restart", ncfg.gmres.restart},
                          {"max_inner", ncfg.gmres.max_iterations},
                          {"newton_tol", ncfg.tol},
                          {"max_newton", ncfg.max_iterations},
                          {"psitc", ncfg.psitc.enabled ? "on" : "off"},
                          {"psitc_mu0", ncfg.psitc.mu0},
                          {"psitc_mu_max", ncfg.psitc.mu_max},
                          {"psitc_norm", "merit"},
                          {"psitc_max_retries", ncfg.psitc.max_retries},
                          {"psitc_retry_factor", ncfg.psitc.retry_factor},
                          {"stagnation_window", ncfg.stagnation_window},
                          {"stagnation_factor", ncfg.stagnation_factor},
                          {"gmres_stagnation_factor", ncfg.gmres.stagnation_factor},
                          {"coarse", sl::to_string(ncfg.coarse)},
                          {"coarse_profiles", c.coarse_profiles},
                          {"coarse_assembly", c.coarse_assembly},
                          {"coarse_factor", c.coarse_factor},
                          {"bisect", ccfg.max_bisections}};
    const auto t0 = std::chrono::steady_clock::now();
    try {
        out.rep = nk.solve_with_continuation(ctx, sc.eps, sc.provider, mspan(out.x), ccfg, ncfg,
                                             sl::slab_stdout_logger);
        out.status = sl::to_string(out.rep.status);
        out.solved = true;
    } catch (const sl::MissingStageInput& e) {
        out.status = sl::MissingStageInput::status();
        out_line(fmt("ABORT missing_stage_input amp=%g: %s", e.amplitude(), e.what()));
        J["abort"] = {{"status", out.status}, {"amplitude", e.amplitude()}, {"what", e.what()}};
    } catch (const StageBuildFailure& e) {
        out.status = e.status();
        out_line(fmt("ABORT %s amp=%g: %s", e.status().c_str(), e.amplitude(), e.what()));
        J["abort"] = {{"status", out.status}, {"amplitude", e.amplitude()}, {"what", e.what()}};
    }
    ctx.synchronize();
    out.seconds = seconds_since(t0);
    out.solver_bytes = nk.allocated_bytes();
    if (out.solved) {
        J["continuation"] = continuation_json(out.rep);
        print_gmres_stats(sc.field, sc.eps, sc.grid.n, out.rep);
    }
}

struct ReportOutcome {
    std::vector<real> U1, U2; ///< host periodic parts of the solved labels, planes 0..N
    std::size_t metrics_bytes = 0;
};

/**
 * CASE (cand=i1o4) + ceiling CASE (cand=oracle_fd4, iff ref.has_psi_or) + EXTRA + HISTORY lines of
 * the solved state, in the prototype's formats (candidate_i.report).
 */
inline ReportOutcome report_solution(CudaContext& ctx, const SolveContext& sc,
                                     const sl::SlabStageInputs& target,
                                     const sl::SlabReferenceData& ref, SolveOutcome& so,
                                     json& J) {
    ReportOutcome ro;
    const sl::InletSlabGrid& g = sc.grid;
    const int N = g.n;
    const std::size_t n2 = g.plane_size();
    DeviceBuffer<real> U1(g.full_size()), U2(g.full_size());
    sl::assemble_full_planes(ctx, g, cspan(so.x), target, mspan(U1), mspan(U2));
    ctx.synchronize();
    sl::SlabMetricsWorkspace mws;
    mws.prepare(g);
    ro.metrics_bytes = mws.allocated_bytes();
    const sl::SlabNewtonReport& fin = so.rep.final_newton;
    const sl::SlabMetrics m = sl::evaluate_metrics(ctx, g, cspan(U1), cspan(U2), ref, mws);
    const CrossPlanes cp_c = grab_cross_planes(g, mws);
    sl::CaseLineExtras ex;
    ex.r_F = fin.r_F;
    ex.its = fin.its;
    ex.t = so.rep.seconds;
    out_line(sl::format_case_line(sc.field, sc.eps, N, "i1o4", m, ex));
    J["metrics"]["i1o4"] = metrics_json(m);
    J["case_line"] = sl::format_case_line(sc.field, sc.eps, N, "i1o4", m, ex);

    sl::SlabMetrics mo;
    CrossPlanes cp_o;
    if (ref.has_psi_or) {
        DeviceBuffer<real> O1(g.full_size()), O2(g.full_size());
        sl::labels_to_periodic_parts(ctx, g, cspan(ref.psi_or[0]), cspan(ref.psi_or[1]),
                                     mspan(O1), mspan(O2));
        mo = sl::evaluate_metrics(ctx, g, cspan(O1), cspan(O2), ref, mws);
        cp_o = grab_cross_planes(g, mws);
        out_line(sl::format_case_line(sc.field, sc.eps, N, "oracle_fd4", mo));
        J["metrics"]["oracle_fd4"] = metrics_json(mo);
        J["ceiling_line"] = sl::format_case_line(sc.field, sc.eps, N, "oracle_fd4", mo);
    }

    // EXTRA (candidate_i.report): defects / ctx.v_rms (= target.v_rms)
    const std::vector<real> vp2 = download(target.vperp_in[0].data(), n2);
    const std::vector<real> vp3 = download(target.vperp_in[1].data(), n2);
    const std::vector<real> vD2_1 = download(ref.vD[1].data() + n2, n2);
    const std::vector<real> vD3_1 = download(ref.vD[2].data() + n2, n2);
    const ExtraDefects ec = extra_defects(cp_c, vp2, vp3, vD2_1, vD3_1, target.v_rms);
    ExtraDefects eo;
    if (ref.has_psi_or)
        eo = extra_defects(cp_o, vp2, vp3, vD2_1, vD3_1, target.v_rms);
    out_line(fmt("EXTRA field=%s eps=%g N=%d cand=%s status=%s r_F=%.3e r_out=%.3e | outlet "
                 "c_perp-vperp_in: cand=%.3e oracle_fd=%.3e | inlet oblique defect (c x e1 - "
                 "vperp_in x e1)/v_rms: j=0(one-sided) cand=%.3e oracle_fd=%.3e; j=1 cand=%.3e "
                 "oracle_fd=%.3e; j=1 vs vD(j=1) cand=%.3e oracle_fd=%.3e | min|c|/v_rms=%.4f "
                 "min|vD|/v_rms=%.4f e_i1=%.3e e_i2=%.3e | t=%.1fs | a_psi1=%.3e a_psi2=%.3e (FD "
                 "order %d)",
                 sc.field.c_str(), sc.eps, N, "i1o4", sl::to_string(fin.status), fin.r_F,
                 fin.r_out, ec.out, eo.out, ec.in0, eo.in0, ec.in1, eo.in1, ec.in1v, eo.in1v,
                 m.min_c / m.v_rms, m.vD_min / m.v_rms, m.e_i1, m.e_i2, so.rep.seconds, m.a_psi1,
                 m.a_psi2, 4));
    J["extra"] = {{"outlet_cand", jnum(ec.out)},     {"outlet_oracle", jnum(eo.out)},
                  {"inlet_j0_cand", jnum(ec.in0)},   {"inlet_j0_oracle", jnum(eo.in0)},
                  {"inlet_j1_cand", jnum(ec.in1)},   {"inlet_j1_oracle", jnum(eo.in1)},
                  {"inlet_j1_vD_cand", jnum(ec.in1v)}, {"inlet_j1_vD_oracle", jnum(eo.in1v)},
                  {"min_c_over_vrms", jnum(m.min_c / m.v_rms)},
                  {"min_vD_over_vrms", jnum(m.vD_min / m.v_rms)}};

    std::string hF, hO, hFull, hOFull;
    for (std::size_t i = 0; i < fin.hist_r_F.size(); ++i) {
        hF += fmt("%s%.2e", i ? " " : "", fin.hist_r_F[i]);
        hO += fmt("%s%.2e", i ? " " : "", fin.hist_r_out[i]);
        hFull += fmt("%s%.17e", i ? " " : "", fin.hist_r_F[i]);
        hOFull += fmt("%s%.17e", i ? " " : "", fin.hist_r_out[i]);
    }
    out_line(fmt("HISTORY field=%s eps=%g N=%d cand=i1o4 r_F: %s", sc.field.c_str(), sc.eps, N,
                 hF.c_str()));
    out_line(fmt("HISTORY field=%s eps=%g N=%d cand=i1o4 r_out: %s", sc.field.c_str(), sc.eps, N,
                 hO.c_str()));
    // full-precision histories (compare_proto.py: r_F history agreement vs the prototype's hist)
    out_line(fmt("HISTORY_FULL field=%s eps=%g N=%d cand=i1o4 r_F: %s", sc.field.c_str(), sc.eps,
                 N, hFull.c_str()));
    out_line(fmt("HISTORY_FULL field=%s eps=%g N=%d cand=i1o4 r_out: %s", sc.field.c_str(),
                 sc.eps, N, hOFull.c_str()));

    ro.U1 = download(U1.data(), U1.size());
    ro.U2 = download(U2.data(), U2.size());
    return ro;
}

/// --save-solution: u1.npy, u2.npy ((N+1, N, N), periodic parts on planes 0..N) + solution.json
/// in the layout of export_proto.py --solutions (readable by sl::load_solution).
inline void save_solution(const std::string& dir, const SolveContext& sc, const SolveOutcome& so,
                          const ReportOutcome& ro, const json& J) {
    make_dir(dir);
    const std::size_t N = static_cast<std::size_t>(sc.grid.n);
    const std::vector<std::size_t> shape = {N + 1, N, N};
    sl::write_npy(dir + "/u1.npy", shape, ro.U1.data(), ro.U1.size());
    sl::write_npy(dir + "/u2.npy", shape, ro.U2.data(), ro.U2.size());
    const sl::SlabNewtonReport& fin = so.rep.final_newton;
    json s;
    s["field"] = sc.field;
    s["eps"] = sc.eps;
    s["N"] = sc.grid.n;
    s["variant"] = "i1";
    s["order"] = 4;
    s["cand"] = "i1o4";
    s["status"] = sl::to_string(fin.status);
    s["continuation_status"] = so.status;
    s["its"] = fin.its;
    s["r_F"] = jnum(fin.r_F);
    s["r_out"] = jnum(fin.r_out);
    s["t"] = so.rep.seconds;
    s["path"] = so.rep.path;
    json hist = json::array();
    for (std::size_t i = 0; i < fin.hist_r_F.size(); ++i)
        hist.push_back({jnum(fin.hist_r_F[i]), jnum(fin.hist_r_out[i])});
    s["hist"] = hist;
    s["metrics_json"] = J.contains("metrics") ? J["metrics"] : json::object();
    s["source"] = "inlet_slab (SF-33 N5 GPU driver)";
    std::ofstream f(dir + "/solution.json");
    f << s.dump(1) << "\n";
    out_line("SAVED " + dir + " (u1.npy, u2.npy, solution.json)");
}

inline void write_summary(const std::string& path, const json& J) {
    if (path.empty())
        return;
    const std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos)
        make_dir(path.substr(0, slash));
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("cannot write summary " + path);
    f << J.dump(1) << "\n";
    out_line("SUMMARY_JSON " + path);
}

inline void print_timing_memory(const PhaseLog& pl, std::size_t ws_bytes, double total_s,
                                json& J) {
    std::string t = "TIMING";
    for (const auto& kv : pl.by_name())
        t += fmt(" %s=%.3fs", kv.first.c_str(), kv.second);
    t += fmt(" total=%.3fs", total_s);
    out_line(t);
    out_line(fmt("MEMORY peak_device_bytes=%zu free_min=%zu total_device=%zu "
                 "workspaces_bytes=%zu (peak/free: cudaMemGetInfo polls before/after every phase, "
                 "device-wide)",
                 pl.peak_used(), pl.free_min(), pl.total(), ws_bytes));
    J["timing"] = pl.by_name();
    J["timing"]["total"] = total_s;
    J["phases"] = pl.phases();
    J["memory"] = {{"peak_device_bytes", pl.peak_used()},
                   {"free_min", pl.free_min()},
                   {"total_device", pl.total()},
                   {"workspaces_bytes", ws_bytes}};
}

inline int finish(const std::string& status, json& J, const DriverConfig& c) {
    const int code = exit_code_of(status);
    J["status"] = status;
    J["exit_code"] = code;
    write_summary(c.summary_path, J);
    out_line("STATUS " + status);
    return code;
}

// ================================================================================================
// --proto
// ================================================================================================

inline int run_proto(CudaContext& ctx, const DriverConfig& c) {
    const auto t_all = std::chrono::steady_clock::now();
    json J;
    J["mode"] = "proto";
    J["command"] = c.command_line;
    J["case_dir"] = c.case_dir;
    PhaseLog pl;

    pl.begin("load");
    const sl::ProtoCaseMeta meta0 = sl::read_proto_case_meta(c.case_dir);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(meta0.N);
    sl::ProtoCase pc = sl::load_proto_case(ctx, c.case_dir, grid);
    sl::ProtoStageProvider prov(ctx, c.case_dir, grid, pc.meta.field);
    pl.end();
    const real eps = c.has_eps_target ? c.eps_target : pc.meta.eps;
    out_line(fmt("CASE_DIR %s field=%s eps=%g N=%d nphi=%d Q0=%.15g inlet_vmin=%.6e v_rms=%.15g "
                 "cache_hit=%d stage_amplitudes=%zu target_eps=%g",
                 c.case_dir.c_str(), pc.meta.field.c_str(), pc.meta.eps, pc.meta.N, pc.meta.nphi,
                 pc.meta.Q0, pc.meta.inlet_vmin, pc.meta.v_rms, pc.meta.cache_hit ? 1 : 0,
                 pc.meta.stage_amplitudes.size(), eps));
    out_line("SETUP grad_lnk=analytic (prototype export: complex-step gradient of the analytic "
             "ln k at the vertices)");
    J["field"] = pc.meta.field;
    J["eps"] = eps;
    J["N"] = grid.n;

    SolveContext sc;
    sc.field = pc.meta.field;
    sc.eps = eps;
    sc.grid = grid;
    sc.provider = prov.callback();

    sl::SlabNewtonKrylov nk;
    pl.begin("prepare");
    nk.prepare(ctx, grid, c.restart, c.max_newton, c.max_inner);
    if (c.coarse != "off")
        nk.prepare_coarse(ctx, c.coarse_profiles, coarse_assembly_of(c), coarse_factor_of(c));
    pl.end();
    SolveOutcome so;
    pl.begin("solve");
    run_solve(ctx, c, sc, nk, so, J);
    pl.end();
    std::size_t ws_bytes = so.solver_bytes;

    if (so.solved) {
        if (std::fabs(eps - pc.meta.eps) > 1e-12) {
            out_line(fmt("NOTE target eps=%g differs from the case eps=%g: no reference data at "
                         "the target, CASE / EXTRA / FIELDDIFF skipped",
                         eps, pc.meta.eps));
        } else {
            pl.begin("metrics");
            const sl::SlabStageInputs& target = prov(eps);
            const ReportOutcome ro = report_solution(ctx, sc, target, pc.ref, so, J);
            pl.end();
            ws_bytes += ro.metrics_bytes;
            if (!c.solution_dir.empty()) {
                const sl::ProtoSolution ps = sl::load_solution(c.solution_dir);
                if (ps.N != grid.n)
                    throw std::runtime_error("--solution: N differs from the case");
                const real d1 = sl::max_relative_difference(ro.U1, ps.u1);
                const real d2 = sl::max_relative_difference(ro.U2, ps.u2);
                // absolute differences and the joint normalization max|u_proto| over both fields
                // (a label whose periodic part is roundoff-sized, e.g. control2d u2 ~ 1e-13, has a
                // meaningless per-field relative difference)
                auto maxabs_diff = [](const std::vector<real>& a, const std::vector<real>& b) {
                    real m = 0.0;
                    for (std::size_t i = 0; i < a.size(); ++i)
                        m = std::max(m, std::fabs(a[i] - b[i]));
                    return m;
                };
                auto maxabs = [](const std::vector<real>& a) {
                    real m = 0.0;
                    for (real v : a)
                        m = std::max(m, std::fabs(v));
                    return m;
                };
                const real a1 = maxabs_diff(ro.U1, ps.u1), a2 = maxabs_diff(ro.U2, ps.u2);
                const real s1 = maxabs(ps.u1), s2 = maxabs(ps.u2);
                const real joint = std::max(a1, a2) / std::max(s1, s2);
                out_line(fmt("FIELDDIFF max_rel_diff_u1=%.3e max_rel_diff_u2=%.3e "
                             "max_rel_diff_joint=%.3e max_abs_diff_u1=%.3e max_abs_diff_u2=%.3e "
                             "max_abs_u_proto=(%.3e,%.3e) (rel: max|u_gpu - u_proto| / "
                             "max|u_proto| over planes 0..N; joint: max abs diff / max over both "
                             "fields; prototype %s status=%s its=%d path=%s r_F=%.3e)",
                             d1, d2, joint, a1, a2, s1, s2, c.solution_dir.c_str(),
                             ps.status.c_str(), ps.its, ps.path.c_str(), ps.r_F));
                J["fielddiff"] = {{"max_rel_diff_u1", jnum(d1)},
                                  {"max_rel_diff_u2", jnum(d2)},
                                  {"max_rel_diff_joint", jnum(joint)},
                                  {"max_abs_diff_u1", jnum(a1)},
                                  {"max_abs_diff_u2", jnum(a2)},
                                  {"max_abs_u_proto", {jnum(s1), jnum(s2)}},
                                  {"solution_dir", c.solution_dir},
                                  {"proto_status", ps.status},
                                  {"proto_path", ps.path},
                                  {"proto_its", ps.its},
                                  {"proto_r_F", jnum(ps.r_F)}};
            }
            if (!c.save_solution_dir.empty())
                save_solution(c.save_solution_dir, sc, so, ro, J);
        }
    }
    print_timing_memory(pl, ws_bytes, seconds_since(t_all), J);
    return finish(so.status, J, c);
}

// ================================================================================================
// --production
// ================================================================================================

/// StageInputProvider over N3's build_production_stage: one SF-19 solve per amplitude, cached.
/// Non-target stages release their oracle-only data (inlet-label device tables, host splines).
class ProductionProvider {
  public:
    ProductionProvider(CudaContext& ctx, const sl::InletSlabGrid& g, const sl::SlabFieldSource& src,
                       real eps_target, const sl::ProductionStageOptions& opt)
        : ctx_(&ctx), grid_(g), src_(&src), eps_(eps_target), opt_(opt) {}

    sl::ProductionStage& stage(real amp) {
        const std::string key = fmt("%.17g", amp);
        auto it = cache_.find(key);
        if (it != cache_.end())
            return *it->second;
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<sl::ProductionStage> st(new sl::ProductionStage(
            sl::build_production_stage(*ctx_, grid_, *src_, amp, opt_)));
        const double s = seconds_since(t0);
        seconds_ += s;
        out_block("SETUP ", st->report.summary());
        out_line(fmt("SETUP stage eps=%g built in %.3fs (grad_lnk=%s)", amp, s,
                     src_->kind() == sl::SlabFieldKind::analytic
                         ? "analytic (closure_fields.hpp, hand-derived gradient)"
                         : "spectral (band-limited cell samples, cuFFT)"));
        json r;
        r["eps"] = amp;
        r["status"] = sl::to_string(st->report.status);
        r["seconds"] = s;
        r["mg_levels"] = st->report.mg_levels;
        r["G"] = {jnum(st->report.darcy.G[0]), jnum(st->report.darcy.G[1]),
                  jnum(st->report.darcy.G[2])};
        json pcg = json::array();
        for (int d = 0; d < 3; ++d) {
            const auto& cr = st->report.darcy.corrector_results[d];
            pcg.push_back({{"converged", cr.converged},
                           {"its", cr.iterations},
                           {"rel_residual", jnum(cr.relative_projected_residual)}});
        }
        r["pcg"] = pcg;
        r["inlet_vmin"] = jnum(st->report.inlet_vmin);
        r["inlet_mean"] = jnum(st->report.inlet_mean);
        r["Q0"] = jnum(st->report.Q0);
        r["v_rms"] = jnum(st->report.v_rms);
        r["v1_diff_rms_rel"] = jnum(st->report.v1_diff_rms_rel);
        r["v1_diff_max_rel"] = jnum(st->report.v1_diff_max_rel);
        if (st->report.has_sf18) {
            r["sf18"] = {{"raw_mean", jnum(st->report.sf18.raw_mean)},
                         {"raw_variance", jnum(st->report.sf18.raw_variance)},
                         {"applied_scale", jnum(st->report.sf18.applied_scale)},
                         {"final_variance", jnum(st->report.sf18.final_variance)},
                         {"active_modes", st->report.sf18.active_mode_count}};
        }
        reports_.push_back(r);
        if (!st->ok()) {
            const std::string status = sl::to_string(st->report.status);
            throw StageBuildFailure(status, amp,
                                    fmt("production stage eps=%g: %s (inlet_vmin=%.6e)", amp,
                                        status.c_str(), st->report.inlet_vmin));
        }
        if (std::fabs(amp - eps_) > 1e-12) {
            // the solver needs only the stage inputs; the oracle data is kept for the target only
            st->labels = sl::InletLabels();
            std::vector<real>().swap(st->potential_coefficients);
            std::vector<real>().swap(st->logk_coefficients);
        }
        sl::ProductionStage& ref = *st;
        cache_[key] = std::move(st);
        return ref;
    }

    const sl::SlabStageInputs& operator()(real amp) { return stage(amp).inputs; }
    double seconds() const { return seconds_; }
    const json& reports() const { return reports_; }
    std::size_t stage_device_bytes() const {
        std::size_t b = 0;
        for (const auto& kv : cache_) {
            const sl::ProductionStage& s = *kv.second;
            b += s.inputs.q.capacity() + s.inputs.lnk.capacity();
            for (int d = 0; d < 3; ++d)
                b += s.inputs.grad_lnk[d].capacity() + s.reference.vD[d].capacity();
            for (int i = 0; i < 2; ++i)
                b += s.inputs.u0[i].capacity() + s.inputs.vperp_in[i].capacity() +
                     s.reference.psi_or[i].capacity();
        }
        std::size_t bytes = b * sizeof(real);
        for (const auto& kv : cache_)
            bytes += kv.second->labels.device_bytes();
        return bytes;
    }

  private:
    CudaContext* ctx_;
    sl::InletSlabGrid grid_;
    const sl::SlabFieldSource* src_;
    real eps_;
    sl::ProductionStageOptions opt_;
    std::map<std::string, std::unique_ptr<sl::ProductionStage>> cache_;
    double seconds_ = 0.0;
    json reports_ = json::array();
};

struct OracleEntry {
    double hmax = 0.0, tol = 0.0;
    sl::SlabOracleResult res;
    std::vector<real> psi1, psi2; ///< host copies (ladder label differences)
};

inline OracleEntry run_oracle_entry(CudaContext& ctx, sl::ProductionStage& st, double hmax,
                                    double tol, const DriverConfig& c, DeviceBuffer<real>& p1,
                                    DeviceBuffer<real>& p2) {
    OracleEntry e;
    e.hmax = hmax;
    e.tol = tol;
    sl::SlabOracleOptions opt;
    opt.tol = tol;
    opt.h_max = hmax;
    opt.threads = c.threads;
    opt.max_roundtrip = c.oracle_max_roundtrip;
    const sl::SlabSplineDirectionField fld = st.direction_field();
    out_line(fmt("ORACLE_RUN hmax=%.6e (h/%.6g) tol=%.1e threads=%d max_roundtrip_gate=%.1e",
                 hmax, st.grid.h / hmax, tol, c.threads, c.oracle_max_roundtrip));
    e.res = sl::compute_oracle(ctx, st.grid, fld, st.labels, opt, mspan(p1), mspan(p2));
    for (const sl::SlabOraclePlane& P : e.res.planes) {
        if (P.j == 0)
            continue;
        out_line(fmt("ORACLE hmax=%.6e tol=%.1e plane=%d roundtrip=%.3e nfev=%lld nfev_rt=%lld "
                     "nonok=%lld backflow_enc=%lld%s",
                     hmax, tol, P.j, P.roundtrip, P.nfev, P.nfev_rt, P.back_non_ok + P.fwd_non_ok,
                     P.backflow_encounters, P.flagged ? " FLAGGED" : ""));
    }
    out_line(fmt("ORACLE_SUMMARY hmax=%.6e tol=%.1e status=%s max_roundtrip=%.3e non_ok=%lld "
                 "t_trace=%.2fs t_labels=%.3fs",
                 hmax, tol, sl::to_string(e.res.status), e.res.max_roundtrip, e.res.non_ok,
                 e.res.seconds_trace, e.res.seconds_labels));
    e.psi1 = download(p1.data(), p1.size());
    e.psi2 = download(p2.data(), p2.size());
    return e;
}

inline json oracle_json(const OracleEntry& e) {
    json j;
    j["hmax"] = e.hmax;
    j["tol"] = e.tol;
    j["status"] = sl::to_string(e.res.status);
    j["max_roundtrip"] = jnum(e.res.max_roundtrip);
    j["non_ok"] = e.res.non_ok;
    j["t_trace"] = e.res.seconds_trace;
    j["t_labels"] = e.res.seconds_labels;
    json planes = json::array();
    for (const auto& P : e.res.planes) {
        if (P.j == 0)
            continue;
        planes.push_back({{"j", P.j},
                          {"roundtrip", jnum(P.roundtrip)},
                          {"nfev", P.nfev},
                          {"nfev_rt", P.nfev_rt},
                          {"non_ok", P.back_non_ok + P.fwd_non_ok},
                          {"flagged", P.flagged}});
    }
    j["planes"] = planes;
    return j;
}

inline int run_production(CudaContext& ctx, const DriverConfig& c) {
    const auto t_all = std::chrono::steady_clock::now();
    json J;
    J["mode"] = "production";
    J["command"] = c.command_line;
    PhaseLog pl;
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(c.n);

    pl.begin("field");
    sl::SlabFieldSource src;
    if (c.analytic.empty()) {
        sl::ProductionFieldSpec spec;
        spec.N = c.n;
        spec.sigma2 = c.sigma2;
        spec.ell = c.ell;
        spec.seed = c.seed;
        spec.normalize_variance = true;
        src = sl::SlabFieldSource::gaussian(ctx, spec);
        out_line(fmt("FIELD gaussian (SF-18) N=%d sigma2=%g ell=%g seed=%llu normalize_variance=1 "
                     "applied_scale=%.12e raw_variance=%.12e final_variance=%.12e "
                     "active_modes=%zu | stage field k = exp(eps Y), eps = %g",
                     c.n, c.sigma2, c.ell, c.seed, src.sf18().applied_scale,
                     src.sf18().raw_variance, src.sf18().final_variance,
                     src.sf18().active_mode_count, c.eps));
        J["field_spec"] = {{"kind", "gaussian"},
                           {"sigma2", c.sigma2},
                           {"ell", c.ell},
                           {"seed", c.seed},
                           {"applied_scale", jnum(src.sf18().applied_scale)},
                           {"raw_variance", jnum(src.sf18().raw_variance)},
                           {"final_variance", jnum(src.sf18().final_variance)}};
    } else {
        src = sl::SlabFieldSource::analytic(c.analytic, c.n);
        out_line(fmt("FIELD analytic %s (closure_fields.hpp, unit amplitude) N=%d | stage field "
                     "k = exp(eps Y), eps = %g",
                     c.analytic.c_str(), c.n, c.eps));
        J["field_spec"] = {{"kind", "analytic"}, {"name", c.analytic}};
    }
    pl.end();
    J["field"] = src.name();
    J["eps"] = c.eps;
    J["N"] = c.n;

    sl::ProductionStageOptions sopt;
    sopt.pcg_rtol = c.pcg_rtol;
    ProductionProvider prov(ctx, grid, src, c.eps, sopt);
    SolveContext sc;
    sc.field = src.name();
    sc.eps = c.eps;
    sc.grid = grid;
    sc.provider = [&prov](real a) -> const sl::SlabStageInputs& { return prov(a); };

    sl::SlabNewtonKrylov nk;
    pl.begin("prepare");
    nk.prepare(ctx, grid, c.restart, c.max_newton, c.max_inner);
    if (c.coarse != "off")
        nk.prepare_coarse(ctx, c.coarse_profiles, coarse_assembly_of(c), coarse_factor_of(c));
    pl.end();
    SolveOutcome so;
    pl.begin("solve");
    run_solve(ctx, c, sc, nk, so, J);
    pl.end();
    J["stages_setup"] = prov.reports();
    std::size_t ws_bytes = so.solver_bytes;
    std::string oracle_status = "not_run";

    if (so.solved) {
        sl::ProductionStage& tst = prov.stage(c.eps);
        sl::SlabReferenceData& ref = tst.reference;
        if (!c.no_oracle) {
            pl.begin("oracle");
            for (auto& b : ref.psi_or)
                b.resize(grid.full_size());
            ref.has_psi_or = true;
            DeviceBuffer<real> q1(grid.full_size()), q2(grid.full_size());
            std::vector<OracleEntry> entries;
            const double h = grid.h;
            const double hmax0 = h / static_cast<double>(c.oracle_hmax_div);
            out_line(fmt("ORACLE_CONFIG primary hmax=h/%d=%.6e tol=%.1e (SF-30 default h gives "
                         "step-limited round trips; orchestrator decision: h/8)%s",
                         c.oracle_hmax_div, hmax0, c.oracle_tol,
                         c.oracle_ladder ? " | ladder: + (h/16, tol) + (h/16, 1e-10)" : ""));
            entries.push_back(run_oracle_entry(ctx, tst, hmax0, c.oracle_tol, c, ref.psi_or[0],
                                               ref.psi_or[1]));
            if (c.oracle_ladder) {
                if (c.oracle_hmax_div != 16)
                    entries.push_back(run_oracle_entry(ctx, tst, h / 16.0, c.oracle_tol, c, q1, q2));
                if (!(c.oracle_hmax_div == 16 && c.oracle_tol == 1e-10))
                    entries.push_back(run_oracle_entry(ctx, tst, h / 16.0, 1e-10, c, q1, q2));
                for (std::size_t k = 1; k < entries.size(); ++k) {
                    real d1 = 0.0, d2 = 0.0;
                    for (std::size_t i = 0; i < entries[0].psi1.size(); ++i) {
                        d1 = std::max(d1, std::fabs(entries[k].psi1[i] - entries[0].psi1[i]));
                        d2 = std::max(d2, std::fabs(entries[k].psi2[i] - entries[0].psi2[i]));
                    }
                    out_line(fmt("ORACLE_LABELDIFF (hmax=%.6e tol=%.1e) vs primary: max|dpsi1|=%.3e "
                                 "max|dpsi2|=%.3e",
                                 entries[k].hmax, entries[k].tol, d1, d2));
                }
            }
            oracle_status = sl::to_string(entries[0].res.status);
            json oj = json::array();
            for (const auto& e : entries)
                oj.push_back(oracle_json(e));
            J["oracle"] = oj;
            pl.end();
        } else {
            out_line("ORACLE skipped (--no-oracle): no psi_or, no ceiling line, e_psi absent");
        }
        pl.begin("metrics");
        const ReportOutcome ro = report_solution(ctx, sc, tst.inputs, ref, so, J);
        pl.end();
        ws_bytes += ro.metrics_bytes;
        if (!c.save_solution_dir.empty())
            save_solution(c.save_solution_dir, sc, so, ro, J);
    }
    ws_bytes += prov.stage_device_bytes();
    out_line(fmt("TIMING_SETUP stage_builds=%.3fs (inside the solve phase; Newton-Krylov only = "
                 "%.3fs)",
                 prov.seconds(), pl.seconds("solve") - prov.seconds()));
    J["timing_stage_builds"] = prov.seconds();
    print_timing_memory(pl, ws_bytes, seconds_since(t_all), J);
    J["solver_status"] = so.status;
    J["oracle_status"] = oracle_status;
    std::string status = so.status;
    if (status == "converged" && oracle_status == "oracle_roundtrip_fail")
        status = oracle_status;
    out_line(fmt("STATUS_DETAIL solver=%s oracle=%s", so.status.c_str(), oracle_status.c_str()));
    return finish(status, J, c);
}

// ================================================================================================
// --sf19-crosscheck
// ================================================================================================

inline int run_crosscheck(CudaContext& ctx, const DriverConfig& c) {
    const auto t_all = std::chrono::steady_clock::now();
    json J;
    J["mode"] = "sf19-crosscheck";
    J["command"] = c.command_line;
    J["dir"] = c.crosscheck_dir;
    PhaseLog pl;
    const std::string d = c.crosscheck_dir;
    const sl::NpyArray Y = sl::read_npy(d + "/Y_cells.npy");
    if (Y.ndim() != 3 || Y.shape[0] != Y.shape[1] || Y.shape[1] != Y.shape[2])
        throw std::runtime_error("Y_cells.npy must be (N, N, N)");
    const int N = static_cast<int>(Y.shape[0]);
    const std::vector<std::size_t> ps = {static_cast<std::size_t>(N),
                                         static_cast<std::size_t>(N)};
    const sl::NpyArray v1ref = sl::read_npy_shape(d + "/v1_face_ref.npy", ps);
    const sl::NpyArray v1avg = sl::read_npy_shape(d + "/v1_faceavg_ref.npy", ps);
    const sl::NpyArray vp2ref = sl::read_npy_shape(d + "/vperp_vertex_ref_2.npy", ps);
    const sl::NpyArray vp3ref = sl::read_npy_shape(d + "/vperp_vertex_ref_3.npy", ps);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    out_line(fmt("CROSSCHECK_INPUT dir=%s N=%d (Y_cells = ln k at the cell centres, x-fastest; "
                 "SF-19 qbar=e1 rtol=%.1e auto MG levels)",
                 d.c_str(), N, c.pcg_rtol));

    pl.begin("setup");
    const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(ctx, "crosscheck", N, Y.data);
    sl::ProductionStageOptions sopt;
    sopt.pcg_rtol = c.pcg_rtol;
    // Y_cells already is ln k at the field's amplitude: the stage amplitude is 1.
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, 1.0, sopt);
    pl.end();
    out_block("SETUP ", st.report.summary());
    J["N"] = N;
    if (!st.ok()) {
        print_timing_memory(pl, 0, seconds_since(t_all), J);
        return finish(sl::to_string(st.report.status), J, c);
    }
    const std::size_t n2 = grid.plane_size();
    real sd2 = 0.0, sr2 = 0.0, smx = 0.0, sa2 = 0.0, sar2 = 0.0;
    for (std::size_t p = 0; p < n2; ++p) {
        const real u = st.inlet_v1_samples[p];
        const real dpt = u - v1ref.data[p];
        sd2 += dpt * dpt;
        sr2 += v1ref.data[p] * v1ref.data[p];
        smx = std::max(smx, std::fabs(dpt) / std::fabs(v1ref.data[p]));
        const real da = u - v1avg.data[p];
        sa2 += da * da;
        sar2 += v1avg.data[p] * v1avg.data[p];
    }
    const real rms_rel = std::sqrt(sd2 / sr2), rms_rel_avg = std::sqrt(sa2 / sar2);
    const std::vector<real> vp2 = download(st.inputs.vperp_in[0].data(), n2);
    const std::vector<real> vp3 = download(st.inputs.vperp_in[1].data(), n2);
    real vd2 = 0.0, vmx = 0.0, vr2 = 0.0;
    for (std::size_t p = 0; p < n2; ++p) {
        const real a = vp2[p] - vp2ref.data[p], b = vp3[p] - vp3ref.data[p];
        const real dn = std::sqrt(a * a + b * b);
        vd2 += dn * dn;
        vmx = std::max(vmx, dn);
        vr2 += vp2ref.data[p] * vp2ref.data[p] + vp3ref.data[p] * vp3ref.data[p];
    }
    const real vrms = std::sqrt(vd2 / static_cast<real>(n2));
    const real vref_rms = std::sqrt(vr2 / static_cast<real>(n2));
    const real* G = st.report.darcy.G;
    out_line(fmt("CROSSCHECK N=%d v1_face: rms_rel=%.6e max_rel=%.6e (vs v1_face_ref point values) "
                 "rms_rel_avg=%.6e (vs v1_faceavg_ref) | vperp_vertex: rms=%.6e max=%.6e (spline "
                 "flow k_v (G + grad s_h) at the inlet vertices vs vperp_vertex_ref; ref rms=%.6e) "
                 "| G=(%.12e, %.12e, %.12e)",
                 N, rms_rel, smx, rms_rel_avg, vrms, vmx, vref_rms, G[0], G[1], G[2]));
    J["crosscheck"] = {{"v1_face_rms_rel", jnum(rms_rel)},
                       {"v1_face_max_rel", jnum(smx)},
                       {"v1_face_rms_rel_avg", jnum(rms_rel_avg)},
                       {"vperp_vertex_rms", jnum(vrms)},
                       {"vperp_vertex_max", jnum(vmx)},
                       {"vperp_ref_rms", jnum(vref_rms)},
                       {"G", {jnum(G[0]), jnum(G[1]), jnum(G[2])}},
                       {"inlet_mean", jnum(st.report.inlet_mean)},
                       {"v1_spline_vs_uface_rms_rel", jnum(st.report.v1_diff_rms_rel)}};
    print_timing_memory(pl, 0, seconds_since(t_all), J);
    return finish("ok", J, c);
}

// ================================================================================================
// --linear-probe (SF-33 N7c: discriminating linear-solve experiment)
// ================================================================================================

inline std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == ',') {
            if (!cur.empty())
                out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

struct ProbePrec {
    std::string name;
    sl::SlabCoarseMode mode = sl::SlabCoarseMode::off;
    int profiles = 0;
};

inline ProbePrec parse_probe_prec(const std::string& s) {
    ProbePrec p;
    p.name = s;
    if (s == "pa")
        return p;
    if ((s.rfind("mult", 0) == 0 && s.size() == 5) || (s.rfind("add", 0) == 0 && s.size() == 4)) {
        const char d = s.back();
        if (d == '1' || d == '2') {
            p.mode = s[0] == 'm' ? sl::SlabCoarseMode::mult : sl::SlabCoarseMode::add;
            p.profiles = d - '0';
            return p;
        }
    }
    throw UsageError("--probe-precs: entries pa | mult1 | mult2 | add1 | add2 (got '" + s + "')");
}

inline int run_linear_probe(CudaContext& ctx, const DriverConfig& c) {
    const auto t_all = std::chrono::steady_clock::now();
    json J;
    J["mode"] = "linear_probe";
    J["command"] = c.command_line;
    J["case_dir"] = c.case_dir;
    std::vector<ProbePrec> precs;
    for (const std::string& s : split_csv(c.probe_precs))
        precs.push_back(parse_probe_prec(s));
    if (precs.empty())
        throw UsageError("--probe-precs: empty list");

    const sl::ProtoCaseMeta meta = sl::read_proto_case_meta(c.case_dir);
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(meta.N);
    sl::ProtoStageProvider prov(ctx, c.case_dir, grid, meta.field);
    const real eps_from = c.has_eps_from ? c.eps_from : 0.0;
    const std::string cname = fmt("%s_%g_%d", meta.field.c_str(), meta.eps, meta.N);
    out_line(fmt("PROBE_SETUP case=%s field=%s N=%d stage=%g from=%g k=%d precs=%s tol=%.1e "
                 "restart=%d cap=%d stagnation_factor=%g mu=%s policy_coarse=%s(%d) "
                 "coarse_assembly=%s coarse_factor=%s",
                 cname.c_str(), meta.field.c_str(), grid.n, c.eps_stage, eps_from,
                 c.newton_steps, c.probe_precs.c_str(), c.probe_tol, c.restart, c.max_inner,
                 c.probe_stagnation, c.probe_mu.c_str(), c.coarse.c_str(), c.coarse_profiles,
                 c.coarse_assembly.c_str(), c.coarse_factor.c_str()));
    J["case"] = cname;
    J["stage"] = c.eps_stage;
    J["from"] = eps_from;
    J["k"] = c.newton_steps;

    sl::SlabNewtonKrylov nk;
    nk.prepare(ctx, grid, c.restart, std::max(c.max_newton, std::max(1, c.newton_steps)),
               c.max_inner);
    // policy of the warm start and of the k Newton steps: the driver options (P-A only unless
    // --coarse is given); the preconditioners COMPARED on the frozen Jacobian: --probe-precs
    const sl::SlabNewtonConfig ncfg = newton_config(c);
    if (c.coarse != "off")
        nk.prepare_coarse(ctx, c.coarse_profiles, coarse_assembly_of(c), coarse_factor_of(c));
    DeviceBuffer<real> x(grid.unknown_size());
    const sl::StageInputProvider provider = prov.callback();

    // 1. warm start exactly as the solver: continuation to eps_from
    if (eps_from > 0.0) {
        sl::SlabContinuationConfig ccfg;
        ccfg.max_bisections = c.bisect;
        ccfg.field = meta.field;
        if (!c.probe_ladder.empty()) {
            ccfg.ladder.clear();
            for (const std::string& e : split_csv(c.probe_ladder))
                ccfg.ladder.push_back(detail::parse_double("--probe-ladder", e));
        }
        const sl::SlabContinuationReport cr = nk.solve_with_continuation(
            ctx, eps_from, provider, mspan(x), ccfg, ncfg, sl::slab_stdout_logger);
        out_line(fmt("PROBE_WARM status=%s path=%s r_F=%.3e r_out=%.3e t=%.1fs",
                     sl::to_string(cr.status), cr.path.c_str(), cr.final_newton.r_F,
                     cr.final_newton.r_out, cr.seconds));
        J["warm"] = continuation_json(cr);
        if (cr.status != sl::SlabSolveStatus::converged) {
            J["probe_abort"] = "warm start not converged";
            return finish(sl::to_string(cr.status), J, c);
        }
    } else {
        sl::slab_fill(ctx, 0.0, mspan(x));
    }

    // 2. k Newton steps of the stage with the driver policy (P-A only)
    const sl::SlabStageInputs& in = prov(c.eps_stage);
    const sl::SlabResidualNorms n0 = nk.residual_norms(ctx, in, cspan(x));
    const real m0 = n0.merit();
    out_line(fmt("PROBE_STAGE_START r_F=%.6e r_out=%.6e merit=%.6e", n0.r_F, n0.r_out, m0));
    if (c.newton_steps > 0) {
        sl::SlabNewtonConfig kcfg = ncfg;
        kcfg.max_iterations = c.newton_steps;
        const sl::SlabNewtonReport nr =
            nk.solve(ctx, in, mspan(x), fmt("%s:%g:%d:probe", meta.field.c_str(), c.eps_stage,
                                            grid.n),
                     kcfg, sl::slab_stdout_logger);
        J["steps"] = newton_json(nr);
        const int taken = static_cast<int>(nr.hist_r_F.size()) - 1;
        if (taken != c.newton_steps && nr.status != sl::SlabSolveStatus::converged) {
            out_line(fmt("PROBE_ABORT only %d of %d Newton steps taken (status %s)", taken,
                         c.newton_steps, sl::to_string(nr.status)));
            J["probe_abort"] = "newton steps not taken";
            return finish(sl::to_string(nr.status), J, c);
        }
    }
    const sl::SlabResidualNorms nk_n = nk.residual_norms(ctx, in, cspan(x));
    const real mk = nk_n.merit();
    const bool psitc = ncfg.psitc.enabled;
    const real mu_ser =
        psitc ? std::fmin(std::fmax(ncfg.psitc.mu0 * (mk / m0), 0.0), ncfg.psitc.mu_max) : 0.0;
    out_line(fmt("PROBE_ITERATE k=%d r_F=%.6e r_out=%.6e merit=%.6e mu_SER=%.6e (psitc %s, "
                 "mu0 %g, m0 %.6e)",
                 c.newton_steps, nk_n.r_F, nk_n.r_out, mk, mu_ser, psitc ? "on" : "off",
                 ncfg.psitc.mu0, m0));
    J["iterate"] = {{"r_F", jnum(nk_n.r_F)},   {"r_out", jnum(nk_n.r_out)},
                    {"merit", jnum(mk)},        {"merit_stage_start", jnum(m0)},
                    {"mu_ser", jnum(mu_ser)}};

    // 3. freeze the Jacobian at x
    sl::SlabResidualWorkspace rws;
    rws.prepare(grid);
    DeviceBuffer<real> U1(grid.full_size()), U2(grid.full_size());
    DeviceBuffer<real> E(grid.unknown_size()), rhs(grid.unknown_size()), dx(grid.unknown_size());
    sl::assemble_full_planes(ctx, grid, cspan(x), in, mspan(U1), mspan(U2));
    sl::evaluate_residual(ctx, grid, in, cspan(U1), cspan(U2), mspan(E), rws, nullptr);
    sl::slab_scale_copy(ctx, -1.0, cspan(E), mspan(rhs));
    sl::SlabJvpWorkspace& jws = nk.jvp();
    jws.prepare_base(ctx, grid, in, cspan(U1), cspan(U2));
    sl::SlabModePreconditioner& pa = nk.preconditioner();
    sl::SlabGmres& gm = nk.gmres();
    sl::SlabCoarseCorrection cc1, cc2;
    bool need1 = false, need2 = false;
    for (const auto& p : precs) {
        need1 = need1 || p.profiles == 1;
        need2 = need2 || p.profiles == 2;
    }
    if (need1)
        cc1.prepare(ctx, grid, 1, coarse_assembly_of(c), coarse_factor_of(c));
    if (need2)
        cc2.prepare(ctx, grid, 2, coarse_assembly_of(c), coarse_factor_of(c));

    std::vector<std::pair<std::string, real>> mus;
    if (c.probe_mu != "zero")
        mus.push_back({"ser", mu_ser});
    if (c.probe_mu != "ser" && !(c.probe_mu == "both" && mu_ser == 0.0))
        mus.push_back({"zero", 0.0});

    sl::SlabGmresConfig gcfg;
    gcfg.tol = c.probe_tol;
    gcfg.restart = c.restart;
    gcfg.max_iterations = c.max_inner;
    gcfg.stagnation_factor = c.probe_stagnation;
    json results = json::array();
    for (const auto& mpair : mus) {
        const real mu = mpair.second;
        const sl::SlabPrecFactorReport fr = pa.factor(ctx, grid, in, cspan(U1), cspan(U2), mu);
        const sl::SlabCoarseCorrection::Operator opA = [&](DeviceSpan<const real> a,
                                                           DeviceSpan<real> b) {
            jws.apply(ctx, grid, a, b);
            if (mu != 0.0)
                sl::slab_add_pseudo_time_shift(ctx, grid, in, mu, a, b);
        };
        const sl::SlabCoarseCorrection::Operator opPA = [&](DeviceSpan<const real> a,
                                                            DeviceSpan<real> b) {
            pa.apply(ctx, grid, a, b);
        };
        for (const ProbePrec& p : precs) {
            json r;
            r["prec"] = p.name;
            r["mu_kind"] = mpair.first;
            r["mu"] = jnum(mu);
            r["pa_singular_modes"] = fr.singular_modes;
            sl::SlabCoarseCorrection* cc =
                p.profiles == 1 ? &cc1 : (p.profiles == 2 ? &cc2 : nullptr);
            std::string cinfo;
            bool usable = fr.singular_modes == 0;
            if (cc != nullptr && usable) {
                const sl::SlabCoarseBuildReport br = cc->build(ctx, grid, opA);
                usable = br.zero_pivots == 0;
                cinfo = fmt(" K=%d t_assembly=%.3fs t_lu=%.3fs t_cond=%.3fs norm1=%.3e "
                            "inv_norm1_est=%.3e rcond_est=%.3e min|U_kk|=%.3e max|U_kk|=%.3e "
                            "zero_pivots=%d assembly=%s factor=%s applications_asm=%d kl=%d",
                            br.K, br.t_assembly, br.t_lu, br.t_cond, br.norm1, br.inv_norm1_est,
                            br.rcond_est, br.min_abs_u, br.max_abs_u, br.zero_pivots,
                            sl::to_string(br.assembly), sl::to_string(br.factor),
                            br.applications, br.kl);
                r["coarse"] = {{"K", br.K},
                               {"profiles", br.profiles},
                               {"t_assembly", br.t_assembly},
                               {"t_lu", br.t_lu},
                               {"t_cond", br.t_cond},
                               {"norm1", jnum(br.norm1)},
                               {"inv_norm1_est", jnum(br.inv_norm1_est)},
                               {"rcond_est", jnum(br.rcond_est)},
                               {"min_abs_u", jnum(br.min_abs_u)},
                               {"max_abs_u", jnum(br.max_abs_u)},
                               {"zero_pivots", br.zero_pivots},
                               {"assembly", sl::to_string(br.assembly)},
                               {"factor", sl::to_string(br.factor)},
                               {"applications_asm", br.applications},
                               {"color_period", br.color_period},
                               {"kl", br.kl},
                               {"ku", br.ku}};
            }
            sl::SlabGmresReport gr;
            if (usable) {
                const sl::SlabGmres::Operator opM = [&](DeviceSpan<const real> a,
                                                        DeviceSpan<real> b) {
                    if (cc != nullptr)
                        cc->apply(ctx, grid, p.mode, opA, opPA, a, b);
                    else
                        pa.apply(ctx, grid, a, b);
                };
                gr = gm.solve(ctx, opA, opM, cspan(rhs), mspan(dx), gcfg);
            }
            std::string apinfo;
            if (cc != nullptr && cc->applications() > 0)
                apinfo = fmt(" applications=%d t_apply_avg=%.3fms t_host_solve_avg=%.3fms",
                             cc->applications(), 1e3 * cc->apply_seconds() / cc->applications(),
                             1e3 * cc->host_solve_seconds() / cc->applications());
            out_line(fmt("PROBE case=%s stage=%g from=%g k=%d mu_kind=%s mu=%.3e prec=%s its=%d "
                         "cycles=%d status=%s rel=%.3e t=%.2fs pa_singular=%d%s%s",
                         cname.c_str(), c.eps_stage, eps_from, c.newton_steps,
                         mpair.first.c_str(), mu, p.name.c_str(), gr.iterations, gr.cycles,
                         sl::to_string(gr.status), gr.rel_residual, gr.seconds, fr.singular_modes,
                         cinfo.c_str(), apinfo.c_str()));
            std::string curve;
            for (real v : gr.cycle_true)
                curve += fmt(" %.3e", v);
            out_line(fmt("PROBE_CURVE case=%s stage=%g k=%d mu_kind=%s prec=%s true:%s",
                         cname.c_str(), c.eps_stage, c.newton_steps, mpair.first.c_str(),
                         p.name.c_str(), curve.c_str()));
            r["its"] = gr.iterations;
            r["cycles"] = gr.cycles;
            r["status"] = sl::to_string(gr.status);
            r["rel"] = jnum(gr.rel_residual);
            r["seconds"] = gr.seconds;
            r["cycle_true"] = jvec(gr.cycle_true);
            r["cycle_rec"] = jvec(gr.cycle_recurrence);
            if (cc != nullptr) {
                r["applications"] = cc->applications();
                r["t_apply_avg"] =
                    cc->applications() > 0 ? cc->apply_seconds() / cc->applications() : 0.0;
            }
            results.push_back(r);
        }
    }
    J["results"] = results;
    J["timing_total"] = seconds_since(t_all);
    out_line(fmt("PROBE_DONE case=%s stage=%g k=%d t=%.1fs", cname.c_str(), c.eps_stage,
                 c.newton_steps, seconds_since(t_all)));
    // the GMRES outcomes are the measurement: the probe itself reports "ok" (exit 0)
    return finish("ok", J, c);
}

// ================================================================================================
// entry
// ================================================================================================

inline int run(int argc, char** argv) {
    std::printf("command:");
    for (int i = 0; i < argc; ++i)
        std::printf(" %s", argv[i]);
    std::printf("\n");
    std::fflush(stdout);
    DriverConfig c;
    try {
        c = parse_args(argc, argv);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "inlet_slab: %s\n%s", e.what(), usage_text());
        return kExitUsage;
    }
    try {
        CudaContext ctx(c.device);
        switch (c.mode) {
        case Mode::proto:
            return run_proto(ctx, c);
        case Mode::production:
            return run_production(ctx, c);
        case Mode::crosscheck:
            return run_crosscheck(ctx, c);
        case Mode::linear_probe:
            return run_linear_probe(ctx, c);
        case Mode::none:
            break;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inlet_slab: error: %s\n", e.what());
        std::printf("STATUS exception\n");
        std::fflush(stdout);
        return kExitException;
    }
    return kExitUsage;
}

} // namespace inlet_slab_app
} // namespace macroflow3d
