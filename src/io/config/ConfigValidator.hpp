#pragma once

/**
 * @file ConfigValidator.hpp
 * @brief Semantic validation for AppConfig.
 *
 * Call validate_config() BEFORE any GPU allocation. If it returns
 * errors, the caller should print them and exit early.
 *
 * Every check produces a message like:
 *   "[transport.dt] 0 ≤ 0: must be positive"
 */

#include "Config.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace io {

/**
 * @brief Result of config validation.
 */
struct ValidationResult {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    bool ok() const { return errors.empty(); }

    /// Pretty-print all errors + warnings to stderr.
    void dump(FILE* out = stderr) const {
        for (const auto& w : warnings)
            std::fprintf(out, "  [config WARNING] %s\n", w.c_str());
        for (const auto& e : errors)
            std::fprintf(out, "  [config ERROR]   %s\n", e.c_str());
    }
};

/**
 * @brief Validate the entire AppConfig semantically.
 */
inline ValidationResult validate_config(const AppConfig& cfg) {
    ValidationResult r;

    auto err = [&](const std::string& path, const std::string& msg) {
        r.errors.push_back("[" + path + "] " + msg);
    };
    auto warn = [&](const std::string& path, const std::string& msg) {
        r.warnings.push_back("[" + path + "] " + msg);
    };

    // ── Grid ─────────────────────────────────────────────────────────
    if (cfg.grid.nx <= 0)
        err("grid.nx", std::to_string(cfg.grid.nx) + " <= 0: must be positive");
    if (cfg.grid.ny <= 0)
        err("grid.ny", std::to_string(cfg.grid.ny) + " <= 0: must be positive");
    if (cfg.grid.nz <= 0)
        err("grid.nz", std::to_string(cfg.grid.nz) + " <= 0: must be positive");
    if (cfg.grid.dx <= 0)
        err("grid.dx", std::to_string(cfg.grid.dx) + " <= 0: must be positive");

    // Power-of-2 recommendation (for multigrid)
    auto is_pow2 = [](int n) { return n > 0 && (n & (n - 1)) == 0; };
    if (!is_pow2(cfg.grid.nx) || !is_pow2(cfg.grid.ny) || !is_pow2(cfg.grid.nz)) {
        warn("grid", "dimensions not power-of-2; multigrid may underperform");
    }

    // ── Stochastic ───────────────────────────────────────────────────
    if (cfg.stochastic.sigma2 < 0)
        err("stochastic.sigma2", "must be >= 0");
    if (cfg.stochastic.corr_length <= 0)
        err("stochastic.corr_length", "must be > 0");
    if (cfg.stochastic.n_modes <= 0)
        err("stochastic.n_modes", "must be > 0");
    if (cfg.stochastic.K_mean <= 0)
        err("stochastic.K_mean", "must be > 0");

    // ── Flow ─────────────────────────────────────────────────────────
    const auto& solver = cfg.flow.solver;
    if (solver != "mg" && solver != "cg" && solver != "mg_cg")
        err("flow.solver", "'" + solver + "' unknown; expected mg|cg|mg_cg");
    if (cfg.flow.rtol <= 0)
        err("flow.rtol", "must be > 0");
    if (cfg.flow.mg_levels < 1)
        err("flow.mg_levels", "must be >= 1");

    // Periodic BC must be paired
    auto periodic_pair = [&](const char* lo, const char* hi, BCType tlo, BCType thi) {
        if ((tlo == BCType::Periodic) != (thi == BCType::Periodic)) {
            err(std::string("flow.bc.") + lo + "/" + hi,
                "periodic BC must appear on both faces of the same axis");
        }
    };
    periodic_pair("xmin", "xmax", cfg.flow.bc.xmin.type, cfg.flow.bc.xmax.type);
    periodic_pair("ymin", "ymax", cfg.flow.bc.ymin.type, cfg.flow.bc.ymax.type);
    periodic_pair("zmin", "zmax", cfg.flow.bc.zmin.type, cfg.flow.bc.zmax.type);

    // ── Transport ────────────────────────────────────────────────────
    if (cfg.transport.n_particles <= 0)
        err("transport.n_particles", "must be > 0");
    if (cfg.transport.dt <= 0)
        err("transport.dt", "must be > 0");
    if (cfg.transport.n_steps <= 0)
        err("transport.n_steps", "must be > 0");
    if (cfg.transport.porosity <= 0)
        err("transport.porosity", "must be > 0");
    if (cfg.transport.diffusion < 0)
        err("transport.diffusion", "must be >= 0");
    if (cfg.transport.alpha_l < 0)
        err("transport.alpha_l", "must be >= 0");
    if (cfg.transport.alpha_t < 0)
        err("transport.alpha_t", "must be >= 0");
    if (cfg.transport.alpha_t > cfg.transport.alpha_l)
        warn("transport.alpha_t", "alpha_t > alpha_l is unusual");
    if (cfg.transport.output_every <= 0)
        err("transport.output_every", "must be > 0");
    // transport.velocity_layout is now derived from method; no longer validated
    // here.
    if (cfg.transport.method != "par2" && cfg.transport.method != "pspta")
        err("transport.method",
            "'" + cfg.transport.method + "' unknown; expected 'par2' or 'pspta'");
    if (cfg.transport.pspta_refine.enabled) {
        if (cfg.transport.method != "pspta") {
            warn("transport.pspta_refine.enabled",
                 "enabled but transport.method!='pspta'; refinement will be ignored");
        }
        if (cfg.transport.pspta_refine.outer_iters <= 0)
            err("transport.pspta_refine.outer_iters", "must be > 0 when enabled");
        if (cfg.transport.pspta_refine.omega <= 0 || cfg.transport.pspta_refine.omega > 1)
            err("transport.pspta_refine.omega", "must be in (0, 1]");
        if (cfg.transport.pspta_refine.omega_min <= 0)
            err("transport.pspta_refine.omega_min", "must be > 0");
        if (cfg.transport.pspta_refine.omega_min > cfg.transport.pspta_refine.omega)
            err("transport.pspta_refine.omega_min", "must be <= omega");
        if (cfg.transport.pspta_refine.max_backtracks <= 0)
            err("transport.pspta_refine.max_backtracks", "must be > 0");
        if (cfg.transport.pspta_refine.eps_vx <= 0)
            err("transport.pspta_refine.eps_vx", "must be > 0");
        if (cfg.transport.pspta_refine.source_clip_cells <= 0)
            err("transport.pspta_refine.source_clip_cells", "must be > 0");
        if (cfg.transport.pspta_refine.no_descent_patience <= 0)
            err("transport.pspta_refine.no_descent_patience", "must be > 0");
        if (cfg.transport.pspta_refine.stop_rel_rms < 0)
            err("transport.pspta_refine.stop_rel_rms", "must be >= 0");
        if (cfg.transport.pspta_refine.stop_abs_rms < 0)
            err("transport.pspta_refine.stop_abs_rms", "must be >= 0");
    }

    // ── Analysis / macrodispersion ───────────────────────────────────
    const auto& mac = cfg.analysis.macrodispersion;
    if (mac.enabled) {
        if (mac.NR < 1)
            err("analysis.macrodispersion.NR", "must be >= 1");
        if (mac.lambda <= 0)
            err("analysis.macrodispersion.lambda", "must be > 0");
        if (mac.vmean_norm <= 0)
            err("analysis.macrodispersion.vmean_norm", "must be > 0");
        if (mac.sample_every <= 0)
            err("analysis.macrodispersion.sample_every", "must be > 0");
        if (mac.var_estimator != "biased" && mac.var_estimator != "unbiased")
            err("analysis.macrodispersion.var_estimator", "expected 'biased' or 'unbiased'");
    }

    // ── Analysis / snapshots ─────────────────────────────────────────
    const auto& snap = cfg.analysis.snapshots;
    if (snap.enabled) {
        if (snap.every <= 0)
            err("analysis.snapshots.every", "must be > 0");
        if (snap.stride < 1)
            err("analysis.snapshots.stride", "must be >= 1");
        if (snap.precision < 1 || snap.precision > 20)
            warn("analysis.snapshots.precision",
                 std::to_string(snap.precision) + " outside [1,20]");
    }

    // ── Output ───────────────────────────────────────────────────────
    if (cfg.output.output_dir.empty())
        err("output.output_dir", "must not be empty");

    // ── Streamfunction solver (SF-16 T01) ───────────────────────────────
    // Mirrors the library's own validation ranges
    // (streamfunctions::validate_streamfunction_problem); only checked when
    // the section is enabled, so a disabled/absent section never affects
    // validation outcome.
    const auto& sf = cfg.streamfunction_solver;
    if (sf.enabled) {
        const auto& amv = sf.affine_mean_velocity;
        if (amv.mode != "fixed" && amv.mode != "measured")
            err("streamfunction_solver.affine_mean_velocity.mode",
                "'" + amv.mode + "' unknown; expected 'fixed' or 'measured'");
        if (!std::isfinite(amv.value) || amv.value <= 0)
            err("streamfunction_solver.affine_mean_velocity.value", "must be finite and > 0");

        if (!std::isfinite(sf.epsilon) || sf.epsilon < 0)
            err("streamfunction_solver.epsilon", "must be finite and >= 0");
        if (!std::isfinite(sf.eta) || sf.eta < 0)
            err("streamfunction_solver.eta", "must be finite and >= 0");

        if (sf.picard.max_iter < 0)
            err("streamfunction_solver.picard.max_iter", "must be >= 0");
        if (!std::isfinite(sf.picard.tolerance) || sf.picard.tolerance <= 0)
            err("streamfunction_solver.picard.tolerance", "must be finite and > 0");
        if (!std::isfinite(sf.picard.omega) || sf.picard.omega <= 0 || sf.picard.omega > 1)
            err("streamfunction_solver.picard.omega", "must be finite and in (0, 1]");

        if (!std::isfinite(sf.linear.rtol) || sf.linear.rtol < 0)
            err("streamfunction_solver.linear.rtol", "must be finite and >= 0");
        if (sf.linear.max_iter < 0)
            err("streamfunction_solver.linear.max_iter", "must be >= 0");
        if (sf.linear.check_every <= 0)
            err("streamfunction_solver.linear.check_every", "must be > 0");

        if (sf.mg.num_levels < 1)
            err("streamfunction_solver.mg.num_levels", "must be >= 1");

        // ── Anderson acceleration (SF-21 T03r, re-activation decision R2d) ──
        // Mirrors streamfunctions::require_valid_anderson_config
        // (StreamfunctionWorkspace.cu): validated unconditionally, regardless
        // of `anderson.enabled`, matching that library convention (an invalid
        // Anderson config is invalid even when acceleration is off).
        {
            const auto& an = sf.anderson;
            const char* p = "streamfunction_solver.anderson.";
            if (an.depth < 3 || an.depth > 8)
                err(std::string(p) + "depth", "must be in [3, 8]");
            if (an.start_iteration < 1)
                err(std::string(p) + "start_iteration", "must be >= 1");
            if (!std::isfinite(an.condition_limit) || an.condition_limit <= 1)
                err(std::string(p) + "condition_limit", "must be finite and > 1");
        }

        // ── Newton-Krylov phase (SF-24 T02) ──────────────────────────────
        // Mirrors streamfunctions::require_valid_newton_config
        // (StreamfunctionWorkspace.cu): validated unconditionally, regardless
        // of `newton.enabled`, matching that library convention (an invalid
        // Newton config is invalid even when the phase is off). The SF-24
        // C01 `enabled` requires `adaptive.enabled` rule is checked first, as
        // its own distinct message, mirroring the library's FIRST-check
        // ordering.
        {
            const auto& nw = sf.newton;
            const char* p = "streamfunction_solver.newton.";
            if (nw.enabled && !sf.adaptive.enabled)
                err(std::string(p) + "enabled",
                    "requires streamfunction_solver.adaptive.enabled: true");
            if (!std::isfinite(nw.activation_r_F) || nw.activation_r_F <= 0)
                err(std::string(p) + "activation_r_F", "must be finite and > 0");
            if (!std::isfinite(nw.stagnation_activation_r_F) ||
                nw.stagnation_activation_r_F < nw.activation_r_F)
                err(std::string(p) + "stagnation_activation_r_F",
                    "must be finite and >= activation_r_F");
            if (!std::isfinite(nw.forcing_min) || nw.forcing_min <= 0 ||
                !std::isfinite(nw.forcing_max) || nw.forcing_max < nw.forcing_min ||
                nw.forcing_max > 1)
                err(std::string(p) + "forcing_min/forcing_max",
                    "must satisfy finite 0 < forcing_min <= forcing_max <= 1");
            if (!std::isfinite(nw.forcing_coefficient) || nw.forcing_coefficient <= 0)
                err(std::string(p) + "forcing_coefficient", "must be finite and > 0");
            if (!std::isfinite(nw.armijo_c) || nw.armijo_c < 0 || nw.armijo_c >= 1)
                err(std::string(p) + "armijo_c", "must be finite and in [0, 1)");
            if (!std::isfinite(nw.alpha_min) || nw.alpha_min <= 0 || nw.alpha_min > 1)
                err(std::string(p) + "alpha_min", "must be finite and in (0, 1]");
            if (!std::isfinite(nw.backtrack_factor) ||
                !(nw.backtrack_factor > 0 && nw.backtrack_factor < 1))
                err(std::string(p) + "backtrack_factor", "must be finite and in (0, 1)");
            if (nw.max_newton_iterations < 1)
                err(std::string(p) + "max_newton_iterations", "must be >= 1");
            if (nw.rescue_picard_steps < 0)
                err(std::string(p) + "rescue_picard_steps", "must be >= 0");

            // Mirrors streamfunctions::validate_coupled_gmres_config
            // (CoupledGmres.cu); `rel_tol` is not exposed here (see
            // StreamfunctionNewtonGmresYamlConfig), so it is not validated.
            const char* gp = "streamfunction_solver.newton.gmres.";
            if (nw.gmres.restart < 1 || nw.gmres.restart > 15)
                err(std::string(gp) + "restart", "must be in [1, 15]");
            if (nw.gmres.max_iterations < 1)
                err(std::string(gp) + "max_iterations", "must be >= 1");

            // Mirrors streamfunctions::validate_jvp_delta_config
            // (JacobianVectorProduct.cu).
            const char* dp = "streamfunction_solver.newton.delta.";
            if (!std::isfinite(nw.delta.delta_min) || !std::isfinite(nw.delta.delta_max) ||
                !(nw.delta.delta_min > 0) || !(nw.delta.delta_max > nw.delta.delta_min))
                err(std::string(dp) + "delta_min/delta_max",
                    "must satisfy finite 0 < delta_min < delta_max");
        }

        // ── Field/Darcy sources (SF-21 T03) ─────────────────────────────
        if (sf.field_source != "stochastic" && sf.field_source != "periodic_gaussian")
            err("streamfunction_solver.field_source",
                "'" + sf.field_source + "' unknown; expected 'stochastic' or 'periodic_gaussian'");
        if (sf.darcy_source != "pipeline" && sf.darcy_source != "affine_periodic")
            err("streamfunction_solver.darcy_source",
                "'" + sf.darcy_source + "' unknown; expected 'pipeline' or 'affine_periodic'");

        // Mirrors physics::PeriodicGaussianFieldConfig::validate (SF-18);
        // only checked when field_source actually selects this generator.
        if (sf.field_source == "periodic_gaussian") {
            const auto& pg = sf.periodic_gaussian;
            if (!std::isfinite(pg.sigma2) || pg.sigma2 < 0)
                err("streamfunction_solver.periodic_gaussian.sigma2", "must be finite and >= 0");
            if (!std::isfinite(pg.corr_length) || pg.corr_length <= 0)
                err("streamfunction_solver.periodic_gaussian.corr_length",
                    "must be finite and > 0");
        }

        // ── Continuation (SF-17 T03) ────────────────────────────────────
        // Mirrors streamfunctions::validate_streamfunction_continuation_config
        // (ContinuationController.hpp). The eta axis TARGET is the existing
        // top-level `sf.eta`; the epsilon axis STARTING point is the existing
        // top-level `sf.epsilon` (activation bitácora decision 9) — neither
        // is duplicated in the `continuation` subsection.
        const auto& cont = sf.continuation;
        {
            const auto& e = cont.eta;
            const char* p = "streamfunction_solver.continuation.eta.";
            if (!std::isfinite(e.start))
                err(std::string(p) + "start", "must be finite");
            if (!std::isfinite(e.initial_step))
                err(std::string(p) + "initial_step", "must be finite");
            if (!std::isfinite(e.min_step))
                err(std::string(p) + "min_step", "must be finite");
            if (!std::isfinite(e.max_step))
                err(std::string(p) + "max_step", "must be finite");
            if (!std::isfinite(e.backtrack_factor))
                err(std::string(p) + "backtrack_factor", "must be finite");
            if (!std::isfinite(e.growth_factor))
                err(std::string(p) + "growth_factor", "must be finite");
            if (std::isfinite(e.start) && std::isfinite(sf.eta) && !(e.start <= sf.eta))
                err(std::string(p) + "start", "must be <= streamfunction_solver.eta (the eta target)");
            if (!(e.initial_step > 0))
                err(std::string(p) + "initial_step", "must be > 0");
            if (!(e.min_step > 0))
                err(std::string(p) + "min_step", "must be > 0");
            if (e.min_step > 0 && e.initial_step > 0 && !(e.min_step <= e.initial_step))
                err(std::string(p) + "min_step", "must be <= initial_step");
            if (e.initial_step > 0 && e.max_step > 0 && !(e.initial_step <= e.max_step))
                err(std::string(p) + "max_step", "must be >= initial_step");
            if (!(e.backtrack_factor > 0 && e.backtrack_factor < 1))
                err(std::string(p) + "backtrack_factor", "must be in (0, 1)");
            if (!(e.growth_factor >= 1))
                err(std::string(p) + "growth_factor", "must be >= 1");
            if (!(e.easy_streak >= 1))
                err(std::string(p) + "easy_streak", "must be >= 1");
        }
        {
            const auto& e = cont.epsilon;
            const char* p = "streamfunction_solver.continuation.epsilon.";
            if (!std::isfinite(e.target))
                err(std::string(p) + "target", "must be finite");
            if (!std::isfinite(e.initial_step_log10))
                err(std::string(p) + "initial_step_log10", "must be finite");
            if (!std::isfinite(e.min_step_log10))
                err(std::string(p) + "min_step_log10", "must be finite");
            if (!std::isfinite(e.max_step_log10))
                err(std::string(p) + "max_step_log10", "must be finite");
            if (!std::isfinite(e.backtrack_factor))
                err(std::string(p) + "backtrack_factor", "must be finite");
            if (!std::isfinite(e.growth_factor))
                err(std::string(p) + "growth_factor", "must be finite");
            if (!(e.target > 0))
                err(std::string(p) + "target", "must be > 0");
            if (e.target > 0 && std::isfinite(sf.epsilon) && !(e.target <= sf.epsilon))
                err(std::string(p) + "target",
                    "must be <= streamfunction_solver.epsilon (the epsilon starting value)");
            if (!(e.initial_step_log10 > 0))
                err(std::string(p) + "initial_step_log10", "must be > 0");
            if (!(e.min_step_log10 > 0))
                err(std::string(p) + "min_step_log10", "must be > 0");
            if (e.min_step_log10 > 0 && e.initial_step_log10 > 0 &&
                !(e.min_step_log10 <= e.initial_step_log10))
                err(std::string(p) + "min_step_log10", "must be <= initial_step_log10");
            if (e.initial_step_log10 > 0 && e.max_step_log10 > 0 &&
                !(e.initial_step_log10 <= e.max_step_log10))
                err(std::string(p) + "max_step_log10", "must be >= initial_step_log10");
            if (!(e.backtrack_factor > 0 && e.backtrack_factor < 1))
                err(std::string(p) + "backtrack_factor", "must be in (0, 1)");
            if (!(e.growth_factor >= 1))
                err(std::string(p) + "growth_factor", "must be >= 1");
            if (!(e.easy_streak >= 1))
                err(std::string(p) + "easy_streak", "must be >= 1");
        }

        // ── Lambda (heterogeneity) continuation (SF-21 T03) ─────────────
        // Mirrors validate_streamfunction_heterogeneity_continuation_config's
        // lambda-axis checks (ContinuationController.hpp); the axis TARGET
        // is fixed at 1 by the library, so no `target` field is validated
        // here. `enabled` (this eta/epsilon leg) and `lambda.enabled` are
        // mutually exclusive: the lambda driver replaces both the single-solve
        // and the eta/epsilon-only continuation call for the stage.
        {
            const auto& e = cont.lambda;
            const char* p = "streamfunction_solver.continuation.lambda.";
            if (cont.enabled && e.enabled)
                err("streamfunction_solver.continuation",
                    "'enabled' (eta/epsilon leg) and 'lambda.enabled' cannot both be true");
            if (e.enabled && sf.field_source != "periodic_gaussian")
                err(std::string(p) + "enabled",
                    "requires streamfunction_solver.field_source: periodic_gaussian");
            if (e.enabled && sf.darcy_source != "affine_periodic")
                err(std::string(p) + "enabled",
                    "requires streamfunction_solver.darcy_source: affine_periodic");
            if (!std::isfinite(e.start))
                err(std::string(p) + "start", "must be finite");
            if (!std::isfinite(e.initial_step))
                err(std::string(p) + "initial_step", "must be finite");
            if (!std::isfinite(e.min_step))
                err(std::string(p) + "min_step", "must be finite");
            if (!std::isfinite(e.max_step))
                err(std::string(p) + "max_step", "must be finite");
            if (!std::isfinite(e.backtrack_factor))
                err(std::string(p) + "backtrack_factor", "must be finite");
            if (!std::isfinite(e.growth_factor))
                err(std::string(p) + "growth_factor", "must be finite");
            if (std::isfinite(e.start) && !(e.start <= real{1}))
                err(std::string(p) + "start", "must be <= 1 (the lambda target)");
            if (!(e.initial_step > 0))
                err(std::string(p) + "initial_step", "must be > 0");
            if (!(e.min_step > 0))
                err(std::string(p) + "min_step", "must be > 0");
            if (e.min_step > 0 && e.initial_step > 0 && !(e.min_step <= e.initial_step))
                err(std::string(p) + "min_step", "must be <= initial_step");
            if (e.initial_step > 0 && e.max_step > 0 && !(e.initial_step <= e.max_step))
                err(std::string(p) + "max_step", "must be >= initial_step");
            if (!(e.backtrack_factor > 0 && e.backtrack_factor < 1))
                err(std::string(p) + "backtrack_factor", "must be in (0, 1)");
            if (!(e.growth_factor >= 1))
                err(std::string(p) + "growth_factor", "must be >= 1");
            if (!(e.easy_streak >= 1))
                err(std::string(p) + "easy_streak", "must be >= 1");
        }
    }

    return r;
}

/**
 * @brief Validate and throw if errors found. For callers who want fail-fast.
 */
inline void require_valid_config(const AppConfig& cfg) {
    auto result = validate_config(cfg);
    if (!result.ok()) {
        result.dump();
        std::ostringstream oss;
        oss << "Config validation failed with " << result.errors.size() << " error(s)";
        throw std::runtime_error(oss.str());
    }
    if (!result.warnings.empty()) {
        result.dump(); // print warnings even if no errors
    }
}

} // namespace io
} // namespace macroflow3d
