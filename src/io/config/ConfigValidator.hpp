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
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <stdexcept>

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
    if (cfg.grid.nx <= 0) err("grid.nx", std::to_string(cfg.grid.nx) + " <= 0: must be positive");
    if (cfg.grid.ny <= 0) err("grid.ny", std::to_string(cfg.grid.ny) + " <= 0: must be positive");
    if (cfg.grid.nz <= 0) err("grid.nz", std::to_string(cfg.grid.nz) + " <= 0: must be positive");
    if (cfg.grid.dx <= 0) err("grid.dx", std::to_string(cfg.grid.dx) + " <= 0: must be positive");

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
    auto periodic_pair = [&](const char* lo, const char* hi,
                             BCType tlo, BCType thi) {
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
    // transport.velocity_layout is now derived from method; no longer validated here.
    if (cfg.transport.method != "par2" && cfg.transport.method != "pspta")
        err("transport.method",
            "'" + cfg.transport.method + "' unknown; expected 'par2' or 'pspta'");

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
            err("analysis.macrodispersion.var_estimator",
                "expected 'biased' or 'unbiased'");
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
        result.dump();  // print warnings even if no errors
    }
}

} // namespace io
} // namespace macroflow3d
