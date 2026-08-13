/**
 * @file ConfigYaml.cpp
 * @brief Strict YAML parser for AppConfig
 *
 * Merge-over-defaults: start from make_default_config(), overlay
 * only recognized keys. Unknown keys trigger an error.
 *
 * Etapa 7: strict mode, centralized defaults, effective-config serialization.
 */

#include "Config.hpp"
#include "ConfigDefaults.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace macroflow3d {
namespace io {

namespace {

// Helper to get value or keep existing default
template <typename T> T get_or(const YAML::Node& node, const std::string& key, T default_val) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return default_val;
}

// ── Strict-mode unknown-key detection ──────────────────────────────

/**
 * @brief Check that all keys in `node` are in `known`.
 *
 * @param node   YAML map node to inspect.
 * @param known  Set of recognized keys.
 * @param path   Human-readable path prefix for error messages.
 * @param errs   Accumulator for error strings.
 */
void check_unknown_keys(const YAML::Node& node, const std::set<std::string>& known,
                        const std::string& path, std::vector<std::string>& errs) {
    if (!node || !node.IsMap())
        return;
    for (auto it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (known.find(key) == known.end()) {
            errs.push_back("Unknown key '" + path + "." + key + "'");
        }
    }
}

// Parse BCType from string
BCType parse_bc_type(const std::string& s) {
    if (s == "dirichlet" || s == "Dirichlet")
        return BCType::Dirichlet;
    if (s == "neumann" || s == "Neumann")
        return BCType::Neumann;
    if (s == "periodic" || s == "Periodic")
        return BCType::Periodic;
    throw std::runtime_error("Unknown BC type: " + s);
}

// Parse a single BC face with validation
BCFace parse_bc_face(const YAML::Node& node, const std::string& face_name) {
    BCFace face;
    if (!node)
        return face; // Default: Dirichlet 0

    std::string type_str = get_or<std::string>(node, "type", "dirichlet");
    face.type = parse_bc_type(type_str);

    // Periodic doesn't need a value
    if (face.type == BCType::Periodic) {
        face.value = 0.0; // Ignored
    } else {
        // Dirichlet/Neumann need value
        if (!node["value"]) {
            // Use default 0 but warn (could be intentional)
            face.value = 0.0;
        } else {
            face.value = node["value"].as<real>();
        }
    }
    return face;
}

// Parse grid section
GridConfig parse_grid(const YAML::Node& node, const GridConfig& def,
                      std::vector<std::string>& errs) {
    GridConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {"nx", "ny", "nz", "dx"};
    check_unknown_keys(node, known, "grid", errs);

    cfg.nx = get_or<int>(node, "nx", def.nx);
    cfg.ny = get_or<int>(node, "ny", def.ny);
    cfg.nz = get_or<int>(node, "nz", def.nz);
    cfg.dx = get_or<real>(node, "dx", def.dx);

    return cfg;
}

// Parse stochastic section
StochasticYamlConfig parse_stochastic(const YAML::Node& node, const StochasticYamlConfig& def,
                                      std::vector<std::string>& errs) {
    StochasticYamlConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {"sigma2",          "corr_length", "n_modes",
                                                "covariance_type", "seed",        "K_mean"};
    check_unknown_keys(node, known, "stochastic", errs);

    cfg.sigma2 = get_or<real>(node, "sigma2", def.sigma2);
    cfg.corr_length = get_or<real>(node, "corr_length", def.corr_length);
    cfg.n_modes = get_or<int>(node, "n_modes", def.n_modes);
    cfg.covariance_type = get_or<int>(node, "covariance_type", def.covariance_type);
    cfg.seed = get_or<uint64_t>(node, "seed", def.seed);
    cfg.K_mean = get_or<real>(node, "K_mean", def.K_mean);

    return cfg;
}

// Parse flow section
FlowYamlConfig parse_flow(const YAML::Node& node, const FlowYamlConfig& def,
                          std::vector<std::string>& errs) {
    FlowYamlConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {"solver",          "mg_levels",
                                                "mg_pre_smooth",   "mg_post_smooth",
                                                "mg_coarse_iters", "mg_max_cycles",
                                                "cg_max_iter",     "cg_rtol",
                                                "cg_check_every",  "rtol",
                                                "verify_velocity", "pin",
                                                "pin_first_cell",  "bc"};
    check_unknown_keys(node, known, "flow", errs);

    cfg.solver = get_or<std::string>(node, "solver", def.solver);
    cfg.mg_levels = get_or<int>(node, "mg_levels", def.mg_levels);
    cfg.mg_pre_smooth = get_or<int>(node, "mg_pre_smooth", def.mg_pre_smooth);
    cfg.mg_post_smooth = get_or<int>(node, "mg_post_smooth", def.mg_post_smooth);
    cfg.mg_coarse_iters = get_or<int>(node, "mg_coarse_iters", def.mg_coarse_iters);
    cfg.mg_max_cycles = get_or<int>(node, "mg_max_cycles", def.mg_max_cycles);
    cfg.cg_max_iter = get_or<int>(node, "cg_max_iter", def.cg_max_iter);
    cfg.cg_rtol = get_or<real>(node, "cg_rtol", def.cg_rtol);
    cfg.cg_check_every = get_or<int>(node, "cg_check_every", def.cg_check_every);
    cfg.rtol = get_or<real>(node, "rtol", def.rtol);

    // Verification flag
    cfg.verify_velocity = get_or<bool>(node, "verify_velocity", def.verify_velocity);

    // Pin configuration (legacy: pin1stCell diagonal doubling)
    // Format: flow.pin.mode = "auto" | "on" | "off"
    // Legacy format: flow.pin_first_cell = true/false (backward compat)
    // Note: pin always applies to cell [0,0,0], value is not configurable
    if (node["pin"]) {
        const auto& pin_node = node["pin"];

        // Parse mode: "auto" | "on" | "off"
        std::string mode_str = get_or<std::string>(pin_node, "mode", "auto");
        if (mode_str == "on") {
            cfg.pin.mode = PinMode::On;
        } else if (mode_str == "off") {
            cfg.pin.mode = PinMode::Off;
        } else {
            cfg.pin.mode = PinMode::Auto; // default
        }
        // Note: pin.value and pin.index are ignored (legacy diagonal doubling)
    } else if (node["pin_first_cell"]) {
        // Legacy format backward compatibility
        bool pin_enabled = get_or<bool>(node, "pin_first_cell", false);
        cfg.pin.mode = pin_enabled ? PinMode::On : PinMode::Off;
    }
    // else: defaults (mode=Auto)

    // Parse boundary conditions
    // Support both legacy names (west/east/south/north/bottom/top)
    // and coordinate names (xmin/xmax/ymin/ymax/zmin/zmax)
    if (node["bc"]) {
        const auto& bc_node = node["bc"];

        // X direction: west/east or xmin/xmax
        if (bc_node["west"]) {
            cfg.bc.xmin = parse_bc_face(bc_node["west"], "west(xmin)");
        } else if (bc_node["xmin"]) {
            cfg.bc.xmin = parse_bc_face(bc_node["xmin"], "xmin");
        }

        if (bc_node["east"]) {
            cfg.bc.xmax = parse_bc_face(bc_node["east"], "east(xmax)");
        } else if (bc_node["xmax"]) {
            cfg.bc.xmax = parse_bc_face(bc_node["xmax"], "xmax");
        }

        // Y direction: south/north or ymin/ymax
        if (bc_node["south"]) {
            cfg.bc.ymin = parse_bc_face(bc_node["south"], "south(ymin)");
        } else if (bc_node["ymin"]) {
            cfg.bc.ymin = parse_bc_face(bc_node["ymin"], "ymin");
        }

        if (bc_node["north"]) {
            cfg.bc.ymax = parse_bc_face(bc_node["north"], "north(ymax)");
        } else if (bc_node["ymax"]) {
            cfg.bc.ymax = parse_bc_face(bc_node["ymax"], "ymax");
        }

        // Z direction: bottom/top or zmin/zmax
        if (bc_node["bottom"]) {
            cfg.bc.zmin = parse_bc_face(bc_node["bottom"], "bottom(zmin)");
        } else if (bc_node["zmin"]) {
            cfg.bc.zmin = parse_bc_face(bc_node["zmin"], "zmin");
        }

        if (bc_node["top"]) {
            cfg.bc.zmax = parse_bc_face(bc_node["top"], "top(zmax)");
        } else if (bc_node["zmax"]) {
            cfg.bc.zmax = parse_bc_face(bc_node["zmax"], "zmax");
        }
    }

    return cfg;
}

// Parse transport section
TransportYamlConfig parse_transport(const YAML::Node& node, const TransportYamlConfig& def,
                                    std::vector<std::string>& errs) {
    TransportYamlConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {"n_particles",       "dt",          "n_steps",
                                                "porosity",          "diffusion",   "alpha_l",
                                                "alpha_t",           "seed",        "output_every",
                                                "snapshot_every",    "inject_x",    "method",
                                                "pspta_diagnostics", "pspta_refine"};
    check_unknown_keys(node, known, "transport", errs);

    cfg.n_particles = get_or<int>(node, "n_particles", def.n_particles);
    cfg.dt = get_or<real>(node, "dt", def.dt);
    cfg.n_steps = get_or<int>(node, "n_steps", def.n_steps);
    cfg.porosity = get_or<real>(node, "porosity", def.porosity);
    cfg.diffusion = get_or<real>(node, "diffusion", def.diffusion);
    cfg.alpha_l = get_or<real>(node, "alpha_l", def.alpha_l);
    cfg.alpha_t = get_or<real>(node, "alpha_t", def.alpha_t);
    cfg.seed = get_or<uint64_t>(node, "seed", def.seed);
    cfg.output_every = get_or<int>(node, "output_every", def.output_every);
    cfg.snapshot_every = get_or<int>(node, "snapshot_every", def.snapshot_every);
    cfg.inject_x = get_or<real>(node, "inject_x", def.inject_x);
    cfg.method = get_or<std::string>(node, "method", def.method);
    cfg.pspta_diagnostics = get_or<bool>(node, "pspta_diagnostics", def.pspta_diagnostics);

    if (node["pspta_refine"]) {
        const auto& r = node["pspta_refine"];
        static const std::set<std::string> refine_known = {
            "enabled",           "outer_iters",         "omega",
            "omega_min",         "max_backtracks",      "eps_vx",
            "source_clip_cells", "no_descent_patience", "stop_rel_rms",
            "stop_abs_rms",      "print_every_iter",    "save_history_csv",
            "eq13_diagnostics"};
        check_unknown_keys(r, refine_known, "transport.pspta_refine", errs);

        cfg.pspta_refine.enabled = get_or<bool>(r, "enabled", def.pspta_refine.enabled);
        cfg.pspta_refine.outer_iters = get_or<int>(r, "outer_iters", def.pspta_refine.outer_iters);
        cfg.pspta_refine.omega = get_or<real>(r, "omega", def.pspta_refine.omega);
        cfg.pspta_refine.omega_min = get_or<real>(r, "omega_min", def.pspta_refine.omega_min);
        cfg.pspta_refine.max_backtracks =
            get_or<int>(r, "max_backtracks", def.pspta_refine.max_backtracks);
        cfg.pspta_refine.eps_vx = get_or<real>(r, "eps_vx", def.pspta_refine.eps_vx);
        cfg.pspta_refine.source_clip_cells =
            get_or<real>(r, "source_clip_cells", def.pspta_refine.source_clip_cells);
        cfg.pspta_refine.no_descent_patience =
            get_or<int>(r, "no_descent_patience", def.pspta_refine.no_descent_patience);
        cfg.pspta_refine.stop_rel_rms =
            get_or<real>(r, "stop_rel_rms", def.pspta_refine.stop_rel_rms);
        cfg.pspta_refine.stop_abs_rms =
            get_or<real>(r, "stop_abs_rms", def.pspta_refine.stop_abs_rms);
        cfg.pspta_refine.print_every_iter =
            get_or<bool>(r, "print_every_iter", def.pspta_refine.print_every_iter);
        cfg.pspta_refine.save_history_csv =
            get_or<bool>(r, "save_history_csv", def.pspta_refine.save_history_csv);
        cfg.pspta_refine.eq13_diagnostics =
            get_or<bool>(r, "eq13_diagnostics", def.pspta_refine.eq13_diagnostics);
    }
    // velocity_layout is derived after parsing (not user-configurable)

    return cfg;
}

// Parse output section
OutputYamlConfig parse_output(const YAML::Node& node, const OutputYamlConfig& def,
                              std::vector<std::string>& errs) {
    OutputYamlConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {"output_dir",    "save_K",         "save_head",
                                                "save_velocity", "save_particles", "format"};
    check_unknown_keys(node, known, "output", errs);

    cfg.output_dir = get_or<std::string>(node, "output_dir", def.output_dir);
    cfg.save_K = get_or<bool>(node, "save_K", def.save_K);
    cfg.save_head = get_or<bool>(node, "save_head", def.save_head);
    cfg.save_velocity = get_or<bool>(node, "save_velocity", def.save_velocity);
    cfg.save_particles = get_or<bool>(node, "save_particles", def.save_particles);
    cfg.format = get_or<std::string>(node, "format", def.format);

    return cfg;
}

// Parse streamfunction_solver section (SF-16 T01)
StreamfunctionSolverYamlConfig
parse_streamfunction_solver(const YAML::Node& node, const StreamfunctionSolverYamlConfig& def,
                            std::vector<std::string>& errs) {
    StreamfunctionSolverYamlConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> known = {
        "enabled",     "affine_mean_velocity", "epsilon",          "eta",
        "picard",      "adaptive",             "linear",           "mg",
        "export",      "continuation",         "field_source",     "periodic_gaussian",
        "darcy_source", "anderson",            "newton"};
    check_unknown_keys(node, known, "streamfunction_solver", errs);

    cfg.enabled = get_or<bool>(node, "enabled", def.enabled);
    cfg.epsilon = get_or<real>(node, "epsilon", def.epsilon);
    cfg.eta = get_or<real>(node, "eta", def.eta);
    cfg.field_source = get_or<std::string>(node, "field_source", def.field_source);
    cfg.darcy_source = get_or<std::string>(node, "darcy_source", def.darcy_source);

    if (node["affine_mean_velocity"]) {
        const auto& n = node["affine_mean_velocity"];
        static const std::set<std::string> affine_known = {"mode", "value"};
        check_unknown_keys(n, affine_known, "streamfunction_solver.affine_mean_velocity", errs);

        cfg.affine_mean_velocity.mode =
            get_or<std::string>(n, "mode", def.affine_mean_velocity.mode);
        cfg.affine_mean_velocity.value = get_or<real>(n, "value", def.affine_mean_velocity.value);
    }

    if (node["picard"]) {
        const auto& n = node["picard"];
        static const std::set<std::string> picard_known = {"max_iter", "tolerance", "omega"};
        check_unknown_keys(n, picard_known, "streamfunction_solver.picard", errs);

        cfg.picard.max_iter = get_or<int>(n, "max_iter", def.picard.max_iter);
        cfg.picard.tolerance = get_or<real>(n, "tolerance", def.picard.tolerance);
        cfg.picard.omega = get_or<real>(n, "omega", def.picard.omega);
    }

    if (node["adaptive"]) {
        const auto& n = node["adaptive"];
        static const std::set<std::string> adaptive_known = {"enabled"};
        check_unknown_keys(n, adaptive_known, "streamfunction_solver.adaptive", errs);

        cfg.adaptive.enabled = get_or<bool>(n, "enabled", def.adaptive.enabled);
    }

    if (node["linear"]) {
        const auto& n = node["linear"];
        static const std::set<std::string> linear_known = {"rtol", "max_iter", "check_every"};
        check_unknown_keys(n, linear_known, "streamfunction_solver.linear", errs);

        cfg.linear.rtol = get_or<real>(n, "rtol", def.linear.rtol);
        cfg.linear.max_iter = get_or<int>(n, "max_iter", def.linear.max_iter);
        cfg.linear.check_every = get_or<int>(n, "check_every", def.linear.check_every);
    }

    if (node["mg"]) {
        const auto& n = node["mg"];
        static const std::set<std::string> mg_known = {"num_levels"};
        check_unknown_keys(n, mg_known, "streamfunction_solver.mg", errs);

        cfg.mg.num_levels = get_or<int>(n, "num_levels", def.mg.num_levels);
    }

    if (node["export"]) {
        const auto& n = node["export"];
        static const std::set<std::string> export_known = {"iteration_history", "summary",
                                                            "fields"};
        check_unknown_keys(n, export_known, "streamfunction_solver.export", errs);

        cfg.exports.iteration_history =
            get_or<bool>(n, "iteration_history", def.exports.iteration_history);
        cfg.exports.summary = get_or<bool>(n, "summary", def.exports.summary);
        cfg.exports.fields = get_or<bool>(n, "fields", def.exports.fields);
    }

    if (node["periodic_gaussian"]) {
        const auto& n = node["periodic_gaussian"];
        static const std::set<std::string> pg_known = {"sigma2", "corr_length", "seed",
                                                        "normalize_variance"};
        check_unknown_keys(n, pg_known, "streamfunction_solver.periodic_gaussian", errs);

        // Structural (parse-time) requirement: this subsection is only
        // meaningful with field_source: periodic_gaussian -- checked here
        // (not ConfigValidator.hpp) because "was this YAML key present" is
        // not otherwise recoverable after merge-over-defaults.
        if (cfg.field_source != "periodic_gaussian") {
            errs.push_back(
                "streamfunction_solver.periodic_gaussian requires "
                "streamfunction_solver.field_source: periodic_gaussian");
        }

        cfg.periodic_gaussian.sigma2 = get_or<real>(n, "sigma2", def.periodic_gaussian.sigma2);
        cfg.periodic_gaussian.corr_length =
            get_or<real>(n, "corr_length", def.periodic_gaussian.corr_length);
        cfg.periodic_gaussian.seed =
            get_or<unsigned long long>(n, "seed", def.periodic_gaussian.seed);
        cfg.periodic_gaussian.normalize_variance =
            get_or<bool>(n, "normalize_variance", def.periodic_gaussian.normalize_variance);
    }

    if (node["continuation"]) {
        const auto& n = node["continuation"];
        static const std::set<std::string> continuation_known = {"enabled", "eta", "epsilon",
                                                                   "lambda"};
        check_unknown_keys(n, continuation_known, "streamfunction_solver.continuation", errs);

        cfg.continuation.enabled = get_or<bool>(n, "enabled", def.continuation.enabled);

        if (n["eta"]) {
            const auto& en = n["eta"];
            static const std::set<std::string> eta_known = {
                "start",  "initial_step",    "min_step",     "max_step",
                "backtrack_factor", "growth_factor", "easy_streak"};
            check_unknown_keys(en, eta_known, "streamfunction_solver.continuation.eta", errs);

            cfg.continuation.eta.start = get_or<real>(en, "start", def.continuation.eta.start);
            cfg.continuation.eta.initial_step =
                get_or<real>(en, "initial_step", def.continuation.eta.initial_step);
            cfg.continuation.eta.min_step =
                get_or<real>(en, "min_step", def.continuation.eta.min_step);
            cfg.continuation.eta.max_step =
                get_or<real>(en, "max_step", def.continuation.eta.max_step);
            cfg.continuation.eta.backtrack_factor =
                get_or<real>(en, "backtrack_factor", def.continuation.eta.backtrack_factor);
            cfg.continuation.eta.growth_factor =
                get_or<real>(en, "growth_factor", def.continuation.eta.growth_factor);
            cfg.continuation.eta.easy_streak =
                get_or<int>(en, "easy_streak", def.continuation.eta.easy_streak);
        }

        if (n["epsilon"]) {
            const auto& en = n["epsilon"];
            static const std::set<std::string> epsilon_known = {
                "target",           "initial_step_log10", "min_step_log10",
                "max_step_log10",   "backtrack_factor",   "growth_factor",
                "easy_streak"};
            check_unknown_keys(en, epsilon_known, "streamfunction_solver.continuation.epsilon",
                               errs);

            cfg.continuation.epsilon.target =
                get_or<real>(en, "target", def.continuation.epsilon.target);
            cfg.continuation.epsilon.initial_step_log10 = get_or<real>(
                en, "initial_step_log10", def.continuation.epsilon.initial_step_log10);
            cfg.continuation.epsilon.min_step_log10 =
                get_or<real>(en, "min_step_log10", def.continuation.epsilon.min_step_log10);
            cfg.continuation.epsilon.max_step_log10 =
                get_or<real>(en, "max_step_log10", def.continuation.epsilon.max_step_log10);
            cfg.continuation.epsilon.backtrack_factor = get_or<real>(
                en, "backtrack_factor", def.continuation.epsilon.backtrack_factor);
            cfg.continuation.epsilon.growth_factor =
                get_or<real>(en, "growth_factor", def.continuation.epsilon.growth_factor);
            cfg.continuation.epsilon.easy_streak =
                get_or<int>(en, "easy_streak", def.continuation.epsilon.easy_streak);
        }

        if (n["lambda"]) {
            const auto& ln = n["lambda"];
            static const std::set<std::string> lambda_known = {
                "enabled",    "start",           "initial_step",  "min_step",
                "max_step",   "backtrack_factor", "growth_factor", "easy_streak"};
            check_unknown_keys(ln, lambda_known, "streamfunction_solver.continuation.lambda",
                               errs);

            cfg.continuation.lambda.enabled =
                get_or<bool>(ln, "enabled", def.continuation.lambda.enabled);
            cfg.continuation.lambda.start =
                get_or<real>(ln, "start", def.continuation.lambda.start);
            cfg.continuation.lambda.initial_step =
                get_or<real>(ln, "initial_step", def.continuation.lambda.initial_step);
            cfg.continuation.lambda.min_step =
                get_or<real>(ln, "min_step", def.continuation.lambda.min_step);
            cfg.continuation.lambda.max_step =
                get_or<real>(ln, "max_step", def.continuation.lambda.max_step);
            cfg.continuation.lambda.backtrack_factor =
                get_or<real>(ln, "backtrack_factor", def.continuation.lambda.backtrack_factor);
            cfg.continuation.lambda.growth_factor =
                get_or<real>(ln, "growth_factor", def.continuation.lambda.growth_factor);
            cfg.continuation.lambda.easy_streak =
                get_or<int>(ln, "easy_streak", def.continuation.lambda.easy_streak);
        }
    }

    if (node["anderson"]) {
        const auto& n = node["anderson"];
        static const std::set<std::string> anderson_known = {"enabled", "depth", "start_iteration",
                                                              "condition_limit"};
        check_unknown_keys(n, anderson_known, "streamfunction_solver.anderson", errs);

        cfg.anderson.enabled = get_or<bool>(n, "enabled", def.anderson.enabled);
        cfg.anderson.depth = get_or<int>(n, "depth", def.anderson.depth);
        cfg.anderson.start_iteration =
            get_or<int>(n, "start_iteration", def.anderson.start_iteration);
        cfg.anderson.condition_limit =
            get_or<real>(n, "condition_limit", def.anderson.condition_limit);
    }

    if (node["newton"]) {
        const auto& n = node["newton"];
        static const std::set<std::string> newton_known = {
            "enabled",       "activation_r_F",     "stagnation_activation_r_F",
            "forcing_coefficient", "forcing_min",   "forcing_max",
            "armijo_c",      "alpha_min",           "backtrack_factor",
            "max_newton_iterations", "rescue_picard_steps", "gmres", "delta"};
        check_unknown_keys(n, newton_known, "streamfunction_solver.newton", errs);

        cfg.newton.enabled = get_or<bool>(n, "enabled", def.newton.enabled);
        cfg.newton.activation_r_F =
            get_or<real>(n, "activation_r_F", def.newton.activation_r_F);
        cfg.newton.stagnation_activation_r_F =
            get_or<real>(n, "stagnation_activation_r_F", def.newton.stagnation_activation_r_F);
        cfg.newton.forcing_coefficient =
            get_or<real>(n, "forcing_coefficient", def.newton.forcing_coefficient);
        cfg.newton.forcing_min = get_or<real>(n, "forcing_min", def.newton.forcing_min);
        cfg.newton.forcing_max = get_or<real>(n, "forcing_max", def.newton.forcing_max);
        cfg.newton.armijo_c = get_or<real>(n, "armijo_c", def.newton.armijo_c);
        cfg.newton.alpha_min = get_or<real>(n, "alpha_min", def.newton.alpha_min);
        cfg.newton.backtrack_factor =
            get_or<real>(n, "backtrack_factor", def.newton.backtrack_factor);
        cfg.newton.max_newton_iterations =
            get_or<int>(n, "max_newton_iterations", def.newton.max_newton_iterations);
        cfg.newton.rescue_picard_steps =
            get_or<int>(n, "rescue_picard_steps", def.newton.rescue_picard_steps);

        if (n["gmres"]) {
            const auto& gn = n["gmres"];
            static const std::set<std::string> gmres_known = {"restart", "max_iterations"};
            check_unknown_keys(gn, gmres_known, "streamfunction_solver.newton.gmres", errs);

            cfg.newton.gmres.restart = get_or<int>(gn, "restart", def.newton.gmres.restart);
            cfg.newton.gmres.max_iterations =
                get_or<int>(gn, "max_iterations", def.newton.gmres.max_iterations);
        }

        if (n["delta"]) {
            const auto& dn = n["delta"];
            static const std::set<std::string> delta_known = {"delta_min", "delta_max"};
            check_unknown_keys(dn, delta_known, "streamfunction_solver.newton.delta", errs);

            cfg.newton.delta.delta_min =
                get_or<real>(dn, "delta_min", def.newton.delta.delta_min);
            cfg.newton.delta.delta_max =
                get_or<real>(dn, "delta_max", def.newton.delta.delta_max);
        }
    }

    return cfg;
}

// Parse analysis section
AnalysisConfig parse_analysis(const YAML::Node& node, const AnalysisConfig& def,
                              std::vector<std::string>& errs) {
    AnalysisConfig cfg = def;
    if (!node)
        return cfg;

    static const std::set<std::string> analysis_known = {"macrodispersion", "snapshots"};
    check_unknown_keys(node, analysis_known, "analysis", errs);

    // Macrodispersion sub-section
    if (node["macrodispersion"]) {
        const auto& m = node["macrodispersion"];
        static const std::set<std::string> mac_known = {
            "enabled", "NR", "lambda", "vmean_norm", "sample_every", "var_estimator"};
        check_unknown_keys(m, mac_known, "analysis.macrodispersion", errs);

        cfg.macrodispersion.enabled = get_or<bool>(m, "enabled", def.macrodispersion.enabled);
        cfg.macrodispersion.NR = get_or<int>(m, "NR", def.macrodispersion.NR);
        cfg.macrodispersion.lambda = get_or<real>(m, "lambda", def.macrodispersion.lambda);
        cfg.macrodispersion.vmean_norm =
            get_or<real>(m, "vmean_norm", def.macrodispersion.vmean_norm);
        cfg.macrodispersion.sample_every =
            get_or<int>(m, "sample_every", def.macrodispersion.sample_every);
        cfg.macrodispersion.var_estimator =
            get_or<std::string>(m, "var_estimator", def.macrodispersion.var_estimator);
    }

    // Snapshots sub-section
    if (node["snapshots"]) {
        const auto& s = node["snapshots"];
        static const std::set<std::string> snap_known = {"enabled",           "every",
                                                         "legacy_format",     "include_time",
                                                         "include_status",    "include_wrap_counts",
                                                         "include_unwrapped", "stride",
                                                         "max_particles",     "precision"};
        check_unknown_keys(s, snap_known, "analysis.snapshots", errs);

        cfg.snapshots.enabled = get_or<bool>(s, "enabled", def.snapshots.enabled);
        cfg.snapshots.every = get_or<int>(s, "every", def.snapshots.every);
        cfg.snapshots.legacy_format = get_or<bool>(s, "legacy_format", def.snapshots.legacy_format);
        cfg.snapshots.include_time = get_or<bool>(s, "include_time", def.snapshots.include_time);
        cfg.snapshots.include_status =
            get_or<bool>(s, "include_status", def.snapshots.include_status);
        cfg.snapshots.include_wrap_counts =
            get_or<bool>(s, "include_wrap_counts", def.snapshots.include_wrap_counts);
        cfg.snapshots.include_unwrapped =
            get_or<bool>(s, "include_unwrapped", def.snapshots.include_unwrapped);
        cfg.snapshots.stride = get_or<int>(s, "stride", def.snapshots.stride);
        cfg.snapshots.max_particles = get_or<int>(s, "max_particles", def.snapshots.max_particles);
        cfg.snapshots.precision = get_or<int>(s, "precision", def.snapshots.precision);
    }

    return cfg;
}

} // anonymous namespace

AppConfig load_config_yaml(const std::string& path) {
    // Check file exists
    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("Config file not found: " + path);
    }
    file.close();

    // Parse YAML
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parse error: " + std::string(e.what()));
    }

    // Start from centralized defaults and overlay user values
    AppConfig cfg = make_default_config();
    std::vector<std::string> unknown_errs;

    // Check top-level keys
    static const std::set<std::string> top_known = {
        "run_mode",     "grid",         "stochastic", "flow",
        "transport",    "analysis",     "diagnostics", "output",
        "streamfunction_solver"};
    check_unknown_keys(root, top_known, "", unknown_errs);

    // Run mode (top-level, optional)
    if (root["run_mode"]) {
        cfg.run_mode = parse_run_mode(root["run_mode"].as<std::string>());
    }

    // Parse sections (merge over defaults)
    cfg.grid = parse_grid(root["grid"], cfg.grid, unknown_errs);
    cfg.stochastic = parse_stochastic(root["stochastic"], cfg.stochastic, unknown_errs);
    cfg.flow = parse_flow(root["flow"], cfg.flow, unknown_errs);
    cfg.transport = parse_transport(root["transport"], cfg.transport, unknown_errs);
    // Derive velocity_layout from method — not user-configurable
    cfg.transport.velocity_layout = (cfg.transport.method == "pspta") ? "compact" : "padded";
    cfg.analysis = parse_analysis(root["analysis"], cfg.analysis, unknown_errs);
    // Diagnostics section (simple — single bool)
    if (root["diagnostics"]) {
        const auto& dnode = root["diagnostics"];
        static const std::set<std::string> diag_known = {"velocity_field"};
        check_unknown_keys(dnode, diag_known, "diagnostics", unknown_errs);
        cfg.diagnostics.velocity_field =
            get_or<bool>(dnode, "velocity_field", cfg.diagnostics.velocity_field);
    }

    cfg.output = parse_output(root["output"], cfg.output, unknown_errs);
    cfg.streamfunction_solver =
        parse_streamfunction_solver(root["streamfunction_solver"], cfg.streamfunction_solver,
                                    unknown_errs);

    // Strict mode: reject unknown keys
    if (!unknown_errs.empty()) {
        std::string msg = "YAML strict-mode errors in '" + path + "':\n";
        for (const auto& e : unknown_errs) {
            msg += "  - " + e + "\n";
        }
        throw std::runtime_error(msg);
    }

    return cfg;
}

} // namespace io
} // namespace macroflow3d
