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
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <set>
#include <string>

namespace macroflow3d {
namespace io {

namespace {

// Helper to get value or keep existing default
template<typename T>
T get_or(const YAML::Node& node, const std::string& key, T default_val) {
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
void check_unknown_keys(const YAML::Node& node,
                        const std::set<std::string>& known,
                        const std::string& path,
                        std::vector<std::string>& errs)
{
    if (!node || !node.IsMap()) return;
    for (auto it = node.begin(); it != node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        if (known.find(key) == known.end()) {
            errs.push_back("Unknown key '" + path + "." + key + "'");
        }
    }
}

// Parse BCType from string
BCType parse_bc_type(const std::string& s) {
    if (s == "dirichlet" || s == "Dirichlet") return BCType::Dirichlet;
    if (s == "neumann" || s == "Neumann") return BCType::Neumann;
    if (s == "periodic" || s == "Periodic") return BCType::Periodic;
    throw std::runtime_error("Unknown BC type: " + s);
}

// Parse a single BC face with validation
BCFace parse_bc_face(const YAML::Node& node, const std::string& face_name) {
    BCFace face;
    if (!node) return face;  // Default: Dirichlet 0
    
    std::string type_str = get_or<std::string>(node, "type", "dirichlet");
    face.type = parse_bc_type(type_str);
    
    // Periodic doesn't need a value
    if (face.type == BCType::Periodic) {
        face.value = 0.0;  // Ignored
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
    if (!node) return cfg;

    static const std::set<std::string> known = {"nx","ny","nz","dx"};
    check_unknown_keys(node, known, "grid", errs);

    cfg.nx = get_or<int>(node, "nx", def.nx);
    cfg.ny = get_or<int>(node, "ny", def.ny);
    cfg.nz = get_or<int>(node, "nz", def.nz);
    cfg.dx = get_or<real>(node, "dx", def.dx);

    return cfg;
}

// Parse stochastic section
StochasticYamlConfig parse_stochastic(const YAML::Node& node,
                                      const StochasticYamlConfig& def,
                                      std::vector<std::string>& errs) {
    StochasticYamlConfig cfg = def;
    if (!node) return cfg;

    static const std::set<std::string> known =
        {"sigma2","corr_length","n_modes","covariance_type","seed","K_mean"};
    check_unknown_keys(node, known, "stochastic", errs);

    cfg.sigma2         = get_or<real>(node, "sigma2", def.sigma2);
    cfg.corr_length    = get_or<real>(node, "corr_length", def.corr_length);
    cfg.n_modes        = get_or<int>(node, "n_modes", def.n_modes);
    cfg.covariance_type = get_or<int>(node, "covariance_type", def.covariance_type);
    cfg.seed           = get_or<uint64_t>(node, "seed", def.seed);
    cfg.K_mean         = get_or<real>(node, "K_mean", def.K_mean);

    return cfg;
}

// Parse flow section
FlowYamlConfig parse_flow(const YAML::Node& node, const FlowYamlConfig& def,
                           std::vector<std::string>& errs) {
    FlowYamlConfig cfg = def;
    if (!node) return cfg;

    static const std::set<std::string> known =
        {"solver","mg_levels","mg_pre_smooth","mg_post_smooth","mg_coarse_iters",
         "mg_max_cycles","cg_max_iter","cg_rtol","cg_check_every","rtol",
         "verify_velocity","pin","pin_first_cell","bc"};
    check_unknown_keys(node, known, "flow", errs);

    cfg.solver         = get_or<std::string>(node, "solver", def.solver);
    cfg.mg_levels      = get_or<int>(node, "mg_levels", def.mg_levels);
    cfg.mg_pre_smooth  = get_or<int>(node, "mg_pre_smooth", def.mg_pre_smooth);
    cfg.mg_post_smooth = get_or<int>(node, "mg_post_smooth", def.mg_post_smooth);
    cfg.mg_coarse_iters = get_or<int>(node, "mg_coarse_iters", def.mg_coarse_iters);
    cfg.mg_max_cycles  = get_or<int>(node, "mg_max_cycles", def.mg_max_cycles);
    cfg.cg_max_iter    = get_or<int>(node, "cg_max_iter", def.cg_max_iter);
    cfg.cg_rtol        = get_or<real>(node, "cg_rtol", def.cg_rtol);
    cfg.cg_check_every = get_or<int>(node, "cg_check_every", def.cg_check_every);
    cfg.rtol           = get_or<real>(node, "rtol", def.rtol);

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
            cfg.pin.mode = PinMode::Auto;  // default
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
TransportYamlConfig parse_transport(const YAML::Node& node,
                                    const TransportYamlConfig& def,
                                    std::vector<std::string>& errs) {
    TransportYamlConfig cfg = def;
    if (!node) return cfg;

    static const std::set<std::string> known =
        {"n_particles","dt","n_steps","porosity","diffusion","alpha_l","alpha_t",
         "seed","output_every","snapshot_every","inject_x","velocity_layout"};
    check_unknown_keys(node, known, "transport", errs);

    cfg.n_particles    = get_or<int>(node, "n_particles", def.n_particles);
    cfg.dt             = get_or<real>(node, "dt", def.dt);
    cfg.n_steps        = get_or<int>(node, "n_steps", def.n_steps);
    cfg.porosity       = get_or<real>(node, "porosity", def.porosity);
    cfg.diffusion      = get_or<real>(node, "diffusion", def.diffusion);
    cfg.alpha_l        = get_or<real>(node, "alpha_l", def.alpha_l);
    cfg.alpha_t        = get_or<real>(node, "alpha_t", def.alpha_t);
    cfg.seed           = get_or<uint64_t>(node, "seed", def.seed);
    cfg.output_every   = get_or<int>(node, "output_every", def.output_every);
    cfg.snapshot_every = get_or<int>(node, "snapshot_every", def.snapshot_every);
    cfg.inject_x       = get_or<real>(node, "inject_x", def.inject_x);
    cfg.velocity_layout = get_or<std::string>(node, "velocity_layout", def.velocity_layout);

    return cfg;
}

// Parse output section
OutputYamlConfig parse_output(const YAML::Node& node, const OutputYamlConfig& def,
                              std::vector<std::string>& errs) {
    OutputYamlConfig cfg = def;
    if (!node) return cfg;

    static const std::set<std::string> known =
        {"output_dir","save_K","save_head","save_velocity","save_particles","format"};
    check_unknown_keys(node, known, "output", errs);

    cfg.output_dir     = get_or<std::string>(node, "output_dir", def.output_dir);
    cfg.save_K         = get_or<bool>(node, "save_K", def.save_K);
    cfg.save_head      = get_or<bool>(node, "save_head", def.save_head);
    cfg.save_velocity  = get_or<bool>(node, "save_velocity", def.save_velocity);
    cfg.save_particles = get_or<bool>(node, "save_particles", def.save_particles);
    cfg.format         = get_or<std::string>(node, "format", def.format);

    return cfg;
}

// Parse analysis section
AnalysisConfig parse_analysis(const YAML::Node& node, const AnalysisConfig& def,
                              std::vector<std::string>& errs) {
    AnalysisConfig cfg = def;
    if (!node) return cfg;

    static const std::set<std::string> analysis_known = {"macrodispersion","snapshots"};
    check_unknown_keys(node, analysis_known, "analysis", errs);

    // Macrodispersion sub-section
    if (node["macrodispersion"]) {
        const auto& m = node["macrodispersion"];
        static const std::set<std::string> mac_known =
            {"enabled","NR","lambda","vmean_norm","sample_every","var_estimator"};
        check_unknown_keys(m, mac_known, "analysis.macrodispersion", errs);

        cfg.macrodispersion.enabled       = get_or<bool>(m, "enabled", def.macrodispersion.enabled);
        cfg.macrodispersion.NR            = get_or<int>(m, "NR", def.macrodispersion.NR);
        cfg.macrodispersion.lambda        = get_or<real>(m, "lambda", def.macrodispersion.lambda);
        cfg.macrodispersion.vmean_norm    = get_or<real>(m, "vmean_norm", def.macrodispersion.vmean_norm);
        cfg.macrodispersion.sample_every  = get_or<int>(m, "sample_every", def.macrodispersion.sample_every);
        cfg.macrodispersion.var_estimator = get_or<std::string>(m, "var_estimator", def.macrodispersion.var_estimator);
    }

    // Snapshots sub-section
    if (node["snapshots"]) {
        const auto& s = node["snapshots"];
        static const std::set<std::string> snap_known =
            {"enabled","every","legacy_format","include_time","include_status",
             "include_wrap_counts","include_unwrapped","stride","max_particles","precision"};
        check_unknown_keys(s, snap_known, "analysis.snapshots", errs);

        cfg.snapshots.enabled             = get_or<bool>(s, "enabled", def.snapshots.enabled);
        cfg.snapshots.every               = get_or<int>(s, "every", def.snapshots.every);
        cfg.snapshots.legacy_format       = get_or<bool>(s, "legacy_format", def.snapshots.legacy_format);
        cfg.snapshots.include_time        = get_or<bool>(s, "include_time", def.snapshots.include_time);
        cfg.snapshots.include_status      = get_or<bool>(s, "include_status", def.snapshots.include_status);
        cfg.snapshots.include_wrap_counts = get_or<bool>(s, "include_wrap_counts", def.snapshots.include_wrap_counts);
        cfg.snapshots.include_unwrapped   = get_or<bool>(s, "include_unwrapped", def.snapshots.include_unwrapped);
        cfg.snapshots.stride              = get_or<int>(s, "stride", def.snapshots.stride);
        cfg.snapshots.max_particles       = get_or<int>(s, "max_particles", def.snapshots.max_particles);
        cfg.snapshots.precision           = get_or<int>(s, "precision", def.snapshots.precision);
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
    static const std::set<std::string> top_known =
        {"run_mode","grid","stochastic","flow","transport","analysis","output"};
    check_unknown_keys(root, top_known, "", unknown_errs);

    // Run mode (top-level, optional)
    if (root["run_mode"]) {
        cfg.run_mode = parse_run_mode(root["run_mode"].as<std::string>());
    }

    // Parse sections (merge over defaults)
    cfg.grid       = parse_grid(root["grid"], cfg.grid, unknown_errs);
    cfg.stochastic = parse_stochastic(root["stochastic"], cfg.stochastic, unknown_errs);
    cfg.flow       = parse_flow(root["flow"], cfg.flow, unknown_errs);
    cfg.transport  = parse_transport(root["transport"], cfg.transport, unknown_errs);
    cfg.analysis   = parse_analysis(root["analysis"], cfg.analysis, unknown_errs);
    cfg.output     = parse_output(root["output"], cfg.output, unknown_errs);

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
