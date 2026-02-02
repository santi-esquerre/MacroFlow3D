/**
 * @file ConfigYaml.cpp
 * @brief YAML parser for AppConfig
 * 
 * Uses yaml-cpp. Tolerant parsing: unknown keys are ignored,
 * missing optional fields get defaults.
 */

#include "Config.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace rwpt {
namespace io {

namespace {

// Helper to get value or default
template<typename T>
T get_or(const YAML::Node& node, const std::string& key, T default_val) {
    if (node[key]) {
        return node[key].as<T>();
    }
    return default_val;
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
GridConfig parse_grid(const YAML::Node& node) {
    GridConfig cfg;
    if (!node) return cfg;
    
    cfg.nx = get_or<int>(node, "nx", 64);
    cfg.ny = get_or<int>(node, "ny", 64);
    cfg.nz = get_or<int>(node, "nz", 64);
    cfg.dx = get_or<real>(node, "dx", 1.0);
    
    return cfg;
}

// Parse stochastic section
StochasticYamlConfig parse_stochastic(const YAML::Node& node) {
    StochasticYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.sigma2 = get_or<real>(node, "sigma2", 1.0);
    cfg.corr_length = get_or<real>(node, "corr_length", 1.0);
    cfg.n_modes = get_or<int>(node, "n_modes", 1000);
    cfg.covariance_type = get_or<int>(node, "covariance_type", 0);
    cfg.seed = get_or<uint64_t>(node, "seed", 12345);
    cfg.K_mean = get_or<real>(node, "K_mean", 1.0);
    
    return cfg;
}

// Parse flow section
FlowYamlConfig parse_flow(const YAML::Node& node) {
    FlowYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.solver = get_or<std::string>(node, "solver", "mg");
    cfg.mg_levels = get_or<int>(node, "mg_levels", 4);
    cfg.mg_pre_smooth = get_or<int>(node, "mg_pre_smooth", 2);
    cfg.mg_post_smooth = get_or<int>(node, "mg_post_smooth", 2);
    cfg.mg_coarse_iters = get_or<int>(node, "mg_coarse_iters", 50);
    cfg.mg_max_cycles = get_or<int>(node, "mg_max_cycles", 20);
    cfg.cg_max_iter = get_or<int>(node, "cg_max_iter", 1000);
    cfg.cg_rtol = get_or<real>(node, "cg_rtol", 1e-8);
    cfg.cg_check_every = get_or<int>(node, "cg_check_every", 10);
    cfg.rtol = get_or<real>(node, "rtol", 1e-6);
    
    // Verification flag
    cfg.verify_velocity = get_or<bool>(node, "verify_velocity", false);
    
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
TransportYamlConfig parse_transport(const YAML::Node& node) {
    TransportYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.n_particles = get_or<int>(node, "n_particles", 10000);
    cfg.dt = get_or<real>(node, "dt", 0.01);
    cfg.n_steps = get_or<int>(node, "n_steps", 1000);
    cfg.porosity = get_or<real>(node, "porosity", 1.0);
    cfg.diffusion = get_or<real>(node, "diffusion", 0.0);
    cfg.seed = get_or<uint64_t>(node, "seed", 54321);
    cfg.output_every = get_or<int>(node, "output_every", 100);
    cfg.inject_x = get_or<real>(node, "inject_x", 0.0);
    
    return cfg;
}

// Parse output section
OutputYamlConfig parse_output(const YAML::Node& node) {
    OutputYamlConfig cfg;
    if (!node) return cfg;
    
    cfg.output_dir = get_or<std::string>(node, "output_dir", "./output");
    cfg.save_K = get_or<bool>(node, "save_K", true);
    cfg.save_head = get_or<bool>(node, "save_head", true);
    cfg.save_velocity = get_or<bool>(node, "save_velocity", false);
    cfg.save_particles = get_or<bool>(node, "save_particles", true);
    cfg.format = get_or<std::string>(node, "format", "binary");
    
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
    
    AppConfig cfg;
    
    // Parse sections (all optional, use defaults if missing)
    cfg.grid = parse_grid(root["grid"]);
    cfg.stochastic = parse_stochastic(root["stochastic"]);
    cfg.flow = parse_flow(root["flow"]);
    cfg.transport = parse_transport(root["transport"]);
    cfg.output = parse_output(root["output"]);
    
    // Validate critical fields
    if (cfg.grid.nx <= 0 || cfg.grid.ny <= 0 || cfg.grid.nz <= 0) {
        throw std::runtime_error("Grid dimensions must be positive (nx, ny, nz)");
    }
    if (cfg.grid.dx <= 0) {
        throw std::runtime_error("Grid spacing must be positive (dx)");
    }
    
    return cfg;
}

} // namespace io
} // namespace rwpt
