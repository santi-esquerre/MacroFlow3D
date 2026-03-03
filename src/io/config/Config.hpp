#pragma once

/**
 * @file Config.hpp
 * @brief Application configuration loaded from YAML
 * @ingroup io_config
 * 
 * Aggregates all config needed for the physics pipeline:
 * Grid + Stochastic + Flow + Transport
 * 
 * The config structs here mirror YAML structure and are used for I/O.
 * Runtime numerical types (like PinSpec) are in the numerics layer.
 */

#include "../../core/Scalar.hpp"
#include "../../core/BCSpec.hpp"
#include "../../numerics/pin_spec.hpp"  // PinMode
#include <string>
#include <cstdint>
#include <array>

namespace macroflow3d {
namespace io {

// Re-export PinMode from numerics layer for config usage
using macroflow3d::PinMode;

/**
 * @brief Execution mode for the pipeline.
 */
enum class RunMode {
    SingleRun,     ///< NR=1, single realization
    Ensemble,      ///< NR>=1, full pipeline + ensemble analysis
    AnalysisOnly   ///< No GPU — read existing CSV and run macrodispersion
};

/// Parse run mode from string. Returns Ensemble by default.
inline RunMode parse_run_mode(const std::string& s) {
    if (s == "single_run" || s == "single") return RunMode::SingleRun;
    if (s == "analysis_only" || s == "analysis") return RunMode::AnalysisOnly;
    return RunMode::Ensemble;  // default
}

/**
 * @brief Pin configuration for flow solver
 * 
 * Configures how the solver handles singular systems (all periodic/Neumann).
 * See pin_spec.hpp for full documentation on the pin mechanism.
 */
struct PinConfig {
    PinMode mode = PinMode::Auto;  // auto | on | off
};

/**
 * @brief Grid configuration
 */
struct GridConfig {
    int nx = 64;
    int ny = 64;
    int nz = 64;
    real dx = 1.0;  // Isotropic: dy = dz = dx
    
    // Computed domain size
    real Lx() const { return nx * dx; }
    real Ly() const { return ny * dx; }
    real Lz() const { return nz * dx; }
};

/**
 * @brief Stochastic K field configuration
 */
struct StochasticYamlConfig {
    real sigma2 = 1.0;           // Variance of log-K
    real corr_length = 1.0;      // Correlation length
    int  n_modes = 1000;         // Number of Fourier modes
    int  covariance_type = 0;    // 0 = exponential, 1 = gaussian
    uint64_t seed = 12345;       // RNG seed
    real K_mean = 1.0;           // Geometric mean of K
};

/**
 * @brief Flow solver configuration
 */
struct FlowYamlConfig {
    // Solver type: "mg", "cg", "mg_cg" (MG-preconditioned CG)
    std::string solver = "mg";
    
    // MG parameters
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_coarse_iters = 50;
    int mg_max_cycles = 20;
    
    // CG parameters
    int cg_max_iter = 1000;
    real cg_rtol = 1e-8;
    int cg_check_every = 10;  // Check convergence every N iterations
    
    // Convergence
    real rtol = 1e-6;
    
    // Boundary conditions (6 faces)
    // Legacy names: west/east=x, south/north=y, bottom/top=z
    BCSpec bc;
    
    // Pin configuration for singular systems (legacy: pin1stCell)
    PinConfig pin;
    
    // Verification: compare computed velocity vs theoretical Darcy
    bool verify_velocity = false;
};

/**
 * @brief Particle transport configuration
 */
struct TransportYamlConfig {
    int n_particles = 10000;
    real dt = 0.01;
    int n_steps = 1000;
    real porosity = 1.0;
    real diffusion = 0.0;       // Molecular diffusion (Dm) [L²/T]
    real alpha_l = 0.0;         // Longitudinal dispersivity [L]
    real alpha_t = 0.0;         // Transverse dispersivity [L]
    uint64_t seed = 54321;
    
    // Output frequency
    int output_every = 100;
    
    // Snapshot interval (0 = no snapshots, only final)
    int snapshot_every = 0;
    
    // Injection (default: x=0 plane spanning full YZ domain)
    real inject_x = 0.0;
    
    // Velocity layout — DERIVED from method, NOT user-configurable.
    // "par2"  → "padded";  "pspta" → "compact"
    std::string velocity_layout = "padded";

    // Transport method: "par2" (default) | "pspta"
    std::string method = "par2";

    // Enable PSPTA-specific diagnostics (ψ quality + Newton fail summary).
    // Written to psi_quality.csv and newton_fail_summary.csv.
    // Has no effect for method=="par2".  Default OFF.
    bool pspta_diagnostics = false;
};

/**
 * @brief Macrodispersion analysis configuration
 */
struct MacrodispersionConfig {
    bool enabled = false;
    int NR = 1;                  // Number of realizations
    real lambda = 1.0;           // Correlation length for alpha
    real vmean_norm = 1.0;       // ||<v>|| (provided; could be computed from flow)
    int sample_every = 10;       // Stats sampling frequency (transport steps)
    std::string var_estimator = "biased";  // "biased" (paper) or "unbiased" (Par2_Core raw)
};

/**
 * @brief Snapshot configuration (using Par2_Core CsvSnapshotWriter)
 */
struct SnapshotConfig {
    bool enabled = false;
    int every = 200;             // Steps between snapshots
    bool legacy_format = true;
    bool include_time = false;
    bool include_status = false;
    bool include_wrap_counts = false;
    bool include_unwrapped = false;
    int stride = 1;
    int max_particles = -1;      // -1 = no limit
    int precision = 15;
};

/**
 * @brief Analysis configuration (macrodispersion + snapshots)
 */
struct AnalysisConfig {
    MacrodispersionConfig macrodispersion;
    SnapshotConfig snapshots;
};

/**
 * @brief Diagnostics configuration
 */
struct DiagnosticsConfig {
    bool velocity_field = false;  ///< Run divergence/vorticity/helicity diagnostics
};

/**
 * @brief Output configuration
 */
struct OutputYamlConfig {
    std::string output_dir = "./output";
    bool save_K = true;
    bool save_head = true;
    bool save_velocity = false;
    bool save_particles = true;
    std::string format = "binary";  // "binary" or "vtk"
};

/**
 * @brief Complete application configuration
 */
struct AppConfig {
    RunMode run_mode = RunMode::Ensemble;
    GridConfig grid;
    StochasticYamlConfig stochastic;
    FlowYamlConfig flow;
    TransportYamlConfig transport;
    AnalysisConfig analysis;
    DiagnosticsConfig diagnostics;
    OutputYamlConfig output;
    
    // Validation helpers
    bool is_valid() const {
        return grid.nx > 0 && grid.ny > 0 && grid.nz > 0 && grid.dx > 0;
    }
};

/**
 * @brief Load configuration from YAML file
 * 
 * @param path Path to YAML config file
 * @return AppConfig Parsed configuration with defaults for missing fields
 * @throws std::runtime_error if file not found or critical fields missing
 */
AppConfig load_config_yaml(const std::string& path);

} // namespace io
} // namespace macroflow3d
