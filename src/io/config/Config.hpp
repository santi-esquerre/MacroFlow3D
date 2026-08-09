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

#include "../../core/BCSpec.hpp"
#include "../../core/Scalar.hpp"
#include "../../numerics/pin_spec.hpp" // PinMode
#include <array>
#include <cstdint>
#include <string>

namespace macroflow3d {
namespace io {

// Re-export PinMode from numerics layer for config usage
using macroflow3d::PinMode;

/**
 * @brief Execution mode for the pipeline.
 */
enum class RunMode {
    SingleRun,   ///< NR=1, single realization
    Ensemble,    ///< NR>=1, full pipeline + ensemble analysis
    AnalysisOnly ///< No GPU — read existing CSV and run macrodispersion
};

/// Parse run mode from string. Returns Ensemble by default.
inline RunMode parse_run_mode(const std::string& s) {
    if (s == "single_run" || s == "single")
        return RunMode::SingleRun;
    if (s == "analysis_only" || s == "analysis")
        return RunMode::AnalysisOnly;
    return RunMode::Ensemble; // default
}

/**
 * @brief Pin configuration for flow solver
 *
 * Configures how the solver handles singular systems (all periodic/Neumann).
 * See pin_spec.hpp for full documentation on the pin mechanism.
 */
struct PinConfig {
    PinMode mode = PinMode::Auto; // auto | on | off
};

/**
 * @brief Grid configuration
 */
struct GridConfig {
    int nx = 64;
    int ny = 64;
    int nz = 64;
    real dx = 1.0; // Isotropic: dy = dz = dx

    // Computed domain size
    real Lx() const { return nx * dx; }
    real Ly() const { return ny * dx; }
    real Lz() const { return nz * dx; }
};

/**
 * @brief Stochastic K field configuration
 */
struct StochasticYamlConfig {
    real sigma2 = 1.0;       // Variance of log-K
    real corr_length = 1.0;  // Correlation length
    int n_modes = 1000;      // Number of Fourier modes
    int covariance_type = 0; // 0 = exponential, 1 = gaussian
    uint64_t seed = 12345;   // RNG seed
    real K_mean = 1.0;       // Geometric mean of K
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
    int cg_check_every = 10; // Check convergence every N iterations

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
    struct PsptaRefineConfig {
        bool enabled = false;
        int outer_iters = 5;
        real omega = static_cast<real>(0.5);
        real omega_min = static_cast<real>(1.0e-6);
        int max_backtracks = 18;
        real eps_vx = static_cast<real>(1.0e-10);
        real source_clip_cells = static_cast<real>(0.1);
        int no_descent_patience = 4;
        real stop_rel_rms = static_cast<real>(0.25);
        real stop_abs_rms = static_cast<real>(1.0e-6);
        bool print_every_iter = true;
        bool save_history_csv = true;
        bool eq13_diagnostics = false;
    };

    int n_particles = 10000;
    real dt = 0.01;
    int n_steps = 1000;
    real porosity = 1.0;
    real diffusion = 0.0; // Molecular diffusion (Dm) [L²/T]
    real alpha_l = 0.0;   // Longitudinal dispersivity [L]
    real alpha_t = 0.0;   // Transverse dispersivity [L]
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

    // Optional post-seed refinement of ψ invariants (method=="pspta" only).
    PsptaRefineConfig pspta_refine;
};

/**
 * @brief Macrodispersion analysis configuration
 */
struct MacrodispersionConfig {
    bool enabled = false;
    int NR = 1;                           // Number of realizations
    real lambda = 1.0;                    // Correlation length for alpha
    real vmean_norm = 1.0;                // ||<v>|| (provided; could be computed from flow)
    int sample_every = 10;                // Stats sampling frequency (transport steps)
    std::string var_estimator = "biased"; // "biased" (paper) or "unbiased" (Par2_Core raw)
};

/**
 * @brief Snapshot configuration (using Par2_Core CsvSnapshotWriter)
 */
struct SnapshotConfig {
    bool enabled = false;
    int every = 200; // Steps between snapshots
    bool legacy_format = true;
    bool include_time = false;
    bool include_status = false;
    bool include_wrap_counts = false;
    bool include_unwrapped = false;
    int stride = 1;
    int max_particles = -1; // -1 = no limit
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
    bool velocity_field = false; ///< Run divergence/vorticity/helicity diagnostics
};

/**
 * @brief Affine mean-velocity source configuration for the streamfunction
 *        solver (SF-16 T01).
 *
 * `mode == "fixed"` uses `value` directly as `vbar` in the SF-06 affine
 * gauge; `mode == "measured"` is reserved for a later increment to source
 * `vbar` from the measured Darcy field. This T01 node only parses and
 * validates the surface; nothing reads it yet.
 */
struct StreamfunctionAffineMeanVelocityConfig {
    std::string mode = "fixed"; // "fixed" | "measured"
    real value = 1.0;
};

/**
 * @brief Fixed-relaxation Picard iteration limits (mirrors
 *        `streamfunctions::FixedPicardConfig`, SF-14).
 */
struct StreamfunctionPicardYamlConfig {
    int max_iter = 500;
    real tolerance = 1.0e-6;
    real omega = 0.25;
};

/**
 * @brief Minimal pipeline-facing slice of `streamfunctions::AdaptivePicardConfig`
 *        (SF-15): only the on/off switch is exposed here.
 */
struct StreamfunctionAdaptiveYamlConfig {
    bool enabled = true;
};

/**
 * @brief Pipeline-facing slice of `solvers::ProjectedPCGConfig` used by the
 *        streamfunction linear subproblem.
 */
struct StreamfunctionLinearYamlConfig {
    real rtol = 1.0e-10;
    int max_iter = 1000;
    int check_every = 10;
};

/**
 * @brief Pipeline-facing slice of `multigrid::MGConfig` used by the
 *        streamfunction linear subproblem.
 */
struct StreamfunctionMgYamlConfig {
    int num_levels = 4;
};

/**
 * @brief Streamfunction-solver export/output switches (wired in a later
 *        SF-16 node). YAML key is "export"; the member is named `exports`
 *        because `export` is a reserved C++ keyword.
 */
struct StreamfunctionExportConfig {
    bool iteration_history = true;
    bool summary = true;
    bool fields = false; // raw double u1/u2 dumps
};

/**
 * @brief Eta-axis continuation stepper parameters (SF-17 T03), mirroring
 *        `streamfunctions::ContinuationAxisConfig` for the eta axis.
 *
 * `start` is the eta continuation's linear-space starting point (spec-locked
 * default `0.0`); the axis TARGET is deliberately not duplicated here (SF-17
 * activation bitácora decision 9) — it is the existing top-level
 * `streamfunction_solver.eta` field, so no field is silently ignored.
 */
struct StreamfunctionContinuationEtaYamlConfig {
    real start = 0.0;
    real initial_step = 0.1;
    real min_step = 0.0125;
    real max_step = 0.25;
    real backtrack_factor = 0.5;
    real growth_factor = 1.5;
    int easy_streak = 2;
};

/**
 * @brief Epsilon-axis continuation stepper parameters (SF-17 T03), mirroring
 *        `streamfunctions::ContinuationAxisConfig` for the epsilon axis, in
 *        PHYSICAL epsilon units (the step fields are still decades in
 *        `p = -log10(epsilon)` space, matching the library's
 *        `epsilon_log10` axis).
 *
 * `target` is the physical epsilon continuation TARGET (mapped to
 * `epsilon_log10.target = -log10(target)`); the axis STARTING point is
 * deliberately not duplicated here (decision 9) — it is the existing
 * top-level `streamfunction_solver.epsilon` field.
 */
struct StreamfunctionContinuationEpsilonYamlConfig {
    real target = 1.0e-6;
    real initial_step_log10 = 1.0;
    real min_step_log10 = 0.125;
    real max_step_log10 = 1.0;
    real backtrack_factor = 0.5;
    real growth_factor = 1.5;
    int easy_streak = 2;
};

/**
 * @brief Strict pipeline surface for the SF-17 eta/epsilon continuation
 *        controller. `enabled == false` (the default) means the SF-16
 *        single-solve path runs byte-identically, regardless of whether this
 *        subsection (or any of its nested fields) is present in the YAML.
 */
struct StreamfunctionContinuationYamlConfig {
    bool enabled = false;
    StreamfunctionContinuationEtaYamlConfig eta;
    StreamfunctionContinuationEpsilonYamlConfig epsilon;
};

/**
 * @brief Strict, minimal pipeline configuration surface for the Lester
 *        equation (14) streamfunction solver (SF-16 T01, extended by SF-17
 *        T03 with the `continuation` subsection).
 *
 * This maps onto `streamfunctions::StreamfunctionSolverConfig` (see
 * `src/physics/streamfunctions/StreamfunctionTypes.hpp`) in a later
 * increment (SF-16 T02); it intentionally does NOT expose the full SF-15
 * `AdaptivePicardConfig` field set (only `adaptive.enabled`) and does NOT
 * expose any Anderson/Newton keys.
 *
 * `enabled == false` (the default) means the section, and every nested
 * subsection, may be entirely absent from the YAML with zero behavior
 * change: nothing reads this struct at runtime yet.
 */
struct StreamfunctionSolverYamlConfig {
    bool enabled = false;
    StreamfunctionAffineMeanVelocityConfig affine_mean_velocity;
    real epsilon = 1.0e-2;
    real eta = 1.0;
    StreamfunctionPicardYamlConfig picard;
    StreamfunctionAdaptiveYamlConfig adaptive;
    StreamfunctionLinearYamlConfig linear;
    StreamfunctionMgYamlConfig mg;
    StreamfunctionExportConfig exports;
    StreamfunctionContinuationYamlConfig continuation;
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
    std::string format = "binary"; // "binary" or "vtk"
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
    StreamfunctionSolverYamlConfig streamfunction_solver;

    // Validation helpers
    bool is_valid() const { return grid.nx > 0 && grid.ny > 0 && grid.nz > 0 && grid.dx > 0; }
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
