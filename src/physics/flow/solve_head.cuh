#pragma once

/**
 * @file solve_head.cuh
 * @brief Solve head (Darcy flow) equation using MG, CG, or PCG
 * 
 * Legacy correspondence: main_transport_JSON_input.cu solver calls
 * 
 * Solves: -∇·(K∇h) = 0  with boundary conditions
 * 
 * The operator uses harmonic mean for K at faces, matching legacy.
 * 
 * ## Solver Types
 * 
 * - CG: Plain conjugate gradient (no preconditioner)
 * - MG: Standalone multigrid V-cycles
 * - PCG_MG: CG preconditioned with MG V-cycle (legacy default: solver_CG + PCCMG_CG)
 * 
 * ## Pin for Singular Systems
 * 
 * When all boundaries are periodic or Neumann, the system is singular.
 * The pin mechanism (diagonal doubling at cell [0,0,0]) breaks this degeneracy.
 * See pin_spec.hpp for full documentation.
 */

#include "../../core/Grid3D.hpp"
#include "../../core/BCSpec.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/pin_spec.hpp"
#include "../../io/config/Config.hpp"
#include "../common/workspaces.cuh"

namespace macroflow3d {
namespace physics {

// Re-export for convenience in physics layer
using macroflow3d::io::PinConfig;
using macroflow3d::PinMode;

/**
 * @brief Solver type for head equation
 * 
 * Legacy correspondence:
 * - CG: plain conjugate gradient (no preconditioner)
 * - MG: standalone multigrid V-cycles
 * - PCG_MG: CG preconditioned with MG V-cycle (legacy default: solver_CG + PCCMG_CG)
 */
enum class HeadSolverType {
    CG,      // Plain CG (no preconditioner)
    MG,      // Standalone MG V-cycles
    PCG_MG   // CG preconditioned with MG (legacy default)
};

/**
 * @brief Configuration for head equation solve
 * 
 * Groups all parameters needed to solve the head equation.
 * Can be constructed from FlowYamlConfig or set directly.
 */
struct HeadSolveConfig {
    // Solver type (default: PCG_MG to match legacy)
    HeadSolverType solver_type = HeadSolverType::PCG_MG;
    
    // MG parameters (used for MG and PCG_MG)
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_max_cycles = 20;      // Max V-cycles for standalone MG
    int mg_coarse_iters = 50;
    
    // CG/PCG parameters
    int cg_max_iter = 1000;      // Max CG iterations
    int cg_check_every = 10;     // Check convergence every N iterations
    real cg_rtol = 1e-8;         // CG relative tolerance
    
    // Overall convergence
    real rtol = 1e-6;
    
    // Pin configuration for singular systems
    // See pin_spec.hpp for documentation
    PinConfig pin;
    
    HeadSolveConfig() = default;
    
    /**
     * @brief Factory: Create HeadSolveConfig from FlowYamlConfig
     * 
     * Converts the string-based solver type to enum and copies all parameters.
     * This centralizes the IO→numerics conversion logic.
     * 
     * @param flow_cfg FlowYamlConfig from YAML parsing
     * @return HeadSolveConfig ready for use by solve_head()
     */
    static HeadSolveConfig from_yaml(const macroflow3d::io::FlowYamlConfig& flow_cfg) {
        HeadSolveConfig cfg;
        
        // Parse solver type from string
        if (flow_cfg.solver == "cg") {
            cfg.solver_type = HeadSolverType::CG;
        } else if (flow_cfg.solver == "mg") {
            cfg.solver_type = HeadSolverType::MG;
        } else if (flow_cfg.solver == "pcg_mg" || flow_cfg.solver == "mg_cg") {
            cfg.solver_type = HeadSolverType::PCG_MG;
        } else {
            // Default to PCG_MG for unknown strings
            cfg.solver_type = HeadSolverType::PCG_MG;
        }
        
        // Copy MG parameters
        cfg.mg_levels = flow_cfg.mg_levels;
        cfg.mg_pre_smooth = flow_cfg.mg_pre_smooth;
        cfg.mg_post_smooth = flow_cfg.mg_post_smooth;
        cfg.mg_max_cycles = flow_cfg.mg_max_cycles;
        cfg.mg_coarse_iters = flow_cfg.mg_coarse_iters;
        
        // Copy CG parameters
        cfg.cg_max_iter = flow_cfg.cg_max_iter;
        cfg.cg_rtol = flow_cfg.cg_rtol;
        cfg.cg_check_every = flow_cfg.cg_check_every;
        
        // Copy convergence and pin
        cfg.rtol = flow_cfg.rtol;
        cfg.pin = flow_cfg.pin;
        
        return cfg;
    }
};

/**
 * @brief Result of head solve
 */
struct HeadSolveResult {
    int num_iterations = 0;      // MG cycles or CG iterations
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

/**
 * @brief Solve head equation: -∇·(K∇h) = 0 with BCs
 * 
 * Uses MG by default. The RHS is zero (no sources).
 * Dirichlet BCs set the head at boundaries.
 * 
 * @param h         Output: head field (device, size = num_cells)
 * @param K         Input: conductivity field (device, size = num_cells)
 * @param grid      Grid specification
 * @param bc        Boundary conditions
 * @param cfg       Solver configuration
 * @param ctx       CUDA context
 * @param workspace Flow workspace (contains MG hierarchy)
 * @return HeadSolveResult with convergence info
 */
HeadSolveResult solve_head(
    DeviceSpan<real> h,
    DeviceSpan<const real> K,
    const Grid3D& grid,
    const BCSpec& bc,
    const HeadSolveConfig& cfg,
    CudaContext& ctx,
    FlowWorkspace& workspace
);

/**
 * @brief Initialize initial guess for head
 * 
 * Default: linear interpolation between Dirichlet BCs in x-direction.
 * Falls back to zero if no Dirichlet BCs.
 * 
 * @param h    Output: initial guess
 * @param grid Grid
 * @param bc   Boundary conditions
 * @param ctx  CUDA context
 */
void init_head_guess(
    DeviceSpan<real> h,
    const Grid3D& grid,
    const BCSpec& bc,
    const CudaContext& ctx
);

} // namespace physics
} // namespace macroflow3d
