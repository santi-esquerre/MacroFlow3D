#pragma once

/**
 * @file workspaces.cuh
 * @brief Pre-allocated workspace buffers for physics modules
 *
 * Each workspace owns temporary GPU memory needed by its module.
 * Allocate once before simulation loop, reuse across steps.
 * No cudaMalloc inside inner loops.
 *
 * Pattern:
 *   Workspace ws;
 *   ws.allocate(grid, config);  // Once at setup
 *   for (step : steps) {
 *       module_function(ws, ...);  // Uses pre-allocated buffers
 *   }
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/solvers/cg_types.hpp"
#include "../../numerics/solvers/pcg.cuh"
#include "physics_config.hpp"
#include <curand_kernel.h>

namespace macroflow3d {
namespace physics {

// ============================================================================
// Stochastic K generation workspace
// ============================================================================

/**
 * @brief Workspace for stochastic K field generation
 *
 * Legacy correspondence: random_field_generation.cu
 * Buffers for Fourier mode coefficients (k1, k2, k3, a, b, vartheta)
 * and RNG states.
 */
struct StochasticWorkspace {
    // Fourier mode coefficients (size = n_modes)
    DeviceBuffer<real> k1;       ///< Wavenumber component x
    DeviceBuffer<real> k2;       ///< Wavenumber component y
    DeviceBuffer<real> k3;       ///< Wavenumber component z (3D only)
    DeviceBuffer<real> coef_a;   ///< Coefficient a for spectral sum
    DeviceBuffer<real> coef_b;   ///< Coefficient b for spectral sum
    DeviceBuffer<real> vartheta; ///< Phase angles

    // RNG states (one per mode for parallel generation)
    DeviceBuffer<curandState> rng_states;

    // Intermediate buffer for log(K) before exp()
    DeviceBuffer<real> logK;

    // Allocated sizes
    int n_modes = 0;
    size_t n_cells = 0;

    StochasticWorkspace() = default;

    // Allocate for given config and grid
    void allocate(const Grid3D& grid, const StochasticConfig& cfg) {
        n_modes = cfg.n_modes;
        n_cells = grid.num_cells();

        // Mode coefficients
        k1.resize(n_modes);
        k2.resize(n_modes);
        k3.resize(n_modes);
        coef_a.resize(n_modes);
        coef_b.resize(n_modes);
        vartheta.resize(n_modes);

        // RNG states
        rng_states.resize(n_modes);

        // Intermediate logK field
        logK.resize(n_cells);
    }

    // Check if allocated
    bool is_allocated() const { return n_modes > 0 && n_cells > 0; }

    // Clear (free memory)
    void clear() {
        k1 = DeviceBuffer<real>();
        k2 = DeviceBuffer<real>();
        k3 = DeviceBuffer<real>();
        coef_a = DeviceBuffer<real>();
        coef_b = DeviceBuffer<real>();
        vartheta = DeviceBuffer<real>();
        rng_states = DeviceBuffer<curandState>();
        logK = DeviceBuffer<real>();
        n_modes = 0;
        n_cells = 0;
    }
};

// ============================================================================
// Flow solver workspace
// ============================================================================

/**
 * @brief Workspace for flow (head) solver
 *
 * Contains temporary buffers for MG and CG solvers.
 * Includes MG hierarchy for multigrid solve.
 */
struct FlowWorkspace {
    // Residual and auxiliary vectors for iterative solvers
    DeviceBuffer<real> residual;
    DeviceBuffer<real> aux1;
    DeviceBuffer<real> aux2;

    // RHS buffer (can be modified during solve)
    DeviceBuffer<real> rhs;

    // MG hierarchy (allocated on first use)
    multigrid::MGHierarchy mg_hierarchy;

    // CG solver workspace (for HeadSolverType::CG)
    solvers::CGWorkspace cg_workspace;

    // PCG solver workspace (for HeadSolverType::PCG_MG)
    solvers::PCGWorkspace pcg_workspace;

    // Size tracking
    size_t n_cells = 0;

    FlowWorkspace() = default;

    // Allocate for given grid
    void allocate(const Grid3D& grid) {
        n_cells = grid.num_cells();
        residual.resize(n_cells);
        aux1.resize(n_cells);
        aux2.resize(n_cells);
        rhs.resize(n_cells);
        // mg_hierarchy is allocated on-demand in solve_head
        // cg_workspace/pcg_workspace are allocated on-demand in solve_head
    }

    // Allocate with MG hierarchy (num_levels)
    void allocate(const Grid3D& grid, int mg_levels) {
        allocate(grid);
        if (mg_levels > 0) {
            mg_hierarchy = multigrid::MGHierarchy(grid, mg_levels);
        }
    }

    // Check if allocated
    bool is_allocated() const { return n_cells > 0; }

    // Clear (free memory)
    void clear() {
        residual = DeviceBuffer<real>();
        aux1 = DeviceBuffer<real>();
        aux2 = DeviceBuffer<real>();
        rhs = DeviceBuffer<real>();
        mg_hierarchy = multigrid::MGHierarchy();
        cg_workspace = solvers::CGWorkspace();
        pcg_workspace = solvers::PCGWorkspace();
        n_cells = 0;
    }
};

// ============================================================================
// Particle transport workspace
// ============================================================================

/**
 * @brief Workspace for particle transport (RWPT/PAR2)
 *
 * Contains particle state arrays and temporary buffers for stepping.
 * Position arrays are (x, y, z) for each particle.
 */
struct ParticlesWorkspace {
    // Particle positions (size = n_particles each)
    DeviceBuffer<real> x;
    DeviceBuffer<real> y;
    DeviceBuffer<real> z;

    // Previous positions (for reflection BCs, optional)
    DeviceBuffer<real> x_prev;
    DeviceBuffer<real> y_prev;
    DeviceBuffer<real> z_prev;

    // Particle status (0=active, 1=exited, etc.)
    DeviceBuffer<int> status;

    // RNG states for diffusion (one per particle)
    DeviceBuffer<curandState> rng_states;

    // Temporary buffers for velocity interpolation
    DeviceBuffer<real> u_interp;
    DeviceBuffer<real> v_interp;
    DeviceBuffer<real> w_interp;

    // Size tracking
    int n_particles = 0;

    ParticlesWorkspace() = default;

    // Allocate for given config
    void allocate(const TransportConfig& cfg) {
        n_particles = cfg.n_particles;

        // Positions
        x.resize(n_particles);
        y.resize(n_particles);
        z.resize(n_particles);

        // Previous positions
        x_prev.resize(n_particles);
        y_prev.resize(n_particles);
        z_prev.resize(n_particles);

        // Status
        status.resize(n_particles);

        // RNG states
        rng_states.resize(n_particles);

        // Interpolation temporaries
        u_interp.resize(n_particles);
        v_interp.resize(n_particles);
        w_interp.resize(n_particles);
    }

    // Check if allocated
    bool is_allocated() const { return n_particles > 0; }

    // Number of active particles (requires host sync - use sparingly)
    // int count_active() const;  // TODO: implement with reduction

    // Clear (free memory)
    void clear() {
        x = DeviceBuffer<real>();
        y = DeviceBuffer<real>();
        z = DeviceBuffer<real>();
        x_prev = DeviceBuffer<real>();
        y_prev = DeviceBuffer<real>();
        z_prev = DeviceBuffer<real>();
        status = DeviceBuffer<int>();
        rng_states = DeviceBuffer<curandState>();
        u_interp = DeviceBuffer<real>();
        v_interp = DeviceBuffer<real>();
        w_interp = DeviceBuffer<real>();
        n_particles = 0;
    }
};

// ============================================================================
// Combined simulation workspace
// ============================================================================

/**
 * @brief Combined workspace for full simulation
 *
 * Convenience container for all workspaces.
 */
struct SimulationWorkspace {
    StochasticWorkspace stochastic;
    FlowWorkspace flow;
    ParticlesWorkspace particles;

    SimulationWorkspace() = default;

    // Allocate all workspaces
    void allocate(const Grid3D& grid, const SimulationConfig& cfg) {
        stochastic.allocate(grid, cfg.stochastic);
        flow.allocate(grid);
        particles.allocate(cfg.transport);
    }

    // Clear all
    void clear() {
        stochastic.clear();
        flow.clear();
        particles.clear();
    }
};

} // namespace physics
} // namespace macroflow3d
