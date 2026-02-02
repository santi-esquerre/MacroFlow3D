/**
 * @file rwpt_flow_transport.cu
 * @brief Main application for RWPT flow + transport simulation (Etapa 5)
 * 
 * Pipeline:
 *   1. Load config from YAML
 *   2. Generate stochastic K field
 *   3. Solve head equation (MG or CG)
 *   4. Compute velocity from head
 *   5. Run particle transport
 *   6. Compute statistics / output
 * 
 * This file is the skeleton; actual computation added in subsequent tasks.
 */

#include "../src/io/config/Config.hpp"
#include "../src/numerics/pin_spec.hpp"  // pin_enabled, needs_pin, pin_mode_str
#include "../src/runtime/CudaContext.cuh"
#include "../src/core/Grid3D.hpp"
#include "../src/physics/common/physics_types.cuh"
#include "../src/physics/stochastic/stochastic.cuh"
#include "../src/physics/flow/solve_head.cuh"
#include "../src/physics/flow/velocity_from_head.cuh"
#include <iostream>
#include <iomanip>
#include <string>

using namespace rwpt;
using namespace rwpt::io;
using namespace rwpt::physics;

// Helper to convert BC type to string
const char* bc_type_str(BCType t) {
    switch (t) {
        case BCType::Dirichlet: return "Dirichlet";
        case BCType::Neumann:   return "Neumann";
        case BCType::Periodic:  return "Periodic";
        default:                return "Unknown";
    }
}

void print_config_summary(const AppConfig& cfg) {
    std::cout << "=== Configuration Summary ===\n\n";
    
    // Grid
    std::cout << "Grid:\n";
    std::cout << "  Dimensions: " << cfg.grid.nx << " x " << cfg.grid.ny << " x " << cfg.grid.nz << "\n";
    std::cout << "  Cell size:  dx = " << cfg.grid.dx << "\n";
    std::cout << "  Domain:     [0, " << cfg.grid.Lx() << "] x [0, " << cfg.grid.Ly() 
              << "] x [0, " << cfg.grid.Lz() << "]\n";
    std::cout << "  Total cells: " << (cfg.grid.nx * cfg.grid.ny * cfg.grid.nz) << "\n\n";
    
    // Stochastic
    std::cout << "Stochastic K:\n";
    std::cout << "  sigma^2:     " << cfg.stochastic.sigma2 << "\n";
    std::cout << "  corr_length: " << cfg.stochastic.corr_length << "\n";
    std::cout << "  n_modes:     " << cfg.stochastic.n_modes << "\n";
    std::cout << "  seed:        " << cfg.stochastic.seed << "\n\n";
    
    // Flow
    std::cout << "Flow solver:\n";
    std::cout << "  Solver:      " << cfg.flow.solver << "\n";
    std::cout << "  MG levels:   " << cfg.flow.mg_levels << "\n";
    std::cout << "  Tolerance:   " << cfg.flow.rtol << "\n";
    
    // Pin config
    bool pin_is_enabled = pin_enabled(cfg.flow.pin.mode, cfg.flow.bc);
    std::cout << "  Pin mode:    " << pin_mode_str(cfg.flow.pin.mode);
    std::cout << " (needs_pin=" << (needs_pin(cfg.flow.bc) ? "yes" : "no");
    std::cout << ", enabled=" << (pin_is_enabled ? "YES" : "no") << ")\n";
    if (pin_is_enabled) {
        std::cout << "    (diagonal doubling at cell [0,0,0])\n";
    }
    
    std::cout << "  BCs (west/east=x, south/north=y, bottom/top=z):\n";
    std::cout << "    west(xmin):   " << bc_type_str(cfg.flow.bc.xmin.type) << " = " << cfg.flow.bc.xmin.value << "\n";
    std::cout << "    east(xmax):   " << bc_type_str(cfg.flow.bc.xmax.type) << " = " << cfg.flow.bc.xmax.value << "\n";
    std::cout << "    south(ymin):  " << bc_type_str(cfg.flow.bc.ymin.type) << " = " << cfg.flow.bc.ymin.value << "\n";
    std::cout << "    north(ymax):  " << bc_type_str(cfg.flow.bc.ymax.type) << " = " << cfg.flow.bc.ymax.value << "\n";
    std::cout << "    bottom(zmin): " << bc_type_str(cfg.flow.bc.zmin.type) << " = " << cfg.flow.bc.zmin.value << "\n";
    std::cout << "    top(zmax):    " << bc_type_str(cfg.flow.bc.zmax.type) << " = " << cfg.flow.bc.zmax.value << "\n\n";
    
    // Transport
    std::cout << "Transport:\n";
    std::cout << "  n_particles: " << cfg.transport.n_particles << "\n";
    std::cout << "  dt:          " << cfg.transport.dt << "\n";
    std::cout << "  n_steps:     " << cfg.transport.n_steps << "\n";
    std::cout << "  Total time:  " << (cfg.transport.dt * cfg.transport.n_steps) << "\n";
    std::cout << "  porosity:    " << cfg.transport.porosity << "\n";
    std::cout << "  diffusion:   " << cfg.transport.diffusion << "\n\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=== RWPT Flow + Transport Simulator ===\n";
    std::cout << "Etapa 5 - Physics Pipeline\n\n";
    
    try {
        // 1. Parse command line
        std::string config_path = "apps/config_example.yaml";
        if (argc > 1) {
            config_path = argv[1];
        }
        std::cout << "Config file: " << config_path << "\n\n";
        
        // 2. Load configuration
        AppConfig cfg = load_config_yaml(config_path);
        print_config_summary(cfg);
        
        // 3. Initialize CUDA context
        std::cout << "Initializing CUDA...\n";
        CudaContext ctx(0);
        std::cout << "  Device ready.\n\n";
        
        // 4. Create Grid3D from config
        Grid3D grid(cfg.grid.nx, cfg.grid.ny, cfg.grid.nz,
                    cfg.grid.dx, cfg.grid.dx, cfg.grid.dx);  // Isotropic
        
        // 5. Allocate fields (Etapa 5.0 types)
        std::cout << "Allocating fields...\n";
        
        KField K(grid);
        std::cout << "  K field:        " << K.size() << " cells ("
                  << (K.size() * sizeof(real) / 1024.0 / 1024.0) << " MB)\n";
        
        HeadField h(grid);
        std::cout << "  Head field:     " << h.size() << " cells\n";
        
        VelocityField vel(grid);
        std::cout << "  Velocity field: " << vel.total_size() << " faces ("
                  << "U:" << vel.size_U() << ", V:" << vel.size_V() << ", W:" << vel.size_W() << ")\n";
        
        // 6. Allocate workspaces
        std::cout << "\nAllocating workspaces...\n";
        
        // Convert config types
        StochasticConfig stoch_cfg;
        stoch_cfg.sigma2 = cfg.stochastic.sigma2;
        stoch_cfg.corr_length = cfg.stochastic.corr_length;
        stoch_cfg.n_modes = cfg.stochastic.n_modes;
        stoch_cfg.covariance_type = cfg.stochastic.covariance_type;
        stoch_cfg.seed = cfg.stochastic.seed;
        
        StochasticWorkspace stoch_ws;
        stoch_ws.allocate(grid, stoch_cfg);
        std::cout << "  Stochastic workspace: " << stoch_ws.n_modes << " modes, "
                  << stoch_ws.n_cells << " cells\n";
        
        FlowWorkspace flow_ws;
        flow_ws.allocate(grid);
        std::cout << "  Flow workspace:       " << flow_ws.n_cells << " cells\n";
        
        TransportConfig trans_cfg;
        trans_cfg.n_particles = cfg.transport.n_particles;
        trans_cfg.dt = cfg.transport.dt;
        trans_cfg.n_steps = cfg.transport.n_steps;
        
        ParticlesWorkspace part_ws;
        part_ws.allocate(trans_cfg);
        std::cout << "  Particles workspace:  " << part_ws.n_particles << " particles\n";
        
        // Sync and report memory
        ctx.synchronize();
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "\nGPU memory: " << std::fixed << std::setprecision(1)
                  << ((total_mem - free_mem) / 1024.0 / 1024.0) << " MB used, "
                  << (free_mem / 1024.0 / 1024.0) << " MB free\n";
        
        // ====================================================================
        // PIPELINE: Generate stochastic K field (Task 5.2)
        // ====================================================================
        std::cout << "\n=== STEP 1: Generate K field ===\n";
        std::cout << "  Covariance type: " << (stoch_cfg.covariance_type == 0 ? "exponential" : "gaussian") << "\n";
        std::cout << "  sigma^2 = " << stoch_cfg.sigma2 << ", corr_length = " << stoch_cfg.corr_length << "\n";
        std::cout << "  n_modes = " << stoch_cfg.n_modes << ", seed = " << stoch_cfg.seed << "\n";
        
        // Generate K field
        generate_K_field(
            DeviceSpan<real>(K.device_ptr(), K.size()),
            stoch_ws,
            grid,
            stoch_cfg,
            ctx
        );
        ctx.synchronize();
        
        // Compute and print statistics
        real K_min, K_max, K_mean;
        compute_field_stats(
            DeviceSpan<const real>(K.device_ptr(), K.size()),
            K_min, K_max, K_mean,
            ctx
        );
        
        std::cout << "\n  K field statistics:\n";
        std::cout << "    min(K)  = " << K_min << "\n";
        std::cout << "    max(K)  = " << K_max << "\n";
        std::cout << "    mean(K) = " << K_mean << "\n";
        std::cout << "    ratio   = " << (K_max / K_min) << "\n";
        
        // Also compute logK stats for comparison
        real logK_min, logK_max, logK_mean;
        compute_field_stats(
            DeviceSpan<const real>(stoch_ws.logK.data(), stoch_ws.n_cells),
            logK_min, logK_max, logK_mean,
            ctx
        );
        std::cout << "\n  logK (Gaussian) statistics:\n";
        std::cout << "    min(logK)  = " << logK_min << "\n";
        std::cout << "    max(logK)  = " << logK_max << "\n";
        std::cout << "    mean(logK) = " << logK_mean << " (should be ~0)\n";
        std::cout << "    Expected Var[logK] ~= " << stoch_cfg.sigma2 << "\n";
        
        // ====================================================================
        // PIPELINE: Solve head equation (Task 5.3)
        // ====================================================================
        std::cout << "\n=== STEP 2: Solve head equation ===\n";
        std::cout << "  Solver: " << cfg.flow.solver << "\n";
        std::cout << "  BCs: west=" << cfg.flow.bc.xmin.value << " (Dirichlet), east=" 
                  << cfg.flow.bc.xmax.value << " (Dirichlet)\n";
        
        // Build HeadSolveConfig from FlowYamlConfig using factory method
        HeadSolveConfig head_cfg = HeadSolveConfig::from_yaml(cfg.flow);
        
        // Allocate MG hierarchy in workspace (now with MG levels)
        flow_ws.allocate(grid, head_cfg.mg_levels);
        
        // Solve head
        HeadSolveResult head_result = solve_head(
            DeviceSpan<real>(h.device_ptr(), h.size()),
            DeviceSpan<const real>(K.device_ptr(), K.size()),
            grid,
            cfg.flow.bc,  // BCSpec from config
            head_cfg,
            ctx,
            flow_ws
        );
        ctx.synchronize();
        
        std::cout << "\n  Solve result:\n";
        std::cout << "    Converged:        " << (head_result.converged ? "YES" : "NO") << "\n";
        std::cout << "    Iterations:       " << head_result.num_iterations << "\n";
        std::cout << std::scientific << std::setprecision(3);
        std::cout << "    Initial residual: " << head_result.initial_residual << "\n";
        std::cout << "    Final residual:   " << head_result.final_residual << "\n";
        if (head_result.initial_residual > 0) {
            std::cout << "    Reduction:        " << (head_result.final_residual / head_result.initial_residual) << "\n";
        }
        std::cout << std::fixed << std::setprecision(1);  // Reset to default
        
        // Head statistics
        real h_min, h_max, h_mean;
        compute_field_stats(
            DeviceSpan<const real>(h.device_ptr(), h.size()),
            h_min, h_max, h_mean,
            ctx
        );
        
        std::cout << "\n  Head field statistics:\n";
        std::cout << "    min(h)  = " << h_min << "\n";
        std::cout << "    max(h)  = " << h_max << "\n";
        std::cout << "    mean(h) = " << h_mean << "\n";
        
        // ====================================================================
        // PIPELINE: Compute velocity from head (Task 5.4)
        // ====================================================================
        std::cout << "\n=== STEP 3: Compute velocity from head ===\n";
        std::cout << "  Using Darcy's law with harmonic mean conductivity\n";
        
        // Compute U, V, W from H and K
        compute_velocity_from_head(vel, h, K, grid, cfg.flow.bc, ctx);
        ctx.synchronize();
        
        // Print checksums
        print_velocity_checksums(vel, ctx);
        
        // Verify mean velocity against theoretical Darcy (if enabled and Dirichlet)
        if (cfg.flow.verify_velocity &&
            cfg.flow.bc.xmin.type == BCType::Dirichlet && 
            cfg.flow.bc.xmax.type == BCType::Dirichlet) {
            verify_mean_velocity_darcy(vel, K, grid, cfg.flow.bc, ctx);
        }
        
        // ====================================================================
        // PLACEHOLDER: Remaining pipeline steps (Tasks 5.5+)
        // ====================================================================
        std::cout << "\n";
        std::cout << "--- Remaining pipeline steps (not implemented yet) ---\n";
        std::cout << "[TODO] 5.5: Run particle transport\n";
        std::cout << "[TODO] 5.6: Compute statistics & output\n";
        std::cout << "-----------------------------------------------------\n\n";
        
        std::cout << "=== OK: K generated, head solved, velocity computed ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
