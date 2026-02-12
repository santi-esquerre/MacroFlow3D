#pragma once

/**
 * @file physics_config.hpp
 * @brief Configuration structs for physics modules (POD, no virtuals)
 * 
 * These are lightweight value types for passing parameters between modules.
 * All fields have sensible defaults. No dynamic allocation in constructors.
 */

#include "../../core/Scalar.hpp"
#include <cstdint>

namespace macroflow3d {
namespace physics {

/**
 * @brief Configuration for stochastic K field generation
 * 
 * Legacy correspondence: parameters from random_field_generation.cu
 * K = exp(f(x)) where f is Gaussian random field with given variance/correlation
 */
struct StochasticConfig {
    // Random field parameters
    real sigma2 = 1.0;           ///< Variance of log-conductivity (σ²_f)
    real corr_length = 1.0;      ///< Correlation length (λ)
    int  n_modes = 1000;         ///< Number of Fourier modes for spectral method
    
    // Covariance type: 0 = exponential, 1 = gaussian
    int covariance_type = 0;
    
    // RNG seeds
    uint64_t seed = 12345;       ///< Base seed for RNG
    
    // Geometric mean for normalization (K_g = exp(<ln K>))
    real K_geometric_mean = 1.0; ///< Target geometric mean of K field
    
    // Default constructor with sensible defaults
    StochasticConfig() = default;
};

/**
 * @brief Configuration for flow (head) solver
 * 
 * Legacy correspondence: parameters from main_transport_JSON_input.cu
 */
struct FlowConfig {
    // Solver selection: 0 = MG only, 1 = CG only, 2 = MG-preconditioned CG
    int solver_type = 0;
    
    // MG parameters
    int mg_levels = 4;
    int mg_pre_smooth = 2;
    int mg_post_smooth = 2;
    int mg_coarse_iters = 50;
    int mg_max_cycles = 20;
    
    // CG parameters  
    int cg_max_iter = 1000;
    real cg_rtol = 1e-8;
    real cg_atol = 0.0;
    
    // Convergence
    real rtol = 1e-6;            ///< Relative tolerance for residual
    
    // Physical parameters (for RHS if needed)
    real source_term = 0.0;      ///< Uniform source/sink term
    
    FlowConfig() = default;
};

/**
 * @brief Configuration for particle transport (PAR2/RWPT)
 * 
 * Legacy correspondence: parameters from main_transport_JSON_input.cu
 */
struct TransportConfig {
    // Particle count
    int n_particles = 10000;
    
    // Time stepping
    real dt = 0.01;              ///< Time step size
    int  n_steps = 1000;         ///< Number of time steps
    int  output_every = 100;     ///< Output frequency (steps)
    
    // Physical parameters
    real porosity = 1.0;         ///< Porosity (θ)
    real diffusion = 0.0;        ///< Molecular diffusion coefficient (D_m)
    
    // Injection plane/volume (defaults: inject at x=0 face)
    real inject_xmin = 0.0;
    real inject_xmax = 0.0;      ///< If xmin==xmax, inject on plane
    real inject_ymin = 0.0;
    real inject_ymax = 1.0;
    real inject_zmin = 0.0;
    real inject_zmax = 1.0;
    
    // RNG seed for particle diffusion
    uint64_t seed = 54321;
    
    TransportConfig() = default;
};

/**
 * @brief Combined configuration for full simulation
 */
struct SimulationConfig {
    StochasticConfig stochastic;
    FlowConfig flow;
    TransportConfig transport;
    
    // Domain (informational, actual grid comes from Grid3D)
    real Lx = 1.0, Ly = 1.0, Lz = 1.0;
    int  Nx = 64,  Ny = 64,  Nz = 64;
    
    SimulationConfig() = default;
};

} // namespace physics
} // namespace macroflow3d
