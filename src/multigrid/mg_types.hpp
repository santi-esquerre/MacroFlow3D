#pragma once

#include "../core/Grid3D.hpp"
#include "../core/DeviceBuffer.cuh"
#include "../core/Scalar.hpp"
#include <vector>

namespace macroflow3d {
namespace multigrid {

/**
 * @brief Multigrid configuration parameters.
 * @ingroup multigrid
 */
// MG configuration
struct MGConfig {
    int num_levels = 4;
    int pre_smooth = 2;
    int post_smooth = 2;
    int coarse_solve_iters = 50;  // GSRB iterations on coarsest level
    int check_convergence_every = 1;  // Check residual norm every N cycles (1 = every cycle)
    real omega = 1.0;  // Relaxation parameter (if needed)
    bool verbose = false;
};

// Single MG level with buffers
struct MGLevel {
    Grid3D grid;
    
    // Solution/correction at this level
    DeviceBuffer<real> x;
    
    // RHS at this level
    DeviceBuffer<real> b;
    
    // Residual at this level
    DeviceBuffer<real> r;
    
    // Conductivity field (if heterogeneous, otherwise can be uniform)
    DeviceBuffer<real> K;
    
    MGLevel() = default;
    
    explicit MGLevel(const Grid3D& g) : grid(g) {
        size_t n = g.num_cells();
        x.resize(n);
        b.resize(n);
        r.resize(n);
        K.resize(n);
    }
    
    void ensure(const Grid3D& g) {
        grid = g;
        size_t n = g.num_cells();
        // Guarantee exact sizes (not just minimum)
        if (x.size() != n) x.resize(n);
        if (b.size() != n) b.resize(n);
        if (r.size() != n) r.resize(n);
        if (K.size() != n) K.resize(n);
    }
};

// MG hierarchy (multiple levels)
struct MGHierarchy {
    std::vector<MGLevel> levels;
    
    MGHierarchy() = default;
    
    // Construct hierarchy from finest grid
    // Coarsening rule: divide by 2 in each dimension
    explicit MGHierarchy(const Grid3D& finest, int num_levels) {
        levels.reserve(num_levels);
        
        Grid3D current = finest;
        for (int l = 0; l < num_levels; ++l) {
            levels.emplace_back(current);
            
            // Coarsen for next level (if not last)
            if (l < num_levels - 1) {
                // Divide by 2 (cell-centered MG)
                current.nx = current.nx / 2;
                current.ny = current.ny / 2;
                current.nz = current.nz / 2;
                current.dx = current.dx * 2.0;
                current.dy = current.dy * 2.0;
                current.dz = current.dz * 2.0;
                
                // Sanity check
                if (current.nx < 2 || current.ny < 2 || current.nz < 2) {
                    break;  // Can't coarsen further
                }
            }
        }
    }
    
    int num_levels() const { return static_cast<int>(levels.size()); }
    
    const Grid3D& finest_grid() const { return levels[0].grid; }
    const Grid3D& coarsest_grid() const { return levels.back().grid; }
};

// MG workspace for temporary buffers (if needed beyond what's in MGLevel)
struct MGWorkspace {
    // Currently empty - all buffers are in MGLevel
    // Can add scratch buffers here if transfer/smooth need them
};

} // namespace multigrid
} // namespace macroflow3d
