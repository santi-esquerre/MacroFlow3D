#pragma once

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include "../blas/reduction_workspace.cuh"
#include <cstddef>

namespace macroflow3d {
namespace solvers {

struct CGConfig {
    int max_iter = 200;
    real rtol = 1e-6;
    real atol = 0.0;
    int check_every = 10;  // Check convergence every N iters (reduces host sync)
    bool verbose = false;  // Print debug info
};

struct CGResult {
    int iters = 0;
    real r_norm = 0.0;
    real r0_norm = 0.0;  // Initial residual for caller logging
    bool converged = false;
};

struct CGWorkspace {
    // Vector buffers
    DeviceBuffer<real> r;
    DeviceBuffer<real> p;
    DeviceBuffer<real> Ap;
    
    // Device scalars (no host sync in hot-path)
    DeviceBuffer<real> d_rr;
    DeviceBuffer<real> d_rr_new;
    DeviceBuffer<real> d_pAp;
    DeviceBuffer<real> d_alpha;
    DeviceBuffer<real> d_beta;
    
    // Breakdown detection
    DeviceBuffer<int> d_is_valid;
    
    // Reduction workspace for dot/nrm2
    blas::ReductionWorkspace red;
    
    size_t n = 0;
    
    void ensure(size_t required_n) {
        if (n < required_n) {
            r.resize(required_n);
            p.resize(required_n);
            Ap.resize(required_n);
            n = required_n;
        }
        
        // Ensure device scalars are allocated
        if (d_rr.size() < 1) d_rr.resize(1);
        if (d_rr_new.size() < 1) d_rr_new.resize(1);
        if (d_pAp.size() < 1) d_pAp.resize(1);
        if (d_alpha.size() < 1) d_alpha.resize(1);
        if (d_beta.size() < 1) d_beta.resize(1);
        if (d_is_valid.size() < 1) d_is_valid.resize(1);
    }
};

} // namespace solvers
} // namespace macroflow3d
