#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace operators {

struct Poisson3DOperator {
    Grid3D grid;
    
    Poisson3DOperator() = default;
    
    explicit Poisson3DOperator(const Grid3D& g) : grid(g) {}
    
    // Matrix-free apply: y = A*x
    // Implements discrete Laplacian (7-point stencil)
    // BC: Dirichlet homogeneous (x=0 outside domain)
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
};

} // namespace operators
} // namespace rwpt
