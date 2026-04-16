#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace blas {

// Axpy: y = a*x + y
void axpy(CudaContext& ctx, real a, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace macroflow3d
