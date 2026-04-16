#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace blas {

// Copy vector: y = x
void copy(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace macroflow3d
