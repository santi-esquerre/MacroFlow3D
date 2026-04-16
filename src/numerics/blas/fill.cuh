#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace blas {

// Fill vector with constant value
void fill(CudaContext& ctx, DeviceSpan<real> y, real value);

} // namespace blas
} // namespace macroflow3d
