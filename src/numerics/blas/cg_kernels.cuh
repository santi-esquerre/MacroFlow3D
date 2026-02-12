#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace blas {

// CG scalar update kernels (device-only, no sync)

// Compute alpha = rr / pAp and store in d_alpha
void compute_alpha(CudaContext& ctx,
                   DeviceSpan<const real> d_rr,
                   DeviceSpan<const real> d_pAp,
                   DeviceSpan<real> d_alpha);

// Compute beta = rr_new / rr and store in d_beta
void compute_beta(CudaContext& ctx,
                  DeviceSpan<const real> d_rr_new,
                  DeviceSpan<const real> d_rr,
                  DeviceSpan<real> d_beta);

// Fused update: x = x + alpha*p, r = r - alpha*Ap
void update_x_and_r(CudaContext& ctx,
                    DeviceSpan<const real> d_alpha,
                    DeviceSpan<const real> p,
                    DeviceSpan<real> x,
                    DeviceSpan<const real> Ap,
                    DeviceSpan<real> r);

// Fused update: p = r + beta*p
void update_p(CudaContext& ctx,
              DeviceSpan<const real> d_beta,
              DeviceSpan<const real> r,
              DeviceSpan<real> p);

// Check if pAp is valid (not zero, not NaN) and write flag to d_is_valid (1=ok, 0=bad)
void check_pAp_valid(CudaContext& ctx,
                     DeviceSpan<const real> d_pAp,
                     DeviceSpan<int> d_is_valid);

} // namespace blas
} // namespace macroflow3d
