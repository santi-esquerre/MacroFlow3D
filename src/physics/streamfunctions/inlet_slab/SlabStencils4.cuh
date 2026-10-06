#pragma once

/**
 * @file SlabStencils4.cuh
 * @brief SF-33 N0: 4th-order stencils of the inlet slab (decision 2026-10-06 item 4; the
 * prototype's `metrics.fd_weights`, `metrics.x1_stencils4`, `candidate_i.Stencils4`, `_dp4`,
 * `_dpp4`, `derivs4`, `metrics.d1_fd4`, `metrics.dp_fd4`).
 *
 * x1 stencils (one table for planes 0..N, built once per grid on the host by the Fornberg routine):
 *   d1  (5 points):  planes 2..N-2 centered (1, -8, 0, 8, -1)/12 on j-2..j+2;
 *                    planes 0, 1: Fornberg on planes 0..4 at x0 = 0 / 1
 *                      (plane 0 = (-25, 48, -36, 16, -3)/12, plane 1 = (-3, -10, 18, -6, 1)/12);
 *                    planes N-1, N: the mirror on planes N-4..N (plane N = (3, -16, 36, -48,
 * 25)/12). d11:             planes 2..N-2 centered (-1, 16, -30, 16, -1)/12 (5 points); planes 0,
 * 1: Fornberg 6 points on planes 0..5 (plane 1 = (10, -15, -4, 14, -6, 1)/12); planes N-1, N: the
 * mirror on planes N-5..N. Weights are stored dimensionless (times h^deriv) and applied as  (sum_k
 * w_k A[first + k]) / h^deriv  with the sum accumulated from 0 in k order (the prototype's
 * `sum(...) / h ** deriv`). The centered weights are the literal `np.array([...]) / 12.0` of the
 * prototype; the boundary weights are the host Fornberg weights. The d11 rows of planes 0 and N are
 * not used by the residual rows (the outlet rows overwrite plane N) but are part of the prototype's
 * `Stencils4` and of the stencil self-check.
 *
 * In-plane (periodic, exact integer wrap):
 *   d_j  u = (-u[+2] + 8 u[+1] - 8 u[-1] + u[-2]) / (12 h)
 *   d_jj u = (-u[+2] + 16 u[+1] - 30 u[0] + 16 u[-1] - u[-2]) / (12 h h)
 *   d23 u  = d3(d2 u) (composition of the two 5-point operators, NOT a 3x3 cross stencil)
 *   d1j u  = the x1 stencil of the plane applied to the 4th-order d_j field.
 *
 * Hessian packing used by every slab routine: H[0] = (0,0), H[1] = (1,1), H[2] = (2,2), H[3] =
 * (0,1), H[4] = (0,2), H[5] = (1,2) (symmetric).
 *
 * All evaluation routines are __host__ __device__ on raw pointers (device kernels pass device
 * pointers and the device table; host tests may pass host arrays and the host table).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Scalar.hpp"
#include "InletSlabGrid.cuh"

#include <cstddef>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

enum SlabHessianIndex : int { kH00 = 0, kH11 = 1, kH22 = 2, kH01 = 3, kH02 = 4, kH12 = 5 };

/// One x1 stencil: derivative at a plane from planes first .. first + npts - 1; weights times
/// h^deriv.
struct X1Stencil {
    int first = 0;
    int npts = 0;
    real w[6] = {0, 0, 0, 0, 0, 0};
};

/// Fornberg (1988) weights, port of `metrics.fd_weights`: c[k][i] approximates d^k/dx^k at x0 by
/// sum_i c[k][i] f(z[i]), k = 0..m. Exact for polynomials of degree < z.size().
std::vector<std::vector<real>> fd_weights(const std::vector<real>& z, real x0, int m);

/// Host tables of the x1 stencils on planes 0..N: d1[p], d11[p] (p = 0..N). Throws on an invalid N.
struct X1StencilTablesHost {
    std::vector<X1Stencil> d1;
    std::vector<X1Stencil> d11;
};
X1StencilTablesHost build_x1_stencils4(int n);

/// Device view of the table (trivially copyable kernel argument).
struct SlabStencilView {
    const X1Stencil* d1 = nullptr;  ///< planes 0..N
    const X1Stencil* d11 = nullptr; ///< planes 0..N
};

/// Device-resident x1 stencil table for one grid. build() allocates (grow-only) and uploads
/// synchronously; call it once per grid (it is a preparation step, never part of a hot loop).
class SlabStencilTable {
  public:
    void build(const InletSlabGrid& g);
    bool built_for(const InletSlabGrid& g) const { return n_ == g.n && n_ > 0; }
    SlabStencilView device_view() const;
    SlabStencilView
    host_view() const; ///< pointers into the host copy (host-side evaluation in tests)
    const X1StencilTablesHost& host() const { return host_; }
    std::size_t allocated_bytes() const { return dev_.capacity() * sizeof(X1Stencil); }

  private:
    int n_ = 0;
    X1StencilTablesHost host_;
    DeviceBuffer<X1Stencil> dev_; ///< [d1 planes 0..N | d11 planes 0..N]
};

// ------------------------------------------------------------------------------------------------
// Pointwise stencil evaluation
// ------------------------------------------------------------------------------------------------

/// sum_k w_k A[first + k] at the in-plane position `off` of a full array (no division by h).
__host__ __device__ inline real x1_stencil_sum(const X1Stencil& s, const real* A, std::size_t plane,
                                               std::size_t off) {
    real acc = 0.0;
    for (int k = 0; k < s.npts; ++k) {
        acc += s.w[k] * A[static_cast<std::size_t>(s.first + k) * plane + off];
    }
    return acc;
}

/// 4th-order periodic first derivative of a full array along x2 (axis = 2) or x3 (axis = 3) at (j,
/// m2, m3).
__host__ __device__ inline real dp4_at(const real* A, const InletSlabGrid& g, int j, int m2, int m3,
                                       int axis) {
    real ap2, ap1, am1, am2;
    if (axis == 2) {
        ap2 = A[g.full_index(j, g.wrap(m2 + 2), m3)];
        ap1 = A[g.full_index(j, g.wrap(m2 + 1), m3)];
        am1 = A[g.full_index(j, g.wrap(m2 - 1), m3)];
        am2 = A[g.full_index(j, g.wrap(m2 - 2), m3)];
    } else {
        ap2 = A[g.full_index(j, m2, g.wrap(m3 + 2))];
        ap1 = A[g.full_index(j, m2, g.wrap(m3 + 1))];
        am1 = A[g.full_index(j, m2, g.wrap(m3 - 1))];
        am2 = A[g.full_index(j, m2, g.wrap(m3 - 2))];
    }
    return (-ap2 + 8.0 * ap1 - 8.0 * am1 + am2) / (12.0 * g.h);
}

/// 4th-order periodic second derivative along x2 (axis = 2) or x3 (axis = 3) at (j, m2, m3).
__host__ __device__ inline real dpp4_at(const real* A, const InletSlabGrid& g, int j, int m2,
                                        int m3, int axis) {
    real ap2, ap1, a0, am1, am2;
    a0 = A[g.full_index(j, m2, m3)];
    if (axis == 2) {
        ap2 = A[g.full_index(j, g.wrap(m2 + 2), m3)];
        ap1 = A[g.full_index(j, g.wrap(m2 + 1), m3)];
        am1 = A[g.full_index(j, g.wrap(m2 - 1), m3)];
        am2 = A[g.full_index(j, g.wrap(m2 - 2), m3)];
    } else {
        ap2 = A[g.full_index(j, m2, g.wrap(m3 + 2))];
        ap1 = A[g.full_index(j, m2, g.wrap(m3 + 1))];
        am1 = A[g.full_index(j, m2, g.wrap(m3 - 1))];
        am2 = A[g.full_index(j, m2, g.wrap(m3 - 2))];
    }
    return (-ap2 + 16.0 * ap1 - 30.0 * a0 + 16.0 * am1 - am2) / (12.0 * g.h * g.h);
}

/// metrics.d1_fd4 at plane j = 0..N (also the d1 of derivs4 on planes 1..N: same table row).
__host__ __device__ inline real d1_fd4_at(const real* A, const SlabStencilView& st,
                                          const InletSlabGrid& g, int j, int m2, int m3) {
    return x1_stencil_sum(st.d1[j], A, g.plane_size(), g.plane_index(m2, m3)) / g.h;
}

/// d23 u = d3 (d2 u) at (j, m2, m3): the 5-point x3 stencil of the 5-point d2 field.
__host__ __device__ inline real d23_at(const real* U, const InletSlabGrid& g, int j, int m2,
                                       int m3) {
    const real ap2 = dp4_at(U, g, j, m2, g.wrap(m3 + 2), 2);
    const real ap1 = dp4_at(U, g, j, m2, g.wrap(m3 + 1), 2);
    const real am1 = dp4_at(U, g, j, m2, g.wrap(m3 - 1), 2);
    const real am2 = dp4_at(U, g, j, m2, g.wrap(m3 - 2), 2);
    return (-ap2 + 8.0 * ap1 - 8.0 * am1 + am2) / (12.0 * g.h);
}

/// d1j u = x1 stencil of plane j applied to the 4th-order d_j field (axis = 2 or 3).
__host__ __device__ inline real d1j_at(const real* U, const SlabStencilView& st,
                                       const InletSlabGrid& g, int j, int m2, int m3, int axis) {
    const X1Stencil& s = st.d1[j];
    real acc = 0.0;
    for (int k = 0; k < s.npts; ++k)
        acc += s.w[k] * dp4_at(U, g, s.first + k, m2, m3, axis);
    return acc / g.h;
}

/// Gradient of a periodic part U (full array, planes 0..N, read-only) at a vertex of plane j = 0..N
/// (the d1 row of plane j; planes 1..N coincide with derivs4, plane 0 is the metrics' one-sided
/// stencil).
__host__ __device__ inline void grad4_at(const real* U, const SlabStencilView& st,
                                         const InletSlabGrid& g, int j, int m2, int m3,
                                         real grad[3]) {
    grad[0] = d1_fd4_at(U, st, g, j, m2, m3);
    grad[1] = dp4_at(U, g, j, m2, m3, 2);
    grad[2] = dp4_at(U, g, j, m2, m3, 3);
}

/**
 * The prototype's `derivs4` at one vertex of an unknown plane j = 1..N: gradient and the six
 * Hessian entries (packing above) of a periodic part U stored on planes 0..N (read-only).
 */
__host__ __device__ inline void derivs4_at(const real* U, const SlabStencilView& st,
                                           const InletSlabGrid& g, int j, int m2, int m3,
                                           real grad[3], real H[6]) {
    grad4_at(U, st, g, j, m2, m3, grad);
    H[kH00] = x1_stencil_sum(st.d11[j], U, g.plane_size(), g.plane_index(m2, m3)) / (g.h * g.h);
    H[kH11] = dpp4_at(U, g, j, m2, m3, 2);
    H[kH22] = dpp4_at(U, g, j, m2, m3, 3);
    H[kH01] = d1j_at(U, st, g, j, m2, m3, 2);
    H[kH02] = d1j_at(U, st, g, j, m2, m3, 3);
    H[kH12] = d23_at(U, g, j, m2, m3);
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
