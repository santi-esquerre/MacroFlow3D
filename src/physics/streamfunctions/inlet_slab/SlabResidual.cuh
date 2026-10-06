#pragma once

/**
 * @file SlabResidual.cuh
 * @brief SF-33 N0: residual of the same-index equation (14) on the inlet slab with the outlet
 * oblique rows (the prototype's `candidate_i.pieces` + `system`, variant i1, order 4).
 *
 * Equation rows (planes j = 1..N-1, both fields; same-index pairing, non-divergence form, NO
 * regularization): g1 = (d1 u1, 1 + d2 u1, d3 u1),  g2 = (d1 u2, d2 u2, 1 + d3 u2)      (derivs4,
 * SlabStencils4.cuh) B  = H2 g1 - H1 g2,   c = g1 x g2,   cc = c . c S_i = ((B x g_i) . c) / cc L_i
 * = H_i(0,0) + H_i(1,1) + H_i(2,2) - grad(ln k) . g_i F_i = -q (L_i - S_i),  q = 1/k at the vertex
 * Outlet rows (plane N, both fields; D-2): with c from the plane-N gradients (one-sided 5-point
 * d1), E_1 = -(2 q_N / h)(c2 - v2_in),   E_2 = -(2 q_N / h)(c3 - v3_in). Residual vector E:
 * field-major [E_1 | E_2], each N^3 for planes 1..N (layout of InletSlabGrid.cuh). Norms
 * (SlabResidualNorms): r_F = sqrt((mean F1^2 + mean F2^2)/2) / q_rms over planes 1..N-1 (q_rms over
 * the same planes); r_out = sqrt(mean((c2 - v2_in)^2 + (c3 - v3_in)^2)) / v_rms over plane N.
 *
 * The pointwise algebra is factored into __host__ __device__ inline functions
 * (`slab_equation_point`, `slab_outlet_point`) so that the Jacobian-vector product of a later node
 * differentiates exactly this algebra.
 *
 * Synchronization / allocation contract:
 *   - SlabResidualWorkspace::prepare(grid) is the only allocating call (grow-only) and uploads the
 * stencil table synchronously; evaluate_residual never allocates (it throws if the workspace was
 * not prepared for the grid).
 *   - The residual kernel performs no host synchronization. With norms_out == nullptr
 * evaluate_residual only enqueues work on ctx.cuda_stream() (no sync at all). With norms_out !=
 * nullptr it additionally enqueues the deterministic two-stage reduction of the four sums (sum
 * F1^2, sum F2^2, sum q^2 on planes 1..N-1 and the outlet defect sum on plane N), copies the 4
 * doubles to the host and performs EXACTLY ONE cudaStreamSynchronize(ctx.cuda_stream()); the norms
 * are formed on the host from those sums.
 *   - Reductions: grid-stride kernels with a grid size that depends only on N
 * (detail::slab_reduce_blocks), fixed block tree, fixed-order final pass: bitwise reproducible run
 * to run on the same device / build.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabStencils4.cuh"

#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

// ------------------------------------------------------------------------------------------------
// Pointwise algebra (shared with the Jacobian-vector product)
// ------------------------------------------------------------------------------------------------

__host__ __device__ inline void slab_cross(const real a[3], const real b[3], real out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__host__ __device__ inline real slab_dot(const real a[3], const real b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// (H v)_i = H(i,0) v0 + H(i,1) v1 + H(i,2) v2 with the packed symmetric Hessian
/// (SlabStencils4.cuh).
__host__ __device__ inline void slab_hv(const real H[6], const real v[3], real out[3]) {
    out[0] = H[kH00] * v[0] + H[kH01] * v[1] + H[kH02] * v[2];
    out[1] = H[kH01] * v[0] + H[kH11] * v[1] + H[kH12] * v[2];
    out[2] = H[kH02] * v[0] + H[kH12] * v[1] + H[kH22] * v[2];
}

/// Intermediates of one equation-row evaluation (the base state of the directional derivative).
struct SlabPointState {
    real B[3];
    real c[3];
    real cc;
    real S1, S2;
    real L1, L2;
    real F1, F2;
};

/**
 * Equation rows at one vertex from the full label gradients g1, g2 (affine parts included), the
 * packed Hessians H1, H2 of the periodic parts, grad(ln k) and q. Operation order follows
 * `candidate_i.pieces`.
 */
__host__ __device__ inline void slab_equation_point(const real g1[3], const real g2[3],
                                                    const real H1[6], const real H2[6],
                                                    const real glnk[3], real q, SlabPointState& s) {
    real h2g1[3], h1g2[3];
    slab_hv(H2, g1, h2g1);
    slab_hv(H1, g2, h1g2);
    s.B[0] = h2g1[0] - h1g2[0];
    s.B[1] = h2g1[1] - h1g2[1];
    s.B[2] = h2g1[2] - h1g2[2];
    slab_cross(g1, g2, s.c);
    s.cc = slab_dot(s.c, s.c);
    real bg[3];
    slab_cross(s.B, g1, bg);
    s.S1 = slab_dot(bg, s.c) / s.cc;
    slab_cross(s.B, g2, bg);
    s.S2 = slab_dot(bg, s.c) / s.cc;
    s.L1 = H1[kH00] + H1[kH11] + H1[kH22] - slab_dot(glnk, g1);
    s.L2 = H2[kH00] + H2[kH11] + H2[kH22] - slab_dot(glnk, g2);
    s.F1 = -q * (s.L1 - s.S1);
    s.F2 = -q * (s.L2 - s.S2);
}

/// Equation rows only (convenience wrapper of slab_equation_point).
__host__ __device__ inline void slab_equation_rows(const real g1[3], const real g2[3],
                                                   const real H1[6], const real H2[6],
                                                   const real glnk[3], real q, real& F1, real& F2) {
    SlabPointState s;
    slab_equation_point(g1, g2, H1, H2, glnk, q, s);
    F1 = s.F1;
    F2 = s.F2;
}

/**
 * Outlet rows at one vertex of plane N from the plane-N label gradients: d2 = c2 - v2_in, d3 = c3 -
 * v3_in, E1 = -(2 q / h) d2, E2 = -(2 q / h) d3.
 */
__host__ __device__ inline void slab_outlet_point(const real g1[3], const real g2[3], real v2_in,
                                                  real v3_in, real q, real h, real& E1, real& E2,
                                                  real& d2, real& d3) {
    real c[3];
    slab_cross(g1, g2, c);
    d2 = c[1] - v2_in;
    d3 = c[2] - v3_in;
    const real sc = 2.0 * q / h;
    E1 = -sc * d2;
    E2 = -sc * d3;
}

/// Full label gradients from the periodic-part gradients (affine e2 / e3 added exactly).
__host__ __device__ inline void slab_label_gradients(const real du1[3], const real du2[3],
                                                     real g1[3], real g2[3]) {
    g1[0] = du1[0];
    g1[1] = 1.0 + du1[1];
    g1[2] = du1[2];
    g2[0] = du2[0];
    g2[1] = du2[1];
    g2[2] = 1.0 + du2[2];
}

// ------------------------------------------------------------------------------------------------
// Workspace and evaluation
// ------------------------------------------------------------------------------------------------

class SlabResidualWorkspace {
  public:
    /// Allocates (grow-only) and builds the stencil table for `g`. The only allocating call.
    void prepare(const InletSlabGrid& g);
    bool prepared_for(const InletSlabGrid& g) const {
        return n_ == g.n && n_ > 0 && table_.built_for(g);
    }
    const SlabStencilTable& stencils() const { return table_; }
    std::size_t allocated_bytes() const;

    // Internal storage (exposed read-only for allocation-stability tests).
    const real* partials_data() const { return partials_.data(); }
    const real* sums_data() const { return sums_.data(); }

  private:
    friend void evaluate_residual(CudaContext&, const InletSlabGrid&, const SlabStageInputs&,
                                  DeviceSpan<const real>, DeviceSpan<const real>, DeviceSpan<real>,
                                  SlabResidualWorkspace&, SlabResidualNorms*);
    int n_ = 0;
    int nblocks_ = 0;
    SlabStencilTable table_;
    DeviceBuffer<real> partials_; ///< 4 * nblocks
    DeviceBuffer<real> sums_;     ///< 4
    real host_sums_[4] = {0, 0, 0, 0};
};

/**
 * Residual E(u) and (optionally) its norms.
 *   U1, U2: periodic parts on planes 0..N (full arrays, (N+1) N^2 each; read-only). Plane 0 must
 * hold the inlet data inputs.u0 (see assemble_full_planes); it is read by the skewed / one-sided x1
 * stencils only. E_out:  2 N^3, field-major [E_1 | E_2] on planes 1..N (equation rows on 1..N-1,
 * outlet rows on N). norms_out: nullptr -> pure enqueue (no reduction, no sync); otherwise one
 * explicit stream sync (see header).
 */
void evaluate_residual(CudaContext& ctx, const InletSlabGrid& grid, const SlabStageInputs& inputs,
                       DeviceSpan<const real> U1, DeviceSpan<const real> U2, DeviceSpan<real> E_out,
                       SlabResidualWorkspace& ws, SlabResidualNorms* norms_out);

/**
 * View convention between the unknown vector and the full arrays: a full array's plane 0 is the
 * inlet data u0_i and its planes 1..N are the contiguous unknown block of field i. Enqueues 4
 * device-to-device copies on ctx.cuda_stream() (no allocation, no sync). u_vec: 2 N^3 ([u1 | u2],
 * planes 1..N);  U1, U2: (N+1) N^2 outputs.
 */
void assemble_full_planes(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> u_vec,
                          const SlabStageInputs& inputs, DeviceSpan<real> U1, DeviceSpan<real> U2);

/// Inverse of assemble_full_planes on planes 1..N: u_vec = [U1 planes 1..N | U2 planes 1..N] (2 D2D
/// copies).
void extract_unknowns(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> U1,
                      DeviceSpan<const real> U2, DeviceSpan<real> u_vec);

/// q = 1.0 / exp(lnk) elementwise on the full arrays of `inputs` (the prototype's `1.0 /
/// np.exp(lnk)`); enqueue only.
void fill_q_from_lnk(CudaContext& ctx, const InletSlabGrid& grid, SlabStageInputs& inputs);

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
