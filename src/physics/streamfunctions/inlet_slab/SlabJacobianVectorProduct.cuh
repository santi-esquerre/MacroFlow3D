#pragma once

/**
 * @file SlabJacobianVectorProduct.cuh
 * @brief SF-33 N1: analytic Jacobian-vector product (JVP) of the inlet-slab residual E(u)
 *        (SlabResidual.cuh; the prototype's `candidate_i.system`, variant i1, order 4).
 *
 * Derivation (N0 sign conventions; Hessian packing kH00, kH11, kH22, kH01, kH02, kH12)
 * -----------------------------------------------------------------------------------
 * E(u) is a composition of LINEAR stencil operators of the periodic parts u_i (gradients g_i,
 * packed Hessians H_i; the affine parts e2, e3 of psi1 = x2 + u1, psi2 = x3 + u2 have zero
 * derivative) and pointwise rational algebra. Its directional derivative at u in the direction du
 * (both fields, planes 1..N; du = 0 on plane 0, the Dirichlet inlet) is therefore the SAME stencil
 * evaluation applied to du,
 *     dg_i = grad4(du_i)      (NO affine part),     dH_i = derivs4 Hessian of du_i,
 * followed by the product rule on the pointwise algebra of `slab_equation_point`:
 *     c    = g1 x g2,  cc = c . c,  B = H2 g1 - H1 g2,  S_i = ((B x g_i) . c) / cc,
 *     L_i  = tr H_i - grad(ln k) . g_i,  F_i = -q (L_i - S_i)
 * gives
 *     dc   = dg1 x g2 + g1 x dg2
 *     dB   = dH2 g1 + H2 dg1 - dH1 g2 - H1 dg2
 *     dS_i = [ (dB x g_i + B x dg_i) . c + (B x g_i) . dc ] / cc  -  S_i (2 c . dc) / cc
 *     dL_i = tr dH_i - grad(ln k) . dg_i
 *     dF_i = -q (dL_i - dS_i)                                    (equation rows, planes 1..N-1)
 *     dE_1 = -(2 q_N / h) dc_2,   dE_2 = -(2 q_N / h) dc_3        (outlet rows, plane N; dc from the
 *                                                                   plane-N one-sided gradients)
 * (dc_2, dc_3 = the x2, x3 components, array indices 1, 2). q, grad(ln k), v_perp,in do not depend
 * on u. No regularization anywhere: cc is used as is (exactly as in the residual).
 *
 * The k = 1, u = 0 linearization (symbol reused by the N2 per-transverse-mode preconditioner)
 * ------------------------------------------------------------------------------------------
 * Base: q = 1, grad(ln k) = 0, u = 0, so g1 = e2, g2 = e3, H_i = 0, B = 0, c = e2 x e3 = e1,
 * cc = 1, S_i = 0. Then dB = dH2 e2 - dH1 e3, all terms with B or S_i vanish, and
 *     dS_1 = ((dB x e2) . e1) = -dB_3 = -(d23 du2 - d33 du1) =  d33 du1 - d23 du2
 *     dS_2 = ((dB x e3) . e1) =  dB_2 =   d22 du2 - d23 du1
 * so the equation rows are (d23 = the composition d3 d2 of N0, NOT a cross stencil):
 *     dF_1 = -[ (d11 + d22) du1 + d23 du2 ]
 *     dF_2 = -[ (d11 + d33) du2 + d23 du1 ]
 * i.e. the d23 coupling enters with a PLUS sign inside the bracket (overall -d23) in both rows.
 * Outlet rows: dc = dg1 x e3 + e2 x dg2 = (d2 du1 + d3 du2, -d1 du1, -d1 du2), hence
 *     dE_1 = -(2/h) dc_2 = +(2/h) d1 du1,     dE_2 = -(2/h) dc_3 = +(2/h) d1 du2
 * (d1 = the one-sided 5-point plane-N row). NOTE the sign: the outlet rows are +(2q/h) d1 du_i,
 * not -(2q/h) d1 du_i; this agrees with the prototype's `ModePrec` (`base11[N-1] = D11[N-1]`,
 * "outlet row: +(2/h) d1 u"). The unit test `inlet_slab_jvp` case 3 pins this symbol to 1e-12.
 *
 * Memory choice (documented trade-off)
 * ------------------------------------
 * The base state is frozen by COPYING the two full periodic-part arrays (planes 0..N) into the
 * workspace; g_i, H_i, B, c, S_i of the base are RECOMPUTED on the fly inside `apply` with the
 * same inline routines as the residual (bitwise-identical base algebra). Footprint: 4 full arrays
 * (2 base + 2 direction with a zero plane 0) = 4 (N+1) N^2 * 8 B = 68 MB at 128^3, instead of
 * ~24 precomputed per-vertex fields (~400 MB at 128^3). Cost: each apply evaluates the 4th-order
 * derivs4 of four arrays per vertex (about twice the residual's stencil work); on the GPU this is
 * memory-light and cache-friendly. `allocated_bytes()` reports the footprint.
 *
 * Inputs (q, grad ln k, the row scale) are REFERENCED, not copied: the SlabStageInputs object
 * passed to `prepare_base` must outlive every subsequent `apply` and its contents must not change
 * between `prepare_base` and `apply` (re-call `prepare_base` after a stage change).
 *
 * Synchronization / allocation contract
 * -------------------------------------
 *   - prepare(grid): the only allocating call (grow-only); builds the stencil table (synchronous
 *     upload) and zeroes plane 0 of the direction arrays (synchronous memset). Invalidates the base.
 *   - prepare_base(...): host-side checks + 2 device-to-device copies enqueued on ctx.cuda_stream();
 *     no allocation, no synchronization.
 *   - apply(...): 2 device-to-device copies (direction planes 1..N -> internal arrays whose plane 0
 *     stays zero) + 1 kernel, all enqueued on ctx.cuda_stream(); NO allocation and NO host
 *     synchronization. The direction is never mutated; the output is not scaled. Throws
 *     std::logic_error if called before prepare_base (or after a prepare that invalidated it).
 *   - Every operation is stream-ordered on ctx.cuda_stream(); callers using the same stream need
 *     no extra synchronization between evaluate_residual / apply / their own kernels.
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"
#include "SlabResidual.cuh"
#include "SlabStencils4.cuh"

#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

// ------------------------------------------------------------------------------------------------
// Pointwise directional derivatives (host + device; differentiate slab_equation_point /
// slab_outlet_point)
// ------------------------------------------------------------------------------------------------

/**
 * Directional derivative of the equation rows at one vertex. (g1, g2, H1, H2, glnk, q) and `s`
 * are the base state exactly as passed to / produced by slab_equation_point; (dg1, dg2, dH1, dH2)
 * the stencil derivatives of the direction (no affine part). Formulas: see the file header.
 */
__host__ __device__ inline void
slab_equation_point_jvp(const real g1[3], const real g2[3], const real H1[6], const real H2[6],
                        const real glnk[3], real q, const SlabPointState& s, const real dg1[3],
                        const real dg2[3], const real dH1[6], const real dH2[6], real& dF1,
                        real& dF2) {
    // dB = dH2 g1 + H2 dg1 - dH1 g2 - H1 dg2
    real a[3], b[3], cdd[3], d[3];
    slab_hv(dH2, g1, a);
    slab_hv(H2, dg1, b);
    slab_hv(dH1, g2, cdd);
    slab_hv(H1, dg2, d);
    real dB[3];
    for (int k = 0; k < 3; ++k)
        dB[k] = a[k] + b[k] - cdd[k] - d[k];
    // dc = dg1 x g2 + g1 x dg2
    real x1[3], x2[3], dc[3];
    slab_cross(dg1, g2, x1);
    slab_cross(g1, dg2, x2);
    for (int k = 0; k < 3; ++k)
        dc[k] = x1[k] + x2[k];
    const real two_cdc = 2.0 * slab_dot(s.c, dc);

    const real* gi[2] = {g1, g2};
    const real* dgi[2] = {dg1, dg2};
    const real Si[2] = {s.S1, s.S2};
    real dS[2];
    for (int i = 0; i < 2; ++i) {
        real bg[3], dbg1[3], dbg2[3], dbg[3];
        slab_cross(s.B, gi[i], bg);
        slab_cross(dB, gi[i], dbg1);
        slab_cross(s.B, dgi[i], dbg2);
        for (int k = 0; k < 3; ++k)
            dbg[k] = dbg1[k] + dbg2[k];
        dS[i] = (slab_dot(dbg, s.c) + slab_dot(bg, dc)) / s.cc - Si[i] * two_cdc / s.cc;
    }
    const real dL1 = dH1[kH00] + dH1[kH11] + dH1[kH22] - slab_dot(glnk, dg1);
    const real dL2 = dH2[kH00] + dH2[kH11] + dH2[kH22] - slab_dot(glnk, dg2);
    dF1 = -q * (dL1 - dS[0]);
    dF2 = -q * (dL2 - dS[1]);
}

/// Directional derivative of the outlet rows at one vertex of plane N:
/// dE_1 = -(2q/h) dc_2, dE_2 = -(2q/h) dc_3 with dc = dg1 x g2 + g1 x dg2.
__host__ __device__ inline void slab_outlet_point_jvp(const real g1[3], const real g2[3],
                                                      const real dg1[3], const real dg2[3], real q,
                                                      real h, real& dE1, real& dE2) {
    real x1[3], x2[3];
    slab_cross(dg1, g2, x1);
    slab_cross(g1, dg2, x2);
    const real sc = 2.0 * q / h;
    dE1 = -sc * (x1[1] + x2[1]);
    dE2 = -sc * (x1[2] + x2[2]);
}

// ------------------------------------------------------------------------------------------------
// Workspace
// ------------------------------------------------------------------------------------------------

class SlabJvpWorkspace {
  public:
    /// Allocates (grow-only) the 4 full arrays and builds the stencil table for `g`; zeroes plane
    /// 0 of the direction arrays. The only allocating call. Invalidates any frozen base state.
    void prepare(const InletSlabGrid& g);
    bool prepared_for(const InletSlabGrid& g) const {
        return n_ == g.n && n_ > 0 && table_.built_for(g);
    }

    /**
     * Freezes the base state: copies the FULL periodic-part arrays U1, U2 (planes 0..N, as
     * evaluate_residual takes them; plane 0 = inlet data) and references `inputs` (q, grad ln k).
     * Enqueue only (2 D2D copies); no allocation, no sync. Throws std::logic_error if the
     * workspace is not prepared for `grid`, std::invalid_argument on size mismatch.
     */
    void prepare_base(CudaContext& ctx, const InletSlabGrid& grid, const SlabStageInputs& inputs,
                      DeviceSpan<const real> U1, DeviceSpan<const real> U2);
    bool has_base() const { return base_inputs_ != nullptr; }

    /**
     * out = J(u_base) * direction. direction, out: 2 N^3, field-major [f1 | f2] on planes 1..N
     * (the direction's plane 0 is implicitly zero). Enqueue only: no allocation, no host sync.
     * `direction` is never modified; `direction` and `out` must not overlap (std::invalid_argument).
     * Throws std::logic_error if no base state was frozen (prepare_base) for the current prepare.
     */
    void apply(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> direction,
               DeviceSpan<real> out);

    std::size_t allocated_bytes() const;
    const SlabStencilTable& stencils() const { return table_; }

  private:
    int n_ = 0;
    SlabStencilTable table_;
    DeviceBuffer<real> base_U_[2]; ///< frozen base periodic parts, planes 0..N
    DeviceBuffer<real> dir_U_[2];  ///< direction, plane 0 = 0 (set in prepare), planes 1..N copied
    const SlabStageInputs* base_inputs_ = nullptr;
};

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
