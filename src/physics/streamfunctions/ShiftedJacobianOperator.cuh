#pragma once

/**
 * @file ShiftedJacobianOperator.cuh
 * @brief SF-25 D-gate enabler: matrix-free shifted-system action
 *        `(mu * A_blk + J) p`, `A_blk = diag(A, A)`, `A = -div(q grad)`.
 *
 * == Rationale (decision record, cite exactly) ==
 *
 * `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` records the
 * eta=1 gauge-manifold hypothesis: the Clebsch-pair recombination freedom
 * (`(psi1, psi2) -> (f(psi1,psi2), g(psi1,psi2))` with unit label-Jacobian)
 * leaves `v = grad(psi1) x grad(psi2)` invariant, so every such pair solves
 * the SAME coupled residual `F` at `eta = 1`. That non-uniqueness manifold's
 * tangent space is exactly the near-null cluster the coupled Jacobian `J`
 * exhibits at the plateau (matching the SF-21 Picard multiplier reaching
 * ~1). Both surveyed remedies (inexact Levenberg-Marquardt, Yamashita &
 * Fukushima 2001 / Dan, Yamashita & Fukushima 2002; pseudo-transient
 * continuation, Kelley & Keyes 1998) reduce to solving the SAME shifted
 * system `(mu * A_blk + J) delta = -F` for some `mu > 0` schedule. With the
 * accepted SF-23 block-diagonal MG preconditioner `M ~ A_blk`,
 * `M^-1 (mu * A_blk + J) ~ mu * I + M^-1 J`: the near-null cluster of
 * `M^-1 J` is moved to `~mu` in the preconditioned spectrum WITHOUT
 * touching the preconditioner itself. This file provides the matrix-free
 * `apply` for that shifted system; it is a D-GATE ENABLER (SF-25 activation
 * decision E1) -- the mu-sweep diagnostic needs this apply to exist, while
 * the TERMINAL SCHEDULE that would call it from inside the Newton phase
 * remains gated behind the D-gate's own verdict (out of scope here).
 *
 * == The A-apply mirrors the accepted residual evaluator exactly ==
 *
 * `A = -div(q grad)` is the SAME `operators::LesterPositiveDiffusionOperator`
 * call the accepted coupled residual evaluator already issues for its own
 * `A u1`/`A u2` terms (`ResidualEvaluator.cu`, step "5) A.apply(u1),
 * A.apply(u2)."):
 *
 *   const operators::LesterPositiveDiffusionOperator op(grid, q);
 *   op.apply(ctx, x, y);
 *
 * i.e. `A(q) u = -div_h(q grad_h u)` (positive semidefinite for positive
 * `q`, periodic BCs, no pinned row -- see
 * `lester_positive_diffusion_operator.cuh`). This file constructs the
 * IDENTICAL operator, over the SAME `(grid, q)` pair the caller supplies at
 * construction (the same `q` the `JvpWorkspace` base was prepared with), and
 * applies it component-wise to the RAW (unprojected) `direction` -- see the
 * mean-zero contract below.
 *
 * == apply() semantics ==
 *
 * `apply(ctx, grid, direction, delta_config, jv_out)`:
 *
 *  1. Forward to `jvp_.apply(ctx, grid, direction, delta_config, jv_out)`,
 *     producing `J p` in `jv_out` under the SAME D2 (forward-difference
 *     delta policy)/D3 (projection discipline) contracts `JvpWorkspace`
 *     documents. `jv_out` is left exactly as `JvpWorkspace::apply` leaves it
 *     (NOT re-projected -- see `JacobianVectorProduct.cuh`, D3). On
 *     `JvpApplyStatus::nonfinite_perturbed_residual` this returns
 *     IMMEDIATELY, WITHOUT applying the shift: a non-finite `J p` makes the
 *     shift meaningless, and the caller (a Krylov iteration) needs the same
 *     recoverable per-iteration signal `JvpWorkspace::apply` already
 *     provides, unchanged.
 *  2. When `mu_ > 0`: compute `A p_i` per component into the one owned `2n`
 *     scratch buffer, then `jv_out_i += mu_ * (A p_i)` via `blas::axpy`.
 *     `mu_ == 0` SKIPS this step entirely -- the result is then a bitwise
 *     passthrough of the plain `J p` `jvp_.apply` already produced (no A
 *     construction, no A.apply call, no axpy).
 *
 * == Mean-zero direction: caller contract, not a re-projection here ==
 *
 * `direction` is `const` and is NEVER mutated by this file (mirrors
 * `JvpWorkspace::apply`'s own D3 contract on its `direction` parameter).
 * `JvpWorkspace::apply` internally projects the direction into ITS OWN
 * private workspace copy before using it (D3) -- so `J p` above is actually
 * `J P(p)`, projected. `A`, by contrast, is applied here directly to the RAW
 * `direction` the caller passed in, with NO re-projection inside this
 * adapter. This is safe and exact ONLY under the caller contract that
 * `direction` is already mean-zero: every direction `CoupledGmres::solve`'s
 * Arnoldi loop forms (the D-gate's sole caller) is a GMRES basis vector,
 * which that loop already projects component-wise immediately after every
 * `J*M^-1` application (see `CoupledGmres.cuh`, E4: "projected AFTER the
 * `J*M^-1` application, BEFORE orthogonalization") -- so by the time any
 * later cycle's basis vector is handed to THIS operator's `apply`, it is
 * already mean-zero, and `A` of a mean-zero, triply-periodic field is
 * well-defined (the periodic Laplacian-like operator has no boundary
 * ambiguity to resolve for a zero-mean input). This mirrors the existing
 * "Jv output is intentionally not re-projected, observability over silent
 * laundering" convention `JacobianVectorProduct.cuh`'s D3 section documents
 * for its own output: this file does not silently re-project a caller input
 * either; a caller that violates the mean-zero contract gets an honestly
 * non-mean-zero `A p` contribution back, not a value quietly cleaned up
 * underneath it.
 *
 * == Grid identity ==
 *
 * No separate grid-identity check is performed here: step 1 above calls
 * `jvp_.apply(ctx, grid, ...)` FIRST, which already fails fast
 * (`std::invalid_argument`) if `grid` does not exactly match the grid
 * `jvp_`'s most recent `prepare_jvp_base` call cached (C01, T01-F1; see
 * `JacobianVectorProduct.cuh`). The `A` operator is built from the `Grid3D`
 * this object was CONSTRUCTED with (cached by value, mirroring
 * `JvpWorkspace`'s own `base_grid_` caching convention); a caller is
 * expected to construct this object with the SAME grid it passes to every
 * subsequent `apply()` call (exactly the grid the D-gate's frozen-plateau
 * fixture uses throughout one solve).
 *
 * == Memory ==
 *
 * `prepare(n)` allocates exactly one `2*n`-real scratch buffer (`A p_1` in
 * `[0, n)`, `A p_2` in `[n, 2n)`); no allocation occurs in `apply()`.
 * `estimate_device_bytes(n)` mirrors this exactly. `mu_ == 0` performs zero
 * extra device work in `apply()` beyond the passthrough `jvp_.apply` call,
 * but the scratch buffer itself is still allocated once by `prepare(n)`
 * regardless of `mu_` (so `set_mu` can move `mu_` away from zero later
 * without a fresh `prepare` call).
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "JacobianVectorProduct.cuh"

#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Light, non-owning adapter presenting the shifted-system action
 * `(mu * A_blk + J) p` through the SAME duck-typed operator interface
 * `CoupledGmres::solve`'s generalized `JacobianOperator` template parameter
 * expects (`apply(CudaContext&, const Grid3D&, ConstCoupledVectorView, const
 * JvpDeltaConfig&, CoupledVectorView) -> JvpApplyReport`). Non-owning: `jvp`
 * must outlive this object; `q` must remain valid for the lifetime of every
 * `apply()` call (the same non-owning contract `LesterPositiveDiffusionOperator`
 * and `JvpWorkspace::prepare_jvp_base` already place on their own `q` spans).
 * Move-only, matching every other accepted streamfunction workspace
 * convention.
 */
class ShiftedJacobianOperator {
  public:
    ShiftedJacobianOperator(JvpWorkspace& jvp, const Grid3D& grid, DeviceSpan<const real> q,
                            real mu);

    ShiftedJacobianOperator(const ShiftedJacobianOperator&) = delete;
    ShiftedJacobianOperator& operator=(const ShiftedJacobianOperator&) = delete;
    ShiftedJacobianOperator(ShiftedJacobianOperator&&) noexcept = default;
    ShiftedJacobianOperator& operator=(ShiftedJacobianOperator&&) noexcept = default;

    // Allocates the one 2n-real scratch buffer (see the file header).
    // Idempotent for an already-prepared n (no device allocation). Throws
    // std::invalid_argument if n == 0.
    void prepare(std::size_t n);

    [[nodiscard]] bool prepared_for(std::size_t n) const noexcept;

    // (mu * A_blk + J) direction, into jv_out (see the file header for the
    // exact two-step semantics and the mean-zero direction caller contract).
    // Throws std::logic_error if prepare(n) was not called for the direction/
    // jv_out component size n; every other precondition (grid identity,
    // delta_config validity, direction/jv_out sizing) is delegated to
    // jvp_.apply, called first.
    [[nodiscard]] JvpApplyReport apply(CudaContext& ctx, const Grid3D& grid,
                                       ConstCoupledVectorView direction,
                                       const JvpDeltaConfig& delta_config,
                                       CoupledVectorView jv_out);

    // Exact sum of the owned 2n-real scratch buffer's capacity. Never
    // allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Host-only prediction of allocated_device_bytes() after a fresh
    // prepare(n) call; kept colocated with prepare() so it cannot drift.
    [[nodiscard]] static std::size_t estimate_device_bytes(std::size_t n);

    [[nodiscard]] real mu() const noexcept { return mu_; }

    // Throws std::invalid_argument if mu is not finite and >= 0.
    void set_mu(real mu);

  private:
    void ensure_prepared() const;

    JvpWorkspace* jvp_;
    Grid3D grid_;
    DeviceSpan<const real> q_;
    real mu_;

    std::size_t n_ = 0;

    // One 2n-real scratch buffer: A p_1 in [0, n), A p_2 in [n, 2n) (see the
    // file header).
    DeviceBuffer<real> ap_scratch_;
};

} // namespace streamfunctions
} // namespace macroflow3d
