#pragma once

/**
 * @file BlockDiagonalMGPreconditioner.cuh
 * @brief SF-23 physics-informed block diagonal GMRES preconditioner
 *        `M = diag(M_A, M_A)`, `M_A` = two successive projected positive
 *        V-cycles on the SAME populated `A(q) = -div(q grad)` hierarchy.
 *
 * `CoupledGmres.cuh` treats `M^-1` as the RIGHT preconditioner (SF-23
 * activation decision E1): the coupled Newton linearization `J(Psi) delta =
 * b` is solved as `J(Psi) M^-1 u = b`, `delta = M^-1 u`. Since the coupled
 * Jacobian couples psi1/psi2 through the nonlinear source terms while the
 * decoupled Picard linear subproblem (`ARCHITECTURE.md` section 4.3) does
 * not, the accepted, already-validated `A(q)` diffusion block is the natural
 * per-component approximation: `M^-1 = diag(M_A, M_A)` applied independently
 * to each coupled component.
 *
 * == Two-V-cycle composition (E2) ==
 *
 * `ProjectedPositiveMGPreconditioner::apply` (SF-05/SF-12,
 * `projected_positive_mg_preconditioner.cuh/.cu`, not modified here) forms
 * `b0 = -P(r)` (legacy sign: the reused kernels solve `L(q) x = b` for
 * `L(q) = div(q grad) = -A(q)`, so `b0 = -P(r)` makes one zero-init
 * `projected_positive_v_cycle` produce `x1 ~= L^-1 b0 = A(q)^-1 P(r)`), then
 * projects `x1` once more. `M_A` here reuses EXACTLY that one-cycle
 * computation for the first cycle, then treats it as one step of stationary
 * defect correction against the SAME linear operator `L(q)`:
 *
 *   x1  = P( V-cycle(b0) ),                          b0 = -P(r)
 *   res = P( b0 - L(q) x1 )                           (compute_residual_3d,
 *                                                       the SAME kernel
 *                                                       `cycle()` itself uses
 *                                                       internally)
 *   dx  = P( V-cycle(res) )
 *   z   = P( x1 + dx )
 *
 * `compute_residual_3d(ctx, grid, x, b, K, r, bc)` computes `r = b - L(q) x`
 * (its own doc comment: "Compute residual: r = b - A*x", where that "A" is
 * this file's `L(q)`) -- exactly the residual `cycle()` forms internally
 * before restricting to the next coarser level, so `res` above is the TRUE
 * level-0 residual of the first cycle's solve of `L(q) x = b0`, not the
 * stale mid-cycle residual `cycle()` leaves in `hierarchy.levels[0].r` after
 * its own presmooth-only residual snapshot (that buffer is NOT reused here;
 * `res` is written to a dedicated, non-aliased scratch buffer).
 *
 * Because `dx` solves `L(q) dx ~= res = b0 - L(q) x1` from a FRESH zero
 * initial guess, `x1 + dx` is exactly one step of classical iterative
 * refinement for `L(q) x = b0`, i.e. TWO successive projected positive
 * V-cycles composed as the spec requires (not `V-cycle(V-cycle(b0))`, which
 * would double-apply `b0` rather than refine against the true residual).
 *
 * == Fixedness / linearity (E2, gated by tests, not assumed) ==
 *
 * Every step above (`V-cycle` from a zero initial guess, `compute_residual_3d`,
 * `MeanZeroProjector::project`) is a FIXED linear map in its right-hand
 * side/input for FIXED `(hierarchy coefficients, MGConfig smoothing counts)`:
 * no data-dependent branching, no randomness, no varying iteration counts.
 * Composing fixed linear maps (including the additive `x1 + dx` combination)
 * yields a fixed linear map `M_A`, hence `M^-1 = diag(M_A, M_A)` is fixed and
 * linear in its coupled input -- this is a PRESPECIFIED TEST in the SF-23
 * bitácora (bitwise repeatability of `M v` across calls AND
 * `||M(a x + b y) - a M x - b M y|| <= 1e-12 * scale`), not an assumption
 * this file relies on unverified.
 *
 * == Ownership / reentrancy ==
 *
 * Like `ProjectedPositiveMGPreconditioner`, this is a NON-OWNING adapter over
 * a caller-owned, already-`q`-populated `multigrid::MGHierarchy&`; `apply()`
 * mutates the hierarchy's per-level work buffers and this object's own
 * mean-zero/scratch workspaces, so it is not reentrant and not safe for
 * concurrent streams. `apply()` applies the two-cycle `M_A` to component 1,
 * then to component 2, SEQUENTIALLY, reusing the SAME hierarchy for both
 * (SF-23 spec: "on the SAME populated hierarchy").
 *
 * == Memory ==
 *
 * Owned bytes are the per-level `MeanZeroWorkspace` set (identical
 * accounting to `ProjectedPositiveMGPreconditioner::estimate_device_bytes`)
 * plus two finest-grid-sized scratch buffers (`x1` stash, defect-correction
 * residual), each `n = hierarchy.finest_grid().num_cells()` reals: exactly
 * `2 * n * sizeof(real)` beyond the per-level workspace total.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../multigrid/cycle/projected_positive_v_cycle.cuh"
#include "JacobianVectorProduct.cuh"

#include <vector>

namespace macroflow3d {
namespace streamfunctions {

class BlockDiagonalMGPreconditioner {
  public:
    BlockDiagonalMGPreconditioner(multigrid::MGHierarchy& hierarchy,
                                  const multigrid::MGConfig& config);

    BlockDiagonalMGPreconditioner(const BlockDiagonalMGPreconditioner&) = delete;
    BlockDiagonalMGPreconditioner& operator=(const BlockDiagonalMGPreconditioner&) = delete;
    BlockDiagonalMGPreconditioner(BlockDiagonalMGPreconditioner&&) = default;
    BlockDiagonalMGPreconditioner& operator=(BlockDiagonalMGPreconditioner&&) = default;

    // z = M^-1 r, M^-1 = diag(M_A, M_A), M_A = two successive projected
    // positive V-cycles (see the file header). `r` is never mutated; applied
    // to c1 then c2, sequentially, on the same hierarchy. Throws
    // std::invalid_argument if either component's size does not match the
    // hierarchy's finest grid cell count.
    void apply(CudaContext& ctx, ConstCoupledVectorView r, CoupledVectorView z) const;

    // Additive, behavior-neutral byte introspection. Exact sum of the owned
    // per-level mean-zero workspace capacities plus the two finest-grid-sized
    // scratch buffers (see the file header). Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Host-only prediction of allocated_device_bytes() after constructing a
    // preconditioner over `hierarchy`; kept colocated with the constructor so
    // it cannot drift. Does not validate `hierarchy`.
    [[nodiscard]] static std::size_t estimate_device_bytes(const multigrid::MGHierarchy& hierarchy);

  private:
    // Applies M_A (two successive projected positive V-cycles) to one
    // n-sized component. `r`/`z` are exactly `hierarchy_->finest_grid().num_cells()`
    // elements; validated once by apply() before either component call.
    void apply_component(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const;

    multigrid::MGHierarchy* hierarchy_;
    multigrid::MGConfig config_;
    mutable std::vector<constraints::MeanZeroWorkspace> mean_zero_workspaces_;

    // Finest-grid-sized scratch: x1 stash (first cycle's projected solution)
    // and the defect-correction residual res = b0 - L(q) x1 (see the file
    // header). Both sized n = hierarchy.finest_grid().num_cells().
    mutable DeviceBuffer<real> x1_scratch_;
    mutable DeviceBuffer<real> residual_scratch_;
};

} // namespace streamfunctions
} // namespace macroflow3d
