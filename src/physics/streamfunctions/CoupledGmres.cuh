#pragma once

/**
 * @file CoupledGmres.cuh
 * @brief SF-23 restarted, right-preconditioned GMRES for the coupled Newton
 *        linearization `J(Psi) delta = b`.
 *
 * `J(Psi)` is available EXCLUSIVELY through the accepted SF-22 matrix-free
 * `JvpWorkspace` (`JacobianVectorProduct.cuh`): every `J v` this file forms
 * is one `JvpWorkspace::apply(...)` call against a base state already
 * frozen by a preceding `prepare_jvp_base(...)` call. No second PDE
 * discretization, no Jacobian matrix, and no Newton/line-search/globalization
 * logic is introduced here (out of scope for SF-23; see the increment
 * specification, `docs/plans/active/lester-eq14/increments/SF-23-gmres-preconditioner.md`).
 * `J` is NONSYMMETRIC (the nonlinear source coupling breaks the block
 * diagonal structure the Picard linear subproblem has), which is exactly why
 * this file exists rather than reusing projected PCG.
 *
 * == E1: RIGHT preconditioning ==
 *
 * `solve()` solves `J(Psi) M^-1 u = b` and returns `delta = M^-1 u`. Right
 * preconditioning keeps the Givens-recurrence residual an estimate of the
 * TRUE unpreconditioned residual `||b - J delta||` (left preconditioning
 * would instead estimate `||M^-1(b - J delta)||`, a different norm), so the
 * "reported vs true residual" agreement check below is meaningful in one
 * consistent norm.
 *
 * == Deferred-preconditioner correction (why storage is (m+1) vectors, not
 *    (2m+1)) ==
 *
 * Standard right-preconditioned GMRES forms `x_m = x0 + sum_j y_j z_j` where
 * `z_j = M^-1 v_j`; storing every `z_j` for the final combination would
 * require `m` ADDITIONAL coupled vectors beyond the `(m+1)`-vector Krylov
 * basis `v_1..v_{m+1}`. Because `M^-1` here is a FIXED LINEAR map (E2,
 * `BlockDiagonalMGPreconditioner`, gated by a bitwise/superposition test, not
 * assumed), that sum can be reordered:
 *
 *   sum_j y_j z_j = sum_j y_j M^-1(v_j) = M^-1( sum_j y_j v_j )
 *
 * so the per-cycle correction is formed by ONE final `M^-1` application to
 * the host-computed linear combination `sum_j y_j v_j` (built entirely from
 * the STORED `v_j` basis), rather than by summing `m` stored preconditioned
 * directions. Only the `(m+1)`-vector `v` basis is stored across the whole
 * cycle; `z = M^-1 v_j` is still computed EVERY Arnoldi iteration (it is
 * needed transiently to form `w = J(z)`), but that transient result lives in
 * one reused `n`-independent-count work buffer (`z_` below), never in
 * per-iteration history. This is the exact mechanism behind the SF-23
 * dashboard-locked memory formula `2*(m+1)*n*sizeof(real)` for the basis
 * (see `estimate_device_bytes` below): at `256^3` (`n = 16777216`), `m=10`,
 * that is `2*11*16777216*8 = 2952790016` bytes, ~2.75 GiB, matching the
 * spec's threshold exactly.
 *
 * == E3/E4: algorithm ==
 *
 * `b` is projected into an OWNED workspace copy on entry (`rhs_ = P(b)`; the
 * caller's `b` view is never mutated). Each outer (restart) cycle:
 *
 *  1. `r0 = P(rhs_ - J(x_total))` (one `JvpWorkspace::apply` call against the
 *     RUNNING correction `x_total`, which starts at zero); `beta = ||r0||`.
 *     TRUE residual recomputed HERE at every restart AND at the final
 *     termination (one extra call after the loop if the last cycle's own
 *     recomputation is not already the terminal one) -- see
 *     `CoupledGmresResidualCheckpoint`. EXCEPT: on the very first cycle
 *     `x_total` is still identically zero (the initial `blas::fill(...,
 *     real{0})`), and `JvpWorkspace::apply` THROWS `std::invalid_argument` on
 *     a zero weighted-direction norm (SF-22 D2 contract, intentionally NOT
 *     special-cased inside `JvpWorkspace` -- see `JacobianVectorProduct.cuh`).
 *     `J(0) = 0` algebraically, so `compute_true_residual` tracks a host-side
 *     `correction_is_zero` flag (true until the first cycle successfully
 *     accumulates a nonzero correction into `x_total`, i.e. false from the
 *     first non-singular back-substitution onward; a singular
 *     back-substitution skips accumulation and leaves the flag unchanged)
 *     and, whenever it is true, SKIPS the `JvpWorkspace::apply` call
 *     entirely: the true residual is simply `||P(rhs_)||`, computed by
 *     copying the already-projected `rhs_` into the scratch output span (the
 *     caller then seeds `v_1` from that same scratch span, so the copy keeps
 *     that seeding path correct without a separate case).
 *
 * == T01-F1/F2 fix note ==
 *
 * `correction_is_zero` above is a `solve()`-local host `bool` (reset to
 * `true` at the start of every `solve()` call, not a persistent
 * `CoupledGmres` member): each `solve()` call restarts `x_total` at zero, so
 * there is nothing to persist across calls, and keeping it local avoids any
 * possibility of a stale flag leaking between unrelated `solve()`
 * invocations on the same (reusable) `CoupledGmres` object. It is threaded
 * into `compute_true_residual` as an explicit parameter (not a member)
 * because the value differs at the two call sites within one `solve()` (the
 * per-cycle recompute and the post-loop termination recompute) depending on
 * whether a correction has been accumulated by that point.
 *  2. `v_1 = r0 / beta` (the first Krylov basis vector).
 *  3. For `j = 1..m` (bounded by the total inner-iteration budget too):
 *     `z = M^-1 v_j` (right preconditioner, `BlockDiagonalMGPreconditioner`
 *     or any type satisfying the same duck-typed
 *     `apply(CudaContext&, ConstCoupledVectorView, CoupledVectorView) const`
 *     interface -- `solve()` is a template exactly so no preconditioner base
 *     class/vtable is required, matching `operator_concept.cuh`'s documented
 *     convention); `w = J(z)` (one `JvpWorkspace::apply` call, `++total
 *     inner iterations`); `w` is projected component-wise (E4: projected
 *     AFTER the `J*M^-1` application, BEFORE orthogonalization). Modified
 *     Gram-Schmidt against `v_1..v_j` (one host scalar transfer per
 *     dot-product-then-subtract step -- MGS is inherently sequential, each
 *     subtraction depends on the previous; this file follows the existing
 *     `AndersonAccelerator`/`JvpWorkspace` convention of small, explicit,
 *     documented per-step host syncs rather than hiding them). If the
 *     classical norm-drop criterion fires
 *     (`||w||_after < (1/sqrt(2)) * ||w||_before`), one ADDITIONAL MGS pass
 *     is run and the event is counted
 *     (`CoupledGmresReport::reorthogonalization_events`). `h_{j+1,j} =
 *     ||w||_after` (post-(re)orthogonalization); if it is (numerically) zero,
 *     GMRES has exhausted this cycle's Krylov subspace ("happy"/"lucky"
 *     breakdown): `v_{j+1}` is NOT formed (the divide would be singular);
 *     see the breakdown-resolution note below for how this is told apart
 *     from a genuine stall. Otherwise `v_{j+1} = w / h_{j+1,j}`. A Givens
 *     rotation zeroes `H[j+1][j]`, updates the least-squares residual
 *     recurrence `g`, and the REPORTED residual for this iteration is
 *     `|g[j+1]|`.
 *  4. When the cycle ends (reported residual within tolerance, the subspace
 *     is exhausted, or the restart size `m` is reached), back-substitute the
 *     small upper-triangular system for `y`, form `combined_v = sum_j y_j
 *     v_j` (device axpy chain over the stored basis), apply the
 *     preconditioner ONCE MORE to get this cycle's correction `M^-1
 *     (combined_v)`, and accumulate it into `x_total` (E4: the accumulated
 *     `x_total` is re-projected after every accumulation -- "the returned
 *     correction projected").
 *  5. Recompute the TRUE residual `||P(rhs_ - J(x_total))||` (step 1 of the
 *     next cycle, or the final termination checkpoint if the loop is about
 *     to stop) and check it against `rel_tol * ||rhs_||`.
 *
 * == E3 residual checkpoints: what is actually being compared (T01-F1) ==
 *
 * The point of recording BOTH a "reported" and a "true" residual per
 * checkpoint is to compare the cheap Givens-recurrence estimate against
 * reality -- so the two fields must never be computed from the same
 * quantity. Each checkpoint is therefore paired as follows:
 *
 *  - The very FIRST checkpoint, taken before any inner iteration has run
 *    (`total_inner_iterations == 0`), has no prior cycle's recurrence
 *    residual to compare against yet, so it records `{reported = beta, true
 *    = beta}` (both fields ARE the freshly recomputed true residual). This
 *    checkpoint is EXCLUDED from E3 reported-vs-true agreement gating by the
 *    `total_inner_iterations == 0` convention.
 *  - Every checkpoint taken at the START of a LATER restart cycle (`k >= 1`,
 *    i.e. a previous cycle actually ran to completion) records `{reported =
 *    that PREVIOUS cycle's final Givens-recurrence residual (`|g[j+1]|` at
 *    the column where it stopped), true = the freshly recomputed beta for
 *    THIS cycle}`. This is the meaningful comparison: it measures how far
 *    the recurrence estimate the previous cycle trusted had drifted from the
 *    true residual measured independently afterward.
 *  - The early-exit checkpoint (pre-cycle `beta <= tol` immediately
 *    terminates the solve without running a cycle) is a restart checkpoint
 *    like any other and follows the SAME rule above: the `k >= 1` pairing if
 *    a previous cycle ran, or the first-checkpoint rule if it is itself the
 *    very first checkpoint.
 *  - The termination checkpoint (max_iterations budget exhausted, or a
 *    genuine breakdown) records `{reported = the last cycle's final
 *    Givens-recurrence residual, true = the freshly recomputed final true
 *    residual}` -- the same pairing rule as the `k >= 1` case, just taken
 *    after the loop instead of before the next cycle.
 *
 * `solve()` carries the last cycle's final recurrence residual in a single
 * host `real` that survives across outer-loop iterations (assigned to
 * `beta` at the start of each cycle, then updated to `|g[j+1]|` on every
 * Arnoldi column); the checkpoint push at the top of each cycle reads that
 * variable's value from the PREVIOUS cycle before overwriting it for the
 * current one.
 *
 * == Breakdown resolution (spec: "breakdown(happy or lucky handled
 *    correctly)") ==
 *
 * A near-zero `h_{j+1,j}` (see step 3) means THIS cycle's Krylov subspace
 * cannot be extended further. If the REPORTED residual at that same column
 * already satisfies the tolerance, this is a "happy" breakdown: the exact
 * (subspace-restricted) solution has been found, and the solve resolves
 * through the ordinary convergence check to `CoupledGmresStatus::converged`
 * -- no special status value is needed for that case. If it does NOT satisfy
 * the tolerance, this is a genuine stall (a "lucky" breakdown that turned out
 * unlucky): the cycle stops early with whatever correction it has, and the
 * WHOLE solve terminates immediately with `CoupledGmresStatus::breakdown`
 * (a fresh restart from the same stalled `x_total` would rebuild the exact
 * same degenerate subspace, so retrying is not attempted here).
 *
 * == Memory ==
 *
 * `prepare(n, m)` allocates: the `(m+1)`-vector basis (`2*(m+1)*n` reals,
 * the dominant term, see above); FOUR `n`-independent-in-count work vectors
 * of `2*n` reals each (`w_`, `z_`, `residual_scratch_`, `correction_total_`)
 * plus the projected-rhs workspace copy `rhs_` (E4) -- five `2*n`-real
 * buffers total; and the `n`-sized `MeanZeroWorkspace` this file's
 * component-wise projections share. No allocation occurs in `solve()`.
 * `estimate_device_bytes(n, m)` mirrors this exactly.
 *
 * == SF-25 T01: duck-typed JacobianOperator (E1, D-gate enabler) ==
 *
 * `solve()` and its private `compute_true_residual` helper are templated on
 * a SECOND parameter, `JacobianOperator`, defaulted to `JvpWorkspace` (the
 * SAME pattern this file's `Preconditioner` template parameter already
 * uses, see E3/E4 above). A `JacobianOperator` must provide
 *
 *   JvpApplyReport apply(CudaContext&, const Grid3D&, ConstCoupledVectorView
 *                        direction, const JvpDeltaConfig&, CoupledVectorView
 *                        jv_out);
 *
 * with the SAME D2 (forward-difference delta policy)/D3 (projection
 * discipline; direction never mutated, output not re-projected) semantics
 * `JvpWorkspace::apply` documents -- `JvpWorkspace` itself is the canonical
 * instance, and every accepted call site (`gmres.solve(ctx, grid, jvp,
 * preconditioner, ...)`) passes a `JvpWorkspace&` unchanged, so template
 * argument deduction resolves `JacobianOperator = JvpWorkspace` exactly as
 * before this parameter was added: this generalization is
 * behavior-neutral for every accepted call site. It exists so a
 * `ShiftedJacobianOperator` (`ShiftedJacobianOperator.cuh`, the shifted
 * system `(mu*A_blk + J) p` the SF-25 manifold-robust terminal solver
 * needs) can be solved by the SAME accepted GMRES machinery without
 * duplicating it. No other `solve()` logic changes with this
 * generalization: the algorithm, checkpoints, breakdown resolution, and
 * memory contract above are unchanged.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../numerics/blas/blas.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "JacobianVectorProduct.cuh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {

// Validated upper bound on the restart size (SF-23 spec: default 10, 15
// permitted only through explicit config); also the fixed dimension of every
// host-side Hessenberg/Givens array `solve()` uses.
constexpr int kMaxCoupledGmresRestart = 15;

/**
 * `restart` (`m`) must satisfy `1 <= m <= kMaxCoupledGmresRestart`; values
 * above the SF-23-dashboard-locked default of 10 are permitted only when the
 * caller sets this explicitly (this struct does not itself distinguish
 * "explicit" from "default" -- that is a caller/config-loading concern; see
 * the increment specification). `max_iterations` bounds the TOTAL inner
 * (Arnoldi) iteration count across every restart cycle. `rel_tol` is the
 * relative tolerance on the reported residual against `||P(b)||` (default
 * `1e-10`, matching the dashboard-locked linear rtol).
 */
struct CoupledGmresConfig {
    int restart{10};
    int max_iterations{100};
    real rel_tol{real{1e-10}};
};

/** Throws `std::invalid_argument` if `config` violates its own contract. */
void validate_coupled_gmres_config(const CoupledGmresConfig& config);

enum class CoupledGmresStatus { converged, max_iterations, breakdown, nonfinite };

/**
 * One true-vs-reported residual sample, taken at the start of every restart
 * cycle (the recomputed true residual driving that cycle's initial `v_1`)
 * and once more at termination (see the file header, E3). `reported_residual`
 * and `true_residual` are NOT computed from the same quantity except for the
 * very first checkpoint: `reported_residual` is the PREVIOUS cycle's final
 * Givens-recurrence estimate (or, for the very first checkpoint only, the
 * same freshly recomputed value as `true_residual`, since no previous
 * cycle exists yet), while `true_residual` is always the independently
 * recomputed `||P(rhs_ - J(x_total))||` for the current point. See the file
 * header's "E3 residual checkpoints" section for the exact per-exit-path
 * pairing rules.
 * `total_inner_iterations` is the CUMULATIVE inner-iteration count at the
 * moment this checkpoint was taken (0 for the very first, pre-iteration
 * checkpoint; that checkpoint is EXCLUDED from reported-vs-true agreement
 * gating by this `== 0` convention, since it has no meaningful prior
 * recurrence estimate to compare).
 */
struct CoupledGmresResidualCheckpoint {
    int total_inner_iterations{};
    real reported_residual{};
    real true_residual{};
};

struct CoupledGmresReport {
    CoupledGmresStatus status{CoupledGmresStatus::max_iterations};
    int total_inner_iterations{};
    int outer_cycles{};
    long long reorthogonalization_events{};
    std::vector<CoupledGmresResidualCheckpoint> checkpoints;
};

/**
 * Owns every scratch buffer needed to run restarted, right-preconditioned
 * GMRES for one grid cell count `n` and one EXACT restart size `m` (mirrors
 * `AndersonAccelerator`'s `prepare(n, depth)` convention: `solve()` requires
 * `prepared_for(n, config.restart)`, i.e. `config.restart` must equal the
 * `m` this object was prepared for -- `prepare()` is not a "reserve enough
 * for any future m up to this" call). Move-only, matching every other
 * accepted streamfunction workspace convention.
 */
class CoupledGmres {
  public:
    CoupledGmres() = default;
    ~CoupledGmres() = default;

    CoupledGmres(const CoupledGmres&) = delete;
    CoupledGmres& operator=(const CoupledGmres&) = delete;
    CoupledGmres(CoupledGmres&&) noexcept = default;
    CoupledGmres& operator=(CoupledGmres&&) noexcept = default;

    // Allocates/resizes every owned buffer for exactly (n, m); throws
    // std::invalid_argument if n == 0 or m is outside [1,
    // kMaxCoupledGmresRestart]. Idempotent for an already-prepared (n, m)
    // pair (no device allocation).
    void prepare(std::size_t n, int m);

    [[nodiscard]] bool prepared_for(std::size_t n, int m) const noexcept;

    // Solves J(Psi) M^-1 u = b (right preconditioning, E1) approximately for
    // the base state most recently frozen by jvp.prepare_jvp_base(...), and
    // writes delta = M^-1 u into `correction`. `b`/`correction` must be
    // exactly `n` elements per component (the `n` this object was prepared
    // for); `b` is never mutated (E4: projected into an owned workspace
    // copy). `preconditioner` must provide
    // `apply(CudaContext&, ConstCoupledVectorView, CoupledVectorView) const`
    // (duck-typed, matching `operator_concept.cuh`'s documented convention;
    // `BlockDiagonalMGPreconditioner` is the SF-23 concrete instance).
    // Throws std::invalid_argument for a config/size mismatch;
    // std::logic_error if !prepared_for(n, config.restart). See the file
    // header for the full algorithm, the breakdown-resolution rule, and the
    // memory contract. `JacobianOperator` is duck-typed (SF-25 T01, E1; see
    // the file header) and defaults to `JvpWorkspace`, so every accepted
    // call site (passing a `JvpWorkspace&`) is unchanged.
    template <typename Preconditioner, typename JacobianOperator = JvpWorkspace>
    CoupledGmresReport solve(CudaContext& ctx, const Grid3D& grid, JacobianOperator& jvp,
                             const Preconditioner& preconditioner, ConstCoupledVectorView b,
                             const CoupledGmresConfig& config, const JvpDeltaConfig& delta_config,
                             CoupledVectorView correction);

    // Exact sum of every owned DeviceBuffer capacity, plus the owned
    // MeanZeroWorkspace/ReductionWorkspace footprints. Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Host-only prediction of allocated_device_bytes() after a fresh
    // prepare(n, m) call; kept colocated with prepare() so it cannot drift.
    [[nodiscard]] static std::size_t estimate_device_bytes(std::size_t n, int m);

  private:
    // -- Non-template internals (Preconditioner-independent); defined in
    // CoupledGmres.cu. --
    void ensure_prepared() const;
    [[nodiscard]] DeviceSpan<real> basis_vector(int index) noexcept;
    [[nodiscard]] DeviceSpan<const real> basis_vector(int index) const noexcept;
    void project_pair(CudaContext& ctx, DeviceSpan<real> c1, DeviceSpan<real> c2) const;
    [[nodiscard]] real dot2n(CudaContext& ctx, DeviceSpan<const real> a,
                             DeviceSpan<const real> b) const;
    [[nodiscard]] real norm2n(CudaContext& ctx, DeviceSpan<const real> a) const;
    void project_rhs(CudaContext& ctx, ConstCoupledVectorView b);

    // r_out = P(rhs_ - J(x_total)); returns ||r_out||; `ok` is false iff the
    // underlying JacobianOperator::apply reported a non-finite perturbed
    // residual (see JacobianVectorProduct.cuh). One JacobianOperator::apply
    // call (does NOT increment the caller's inner-iteration counter --
    // residual recomputation is accounted separately from Arnoldi
    // iterations, per the SF-23 spec's checkpoint list) -- EXCEPT when
    // `correction_is_zero` is true (T01-F2): `x_total` is then identically
    // zero, `J(0) = 0` algebraically, and calling `JacobianOperator::apply`
    // with a zero direction would throw (SF-22 D2 contract, kept as-is in
    // JvpWorkspace itself), so this skips the apply call entirely and
    // returns `||P(rhs_)||` (`r_out` is set to a copy of the already-
    // projected `rhs_`). See the file header for the full rationale.
    // Templated on `JacobianOperator` (SF-25 T01, E1; defined here in the
    // header, not in CoupledGmres.cu, so it can be instantiated for any
    // caller-supplied JacobianOperator type) -- byte-identical logic to the
    // pre-T01 non-template `JvpWorkspace&`-only version.
    template <typename JacobianOperator>
    [[nodiscard]] real compute_true_residual(CudaContext& ctx, const Grid3D& grid,
                                             JacobianOperator& jvp,
                                             const JvpDeltaConfig& delta_config,
                                             DeviceSpan<const real> x_total_flat,
                                             DeviceSpan<real> r_out_flat,
                                             bool correction_is_zero, bool& ok);

    std::size_t n_ = 0;
    int m_ = 0;

    // (m+1)-vector contiguous Krylov basis, 2*(m+1)*n reals (dominant memory
    // term; see the file header).
    DeviceBuffer<real> basis_;

    // Work vectors (each 2*n reals; see the file header for the exact list).
    DeviceBuffer<real> w_;
    DeviceBuffer<real> z_;
    DeviceBuffer<real> residual_scratch_;
    DeviceBuffer<real> correction_total_;
    DeviceBuffer<real> rhs_;

    // mutable: project_pair()/dot2n()/norm2n() are const helper methods (the
    // coupled linear-algebra "read" operations they support are logically
    // const from solve()'s point of view) but the reused reduction/mean-zero
    // scratch they enqueue work into is, like every other accepted
    // workspace's scratch state, mutated in place.
    mutable constraints::MeanZeroWorkspace projection_workspace_;
    mutable blas::ReductionWorkspace reduction_workspace_;
};

template <typename JacobianOperator>
real CoupledGmres::compute_true_residual(CudaContext& ctx, const Grid3D& grid,
                                         JacobianOperator& jvp,
                                         const JvpDeltaConfig& delta_config,
                                         DeviceSpan<const real> x_total_flat,
                                         DeviceSpan<real> r_out_flat, bool correction_is_zero,
                                         bool& ok) {
    if (correction_is_zero) {
        // T01-F2: x_total is identically zero here (the initial
        // blas::fill(..., real{0}), no cycle has accumulated a correction
        // yet). J(0) = 0 algebraically, so the true residual is simply
        // ||P(rhs_)|| -- and this MUST skip JacobianOperator::apply rather
        // than just optimize it away: apply() throws std::invalid_argument
        // on a zero weighted-direction norm (SF-22 D2 contract, unchanged in
        // JvpWorkspace itself, see JacobianVectorProduct.cuh), so calling it
        // with x_total_flat == 0 would abort every solve() call on its very
        // first cycle. rhs_ is already projected (project_rhs()), so a plain
        // copy keeps r_out_flat consistent with the caller's use of
        // residual_scratch_ as the v_1 seed.
        ok = true;
        blas::copy(ctx, DeviceSpan<const real>(rhs_.span()), r_out_flat);
        return norm2n(ctx, DeviceSpan<const real>(r_out_flat));
    }

    // r_out <- J(x_total) (one JacobianOperator::apply call; NOT counted in
    // the caller's inner-iteration budget -- residual recomputation is a
    // separate, explicitly checkpointed activity, see the file header).
    ConstCoupledVectorView x_view = split_coupled_vector(x_total_flat, n_);
    CoupledVectorView jx_view = split_coupled_vector(r_out_flat, n_);
    JvpApplyReport jr = jvp.apply(ctx, grid, x_view, delta_config, jx_view);
    if (jr.status != JvpApplyStatus::ok) {
        ok = false;
        return real{0};
    }
    ok = true;

    // r_out <- P(rhs_) - J(x_total).
    blas::scal(ctx, r_out_flat, real{-1});
    blas::axpy(ctx, real{1}, DeviceSpan<const real>(rhs_.span()), r_out_flat);
    project_pair(ctx, DeviceSpan<real>(r_out_flat.data(), n_),
                DeviceSpan<real>(r_out_flat.data() + n_, n_));
    return norm2n(ctx, DeviceSpan<const real>(r_out_flat));
}

template <typename Preconditioner, typename JacobianOperator>
CoupledGmresReport CoupledGmres::solve(CudaContext& ctx, const Grid3D& grid,
                                       JacobianOperator& jvp,
                                       const Preconditioner& preconditioner,
                                       ConstCoupledVectorView b, const CoupledGmresConfig& config,
                                       const JvpDeltaConfig& delta_config,
                                       CoupledVectorView correction) {
    validate_coupled_gmres_config(config);
    ensure_prepared();
    if (!prepared_for(n_, config.restart)) {
        throw std::logic_error(
            "CoupledGmres::solve requires prepare(n, config.restart) with the EXACT restart size "
            "used by this config");
    }
    if (b.c1.size() != n_ || b.c2.size() != n_ || DeviceSpan<real>(correction.c1).size() != n_ ||
        DeviceSpan<real>(correction.c2).size() != n_) {
        throw std::invalid_argument(
            "CoupledGmres::solve requires b/correction coupled components of exactly n elements");
    }
    validate_jvp_delta_config(delta_config);

    const int m = config.restart;
    const real half_root_two_inv = real{1} / std::sqrt(real{2});
    // Absolute+relative guard distinguishing a genuine (near-)zero Arnoldi
    // norm from ordinary floating-point noise; see the file header's
    // breakdown-resolution note.
    const real breakdown_eps = real{100} * std::numeric_limits<real>::epsilon();

    project_rhs(ctx, b);
    const real rhs_norm = norm2n(ctx, DeviceSpan<const real>(rhs_.span()));

    blas::fill(ctx, correction_total_.span(), real{0});

    CoupledGmresReport report;

    if (!std::isfinite(rhs_norm)) {
        report.status = CoupledGmresStatus::nonfinite;
        blas::copy(ctx, DeviceSpan<const real>(correction_total_.data(), n_), correction.c1);
        blas::copy(ctx, DeviceSpan<const real>(correction_total_.data() + n_, n_), correction.c2);
        return report;
    }
    if (rhs_norm == real{0}) {
        report.status = CoupledGmresStatus::converged;
        report.checkpoints.push_back(CoupledGmresResidualCheckpoint{0, real{0}, real{0}});
        blas::copy(ctx, DeviceSpan<const real>(correction_total_.data(), n_), correction.c1);
        blas::copy(ctx, DeviceSpan<const real>(correction_total_.data() + n_, n_), correction.c2);
        return report;
    }

    int total_iters = 0;
    bool converged = false;
    bool broke = false;
    bool nonfinite_flag = false;
    // T01-F2: true on the first cycle only (x_total == 0 identically, so
    // J(x_total) must not be formed via JvpWorkspace::apply -- see the file
    // header and compute_true_residual's doc comment). Flips to false the
    // first time a cycle successfully accumulates a nonzero correction into
    // x_total (a singular back-substitution skips accumulation and leaves
    // this flag unchanged). solve()-local, not a CoupledGmres member: reset
    // fresh on every solve() call, see the file header.
    bool correction_is_zero = true;
    // T01-F1: the last completed cycle's final Givens-recurrence residual,
    // carried across outer-loop iterations so the checkpoint pushed at the
    // START of the next cycle can report the PREVIOUS cycle's recurrence
    // estimate paired against a freshly (independently) recomputed true
    // residual -- see the file header's "E3 residual checkpoints" section.
    real reported_residual = real{0};

    while (!converged && !broke && !nonfinite_flag && total_iters < config.max_iterations) {
        ++report.outer_cycles;

        bool ok = true;
        real beta = compute_true_residual(ctx, grid, jvp, delta_config,
                                          DeviceSpan<const real>(correction_total_.span()),
                                          residual_scratch_.span(), correction_is_zero, ok);
        if (!ok || !std::isfinite(beta)) {
            nonfinite_flag = true;
            break;
        }
        if (report.outer_cycles == 1) {
            // First checkpoint: no previous cycle's recurrence residual
            // exists yet, excluded from agreement gating by the
            // total_inner_iterations == 0 convention (see the file header).
            report.checkpoints.push_back(CoupledGmresResidualCheckpoint{total_iters, beta, beta});
        } else {
            // reported_residual still holds the PREVIOUS cycle's final
            // recurrence residual here (not yet overwritten for this
            // cycle) -- this is the meaningful reported-vs-true pairing.
            report.checkpoints.push_back(
                CoupledGmresResidualCheckpoint{total_iters, reported_residual, beta});
        }
        if (beta <= config.rel_tol * rhs_norm) {
            converged = true;
            break;
        }

        // v_1 = r0 / beta.
        blas::copy(ctx, DeviceSpan<const real>(residual_scratch_.span()), basis_vector(0));
        blas::scal(ctx, basis_vector(0), real{1} / beta);

        real H[kMaxCoupledGmresRestart + 1][kMaxCoupledGmresRestart] = {};
        real g[kMaxCoupledGmresRestart + 1] = {};
        real cs[kMaxCoupledGmresRestart] = {};
        real sn[kMaxCoupledGmresRestart] = {};
        g[0] = beta;

        int completed = 0;
        reported_residual = beta;
        bool cycle_breakdown = false;

        for (int j = 0; j < m && total_iters < config.max_iterations; ++j) {
            CoupledVectorView vj = split_coupled_vector(basis_vector(j), n_);
            CoupledVectorView zview = split_coupled_vector(z_.span(), n_);
            preconditioner.apply(ctx, ConstCoupledVectorView(vj), zview);

            CoupledVectorView wview = split_coupled_vector(w_.span(), n_);
            JvpApplyReport jr = jvp.apply(ctx, grid, ConstCoupledVectorView(zview), delta_config,
                                          wview);
            ++total_iters;
            if (jr.status != JvpApplyStatus::ok) {
                nonfinite_flag = true;
                break;
            }
            project_pair(ctx, wview.c1, wview.c2);

            real w_norm_before = norm2n(ctx, DeviceSpan<const real>(w_.span()));
            if (!std::isfinite(w_norm_before)) {
                nonfinite_flag = true;
                break;
            }
            for (int i = 0; i <= j; ++i) {
                const real h =
                    dot2n(ctx, basis_vector(i), DeviceSpan<const real>(w_.span()));
                H[i][j] = h;
                blas::axpy(ctx, -h, basis_vector(i), w_.span());
            }
            real w_norm_after = norm2n(ctx, DeviceSpan<const real>(w_.span()));
            if (std::isfinite(w_norm_before) && w_norm_before > real{0} &&
                w_norm_after < half_root_two_inv * w_norm_before) {
                ++report.reorthogonalization_events;
                for (int i = 0; i <= j; ++i) {
                    const real h2 =
                        dot2n(ctx, basis_vector(i), DeviceSpan<const real>(w_.span()));
                    H[i][j] += h2;
                    blas::axpy(ctx, -h2, basis_vector(i), w_.span());
                }
                w_norm_after = norm2n(ctx, DeviceSpan<const real>(w_.span()));
            }
            if (!std::isfinite(w_norm_after)) {
                nonfinite_flag = true;
                break;
            }
            H[j + 1][j] = w_norm_after;

            const bool subspace_exhausted =
                w_norm_after <= breakdown_eps * std::max(real{1}, beta);
            if (!subspace_exhausted) {
                blas::copy(ctx, DeviceSpan<const real>(w_.span()), basis_vector(j + 1));
                blas::scal(ctx, basis_vector(j + 1), real{1} / w_norm_after);
            }

            for (int i = 0; i < j; ++i) {
                const real temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                H[i][j] = temp;
            }
            const real denom = std::sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
            if (denom > real{0}) {
                cs[j] = H[j][j] / denom;
                sn[j] = H[j + 1][j] / denom;
            } else {
                cs[j] = real{1};
                sn[j] = real{0};
            }
            H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
            H[j + 1][j] = real{0};

            const real g_new = cs[j] * g[j] + sn[j] * g[j + 1];
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
            g[j] = g_new;

            reported_residual = std::abs(g[j + 1]);
            completed = j + 1;

            const bool cycle_tolerance_met = reported_residual <= config.rel_tol * rhs_norm;
            if (subspace_exhausted) {
                if (!cycle_tolerance_met) {
                    cycle_breakdown = true;
                }
                break;
            }
            if (cycle_tolerance_met) {
                break;
            }
        }

        if (nonfinite_flag) {
            break;
        }
        if (completed == 0) {
            // No Arnoldi iteration could run at all (max_iterations budget
            // already exhausted at cycle start): stop, status resolved by
            // the outer max_iterations check below.
            break;
        }

        // Back-substitution: H (upper-left `completed x completed`) y = g.
        real y[kMaxCoupledGmresRestart] = {};
        bool singular = false;
        for (int k = completed - 1; k >= 0; --k) {
            real sum = g[k];
            for (int c = k + 1; c < completed; ++c) {
                sum -= H[k][c] * y[c];
            }
            if (H[k][k] == real{0}) {
                singular = true;
                break;
            }
            y[k] = sum / H[k][k];
        }

        if (!singular) {
            // combined_v = sum_k y_k v_k, deferred-preconditioner trick (see
            // the file header): reuse w_ as the accumulation buffer.
            blas::fill(ctx, w_.span(), real{0});
            for (int k = 0; k < completed; ++k) {
                blas::axpy(ctx, y[k], basis_vector(k), w_.span());
            }
            CoupledVectorView combined = split_coupled_vector(w_.span(), n_);
            CoupledVectorView cycle_correction = split_coupled_vector(z_.span(), n_);
            preconditioner.apply(ctx, ConstCoupledVectorView(combined), cycle_correction);

            blas::axpy(ctx, real{1}, DeviceSpan<const real>(z_.span()), correction_total_.span());
            project_pair(ctx, DeviceSpan<real>(correction_total_.data(), n_),
                        DeviceSpan<real>(correction_total_.data() + n_, n_));
            // T01-F2: x_total is no longer identically zero from here on.
            correction_is_zero = false;
        }

        if (singular || cycle_breakdown) {
            broke = true;
        }
    }

    // The pre-cycle "beta <= tol" branch above already recomputed the TRUE
    // residual and recorded it as a checkpoint immediately before breaking
    // the loop, so it already IS the termination checkpoint; every other
    // exit (max_iterations budget exhausted, or a genuine breakdown) needs
    // ONE more true-residual recomputation here for the termination
    // checkpoint. A breakdown-terminated cycle's post-correction true
    // residual is checked here too and can still resolve to `converged`
    // (the authoritative test is always the TRUE residual, never the
    // reported one) -- see the file header's breakdown-resolution note.
    if (!nonfinite_flag && !converged) {
        bool ok = true;
        real final_true = compute_true_residual(ctx, grid, jvp, delta_config,
                                                DeviceSpan<const real>(correction_total_.span()),
                                                residual_scratch_.span(), correction_is_zero, ok);
        if (!ok || !std::isfinite(final_true)) {
            nonfinite_flag = true;
        } else {
            // T01-F1: pair the LAST completed cycle's final recurrence
            // residual against this freshly recomputed true residual (same
            // rule as the k >= 1 restart checkpoint above), not against
            // itself.
            report.checkpoints.push_back(
                CoupledGmresResidualCheckpoint{total_iters, reported_residual, final_true});
            if (final_true <= config.rel_tol * rhs_norm) {
                converged = true;
            }
        }
    }

    if (nonfinite_flag) {
        report.status = CoupledGmresStatus::nonfinite;
    } else if (converged) {
        report.status = CoupledGmresStatus::converged;
    } else if (broke) {
        report.status = CoupledGmresStatus::breakdown;
    } else {
        report.status = CoupledGmresStatus::max_iterations;
    }
    report.total_inner_iterations = total_iters;

    blas::copy(ctx, DeviceSpan<const real>(correction_total_.data(), n_), correction.c1);
    blas::copy(ctx, DeviceSpan<const real>(correction_total_.data() + n_, n_), correction.c2);
    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
