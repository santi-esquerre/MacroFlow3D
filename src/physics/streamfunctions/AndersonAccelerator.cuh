#pragma once

/**
 * @file AndersonAccelerator.cuh
 * @brief SF-20 type-II Anderson acceleration over the coupled Lester
 *        equation (14) streamfunction state.
 *
 * `AndersonAccelerator` is an optional component of `StreamfunctionWorkspace`
 * (present only when `config.anderson.enabled`; see `AndersonConfig` in
 * `StreamfunctionTypes.hpp`), consumed by `solve_streamfunctions`
 * (`StreamfunctionSolver.cu`) to accelerate the SAME SF-15 adaptive-Picard
 * fixed-point map -- it never introduces a new fixed point and never bypasses
 * a safeguard; see `StreamfunctionSolver.cuh` for the exact per-iteration
 * insertion point and rejection/fallback semantics.
 *
 * Mathematical formulation (classical type-II Anderson mixing, beta = 1,
 * over the COUPLED state `X = [u1; u2]`, treated as one length-`2n` vector
 * split into two length-`n` device spans throughout this class):
 *
 *   - At outer iteration k, `x_k` is the current ACCEPTED coupled state and
 *     `u_hat_k` is that iteration's frozen Picard MAP output (the two block
 *     solves, unchanged from SF-14/SF-15). The fixed-point residual is
 *     `f_k = u_hat_k - x_k`.
 *   - The last (up to) `depth` pairs `(DeltaX_j = x_{j+1} - x_j,
 *     DeltaF_j = f_{j+1} - f_j)` are retained in a ring buffer (`m =
 *     num_columns()` currently held, `m <= depth`).
 *   - The accelerated candidate solves `gamma = argmin_gamma
 *     || f_k - F_k gamma ||_2` (`F_k` the DeltaF history matrix), then forms
 *     `x_acc = x_k + f_k - (X_k + F_k) gamma`. Since `f_k = u_hat_k - x_k`,
 *     this is algebraically identical to (and implemented as)
 *     `x_acc = u_hat_k - sum_j gamma_j * (DeltaX_j + DeltaF_j)`, which avoids
 *     ever materializing `x_k + f_k` separately.
 *
 * Least-squares solve: `gamma` solves the `m x m` NORMAL-EQUATIONS system
 * `(F_k^T F_k) gamma = F_k^T f_k` via a pivoted Householder QR of the Gram
 * matrix `F_k^T F_k` (see `AndersonAccelerator.cu`). Forming and factoring
 * the Gram matrix rather than QR-factoring `F_k` directly SQUARES the
 * effective condition number relative to a direct QR/SVD of `F_k` -- this is
 * the standard normal-equations conditioning caveat, explicitly accepted
 * here (per the SF-20 dashboard-locked design) because `m <= 8` is tiny, the
 * dot products needed to assemble the Gram system are cheap fixed-shape GPU
 * reductions (see below), and `condition_limit` (from the R-diagonal ratio
 * of the pivoted QR) is a HARD safeguard: any candidate whose estimate
 * exceeds `condition_limit` is rejected before ever touching device state,
 * and the caller clears the history and falls back to plain Picard
 * backtracking (see `StreamfunctionSolver.cuh`).
 *
 * GPU/host split: every Gram/rhs entry is a dot product of two coupled
 * (`length-n` component-1 + `length-n` component-2) history/residual
 * vectors, computed via `blas::dot_device` (cuBLAS `Ddot`, a deterministic
 * fixed-shape tree reduction for a fixed problem size -- no atomics, no
 * run-to-run reduction-order nondeterminism) into DISTINCT slots of one
 * small device scratch buffer, with NO intermediate host synchronization;
 * the whole buffer (at most `depth*(depth+3)` reals, independent of `n`) is
 * transferred to the host in one `cudaMemcpyAsync` + one `synchronize()` per
 * acceleration attempt. The QR factorization, back-substitution, and
 * condition estimate are then pure host arithmetic on fixed-size `[8][8]`
 * arrays (`kMaxAndersonDepth == 8`, matching `AndersonConfig::depth`'s
 * validated upper bound).
 *
 * Memory: `prepare()` performs all allocation once (idempotent for an
 * unchanged `(n, depth)` pair, matching every other accepted sub-workspace's
 * convention); no method below allocates. `history_bytes()` is EXACTLY
 * `4 * depth * n * sizeof(real)` (`DeltaX1`, `DeltaX2`, `DeltaF1`, `DeltaF2`,
 * each `depth * n` reals). `staging_bytes()` is EXACTLY `4 * n * sizeof(real)`
 * (the previous-accepted-state/previous-fixed-point-residual pair `prev_x1`,
 * `prev_x2`, `prev_f1`, `prev_f2`, each `n` reals -- needed to form the NEXT
 * history column without any additional scratch, see `update_history()`).
 * `scratch_bytes()` is the small `n`-independent dot-product transfer buffer
 * plus the (API-required but functionally unused by `blas::dot_device`)
 * `blas::ReductionWorkspace` footprint.
 *
 * m=1 condition-estimate property (decision 5(e)(ii) revision, T02-F1): when
 * exactly ONE history column is held (`num_columns() == 1`), the R-diagonal
 * condition estimate computed by `form_candidate()` is a max/min ratio over a
 * SINGLE diagonal entry, which is MATHEMATICALLY exactly `1.0` regardless of
 * that column's value -- there is no conditioning question to ask of a
 * single column. Consequently a hostile `condition_limit` (any validated
 * value is `> 1`) can never suppress a legitimate one-column (secant-step)
 * Anderson acceleration; the estimate always passes at `m == 1`. This is by
 * design, not a gap: safety at `m == 1` (and every other `m`) relies on the
 * full SF-15 trial guard chain that every accelerated candidate must still
 * pass in `solve_streamfunctions` before it is ever accepted into the
 * iterate -- `condition_limit` exists to reject ill-conditioned MULTI-column
 * (`m >= 2`) Gram systems, not to gate `m == 1`. See the `anderson_form_candidate_unit`
 * single-column unit contract and the amended `anderson_condition_reset_control`
 * fixture in `tests/streamfunctions/streamfunction_anderson_gpu_cases.cu`.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../numerics/blas/reduction_workspace.cuh"
#include "../../runtime/CudaContext.cuh"

#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {

// AndersonConfig::depth's validated upper bound (StreamfunctionTypes.hpp);
// also the fixed dimension of every host-side dense array used by the QR
// solve in AndersonAccelerator.cu.
constexpr int kMaxAndersonDepth = 8;

/**
 * Pure-host, allocation-free byte prediction mirroring exactly what
 * `AndersonAccelerator::prepare(n, depth)` allocates; see the file header
 * for the exact-byte contract of each field. Used by
 * `estimate_streamfunction_memory` (SF-12/SF-20).
 */
struct AndersonMemoryEstimate {
    std::size_t history_bytes{};
    std::size_t staging_bytes{};
    std::size_t scratch_bytes{};
};

[[nodiscard]] AndersonMemoryEstimate estimate_anderson_device_bytes(std::size_t n, int depth);

/**
 * Owns the SF-20 Anderson history/staging/scratch device storage for one
 * grid cell count `n` and one history `depth`. Move-only, matching every
 * other accepted streamfunction workspace/accelerator convention.
 */
class AndersonAccelerator {
  public:
    AndersonAccelerator() = default;
    ~AndersonAccelerator() = default;

    AndersonAccelerator(const AndersonAccelerator&) = delete;
    AndersonAccelerator& operator=(const AndersonAccelerator&) = delete;
    AndersonAccelerator(AndersonAccelerator&&) noexcept = default;
    AndersonAccelerator& operator=(AndersonAccelerator&&) noexcept = default;

    // Allocates/resizes every owned buffer for exactly (n, depth); a
    // repeated call for an already-prepared (n, depth) pair performs no
    // device allocation. Resets the history (equivalent to clear()).
    void prepare(std::size_t n, int depth);

    [[nodiscard]] bool prepared_for(std::size_t n, int depth) const noexcept;

    // Resets the retained history to empty (num_columns() == 0) without
    // deallocating, and forgets the staged previous (x, f) pair so the next
    // update_history() call re-bootstraps instead of forming a delta against
    // stale state. Called on the SF-20 REJECTED/ill-conditioned fallback
    // path (see StreamfunctionSolver.cuh).
    void clear() noexcept;

    [[nodiscard]] int num_columns() const noexcept { return num_columns_; }

    // Maintains the coupled (DeltaX, DeltaF) ring-buffer history. `x1`/`x2`
    // is the CURRENT accepted coupled state; `u_hat1`/`u_hat2` is this outer
    // iteration's frozen Picard MAP output (SF-14/SF-15 f1()/f2()). The
    // fixed-point residual f = u_hat - x is formed internally, without any
    // extra scratch buffer, by re-using the destination history column (see
    // AndersonAccelerator.cu for the exact op sequence). If a previous (x, f)
    // pair is already staged (i.e. this is not the first call since
    // construction/clear()), a new history column (x - x_prev, f - f_prev)
    // is written, evicting the oldest column once `depth` columns are held
    // (ring buffer; column order does not affect the least-squares solution).
    // The staged (x, f) pair is then unconditionally replaced by the current
    // (x, f). Must be called with exactly `n`-sized coupled-component spans
    // matching the (n, depth) this accelerator was prepared for.
    void update_history(CudaContext& context, DeviceSpan<const real> x1, DeviceSpan<const real> x2,
                        DeviceSpan<const real> u_hat1, DeviceSpan<const real> u_hat2);

    // Solves gamma = argmin_gamma || f_k - F_k gamma ||_2 for the currently
    // held history via the pivoted-QR Gram-matrix solve described in the
    // file header (f_k is the fixed-point residual staged by the most recent
    // update_history() call), and, on success, writes the UNPROJECTED
    // type-II Anderson candidate x_acc = u_hat - sum_j gamma_j*(DeltaX_j +
    // DeltaF_j) into out1/out2 -- the caller applies the mean-zero gauge
    // projection and the full SF-15 trial guard chain before ever touching
    // the accepted state. Returns false, leaving out1/out2 untouched, when
    // num_columns() == 0 or the R-diagonal condition estimate exceeds
    // condition_limit; `condition_estimate` always receives the computed
    // estimate (or +infinity if the solve was not attempted/degenerate).
    [[nodiscard]] bool form_candidate(CudaContext& context, DeviceSpan<const real> u_hat1,
                                      DeviceSpan<const real> u_hat2, real condition_limit,
                                      DeviceSpan<real> out1, DeviceSpan<real> out2,
                                      real& condition_estimate);

    // Exact owned-bytes accounting; see the file header. Never allocates.
    [[nodiscard]] std::size_t history_bytes() const noexcept;
    [[nodiscard]] std::size_t staging_bytes() const noexcept;
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;

  private:
    [[nodiscard]] DeviceSpan<real> column(DeviceBuffer<real>& buffer, int slot) noexcept;
    [[nodiscard]] DeviceSpan<const real> column(const DeviceBuffer<real>& buffer,
                                                int slot) const noexcept;

    std::size_t n_ = 0;
    int depth_ = 0;
    int num_columns_ = 0;
    long long total_pushes_ = 0;
    bool has_previous_ = false;

    // History: 4 * depth * n reals total (history_bytes()).
    DeviceBuffer<real> delta_x1_;
    DeviceBuffer<real> delta_x2_;
    DeviceBuffer<real> delta_f1_;
    DeviceBuffer<real> delta_f2_;

    // Staging: 4 * n reals total (staging_bytes()).
    DeviceBuffer<real> prev_x1_;
    DeviceBuffer<real> prev_x2_;
    DeviceBuffer<real> prev_f1_;
    DeviceBuffer<real> prev_f2_;

    // Small dense dot-product transfer buffer (scratch_bytes(), n-independent).
    DeviceBuffer<real> dot_scratch_;
    blas::ReductionWorkspace reduction_ws_; // unused by blas::dot_device; API completeness only.
};

} // namespace streamfunctions
} // namespace macroflow3d
