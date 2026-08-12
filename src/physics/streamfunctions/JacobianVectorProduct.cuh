#pragma once

/**
 * @file JacobianVectorProduct.cuh
 * @brief Matrix-free Jacobian-vector product for the coupled Lester equation
 *        (14) streamfunction residual (SF-22).
 *
 * The accepted coupled residual (`ResidualEvaluator.cuh`) is
 *
 *   F1(u1, u2) = A u1 - P( rhs_affine1 - eta * q .* S2(u1, u2) )
 *   F2(u1, u2) = A u2 - P( rhs_affine2 - eta * q .* S1(u1, u2) )
 *
 * treated here as one coupled map `F(Psi) = (F1, F2)` of the coupled state
 * `Psi = (u1, u2)`. This module exposes the Jacobian ACTION `J(Psi) p` by a
 * forward difference over that SAME accepted evaluator, reused verbatim on a
 * perturbed state:
 *
 *   J(Psi) p ~= ( F(Psi + delta*p) - F(Psi) ) / delta
 *
 * No second PDE discretization, no Jacobian matrix, no Hessian-field storage,
 * and no Krylov/Newton driver are introduced anywhere in this file; those are
 * explicitly out of scope for SF-22. Every differential/linear-algebra
 * primitive this file uses is reused exactly as accepted in SF-02..SF-21.
 *
 * == CoupledVectorView (spec item 1) ==
 *
 * `CoupledVectorView` pairs two `DeviceSpan<real>` components without
 * copying, so that a persistent `StreamfunctionFields` instance (owning `u1`,
 * `u2` separately) and a contiguous `2n`-element Krylov-style buffer can both
 * be presented through the SAME light view type. Layout contract for a
 * contiguous coupled buffer of size `2n` (see `split_coupled_vector` below):
 * component 0 (psi1-fluctuation-like) occupies `[0, n)`; component 1
 * (psi2-fluctuation-like) occupies `[n, 2n)`. `ConstCoupledVectorView` is the
 * read-only counterpart; a `CoupledVectorView` converts implicitly to a
 * `ConstCoupledVectorView`, mirroring `DeviceSpan`'s own safe non-const ->
 * const conversion.
 *
 * == D1: weighted coupled norm ==
 *
 *   ||(a, b)||_w = sqrt( (RMS(a) / g1)^2 + (RMS(b) / g2)^2 ),
 *   g1 = source_config.v_rms,  g2 = 1
 *
 * the SAME per-component scales `ResidualEvaluator.cuh` uses to normalize
 * `r1`/`r2`. Computed with the existing deterministic `blas` reductions
 * (`nrm2_device`, a fixed-shape cuBLAS tree reduction): both component norms
 * are enqueued into distinct slots of one small device scratch buffer with no
 * intermediate host synchronization, then transferred to the host in exactly
 * ONE `cudaMemcpyAsync` + `synchronize()` per weighted-norm evaluation (the
 * SAME "many device reductions into one buffer, one sync" idiom
 * `AndersonAccelerator.cu` uses for its Gram/rhs assembly). This IS an
 * explicit, documented synchronization; it is not hidden, and `apply()` below
 * issues exactly one of these (for the projected direction's norm) beyond the
 * residual evaluator's own documented `synchronize()`.
 *
 * == D2: forward-difference delta policy ==
 *
 *   delta = sqrt(machine_eps) * (1 + ||Psi||_w) / ||p||_w
 *
 * `||p||_w == 0` or non-finite is a host-misuse precondition and throws
 * `std::invalid_argument` (there is no meaningful forward-difference
 * direction to take). The result is clamped to the caller-configured
 * `[delta_min, delta_max]` interval (`JvpDeltaConfig`); clamp events are
 * counted, both per-call (`JvpApplyReport`) and cumulatively
 * (`JvpWorkspace::delta_min_clamp_count()` / `delta_max_clamp_count()`). If
 * the perturbed residual report comes back non-finite (checked against its
 * RMS/Linf fields), `apply()` returns a STRUCTURED failure
 * (`JvpApplyStatus::nonfinite_perturbed_residual`), not an exception: Krylov
 * callers need a recoverable per-iteration signal, not a thrown exception
 * that would unwind an in-progress linear solve.
 *
 * == D3: projection discipline ==
 *
 * Both direction components are projected (`constraints::MeanZeroProjector`)
 * into a private workspace copy BEFORE use; the caller's `direction` view is
 * never mutated. After forming the perturbed state `Psi + delta*p` (an axpy
 * into workspace-owned buffers), BOTH perturbed components are projected
 * again (idempotent defense against gauge drift, the same rationale SF-15
 * applies to its backtracking trial state). The BASE state `Psi` is a CALLER
 * contract: `prepare_jvp_base` assumes it is already mean-zero projected and
 * does not re-project it. The Jacobian-vector OUTPUT is intentionally NOT
 * projected: `F` is discretely mean-zero by construction (see
 * `ResidualEvaluator.cuh`), so the output's own mean is a defect observable
 * that a caller/test can measure directly against zero, rather than one this
 * module silently launders.
 *
 * == D4: cached base residual ==
 *
 * `JvpWorkspace::prepare_jvp_base` evaluates `F(Psi)` via
 * `enqueue_streamfunction_residual` EXACTLY ONCE and caches it on device
 * (`base_f1_`, `base_f2_`); it also caches the weighted base-norm term
 * `1 + ||Psi||_w` (one weighted-norm evaluation, one sync) and the frozen
 * `(eta, source_config, histogram_config, gauge, q span, fluctuations spans)`
 * identity `apply()` reuses on every call, so `apply()` itself only takes the
 * direction, the delta-clamp config, and the output view. `apply()` performs
 * EXACTLY ONE perturbed residual evaluation per call (one
 * `enqueue_streamfunction_residual` + one
 * `synchronize_streamfunction_residual_report`). `base_evaluations()` /
 * `apply_evaluations()` are monotonically increasing counters (never reset by
 * `prepare()`) that let a later test assert exactly one base evaluation per
 * `prepare_jvp_base` call and exactly one residual evaluation per `apply()`
 * call.
 *
 * `source_config.v_rms` is measured from the Darcy field and is independent
 * of `Psi`; freezing it across every perturbed evaluation this module issues
 * is therefore EXACT (the same reference speed the accepted residual/source
 * evaluators already use), not an approximation introduced here.
 *
 * Grid restriction: this module inherits the accepted SF-02/SF-06 isotropic-
 * grid restriction from `ResidualEvaluator.cuh` (`enqueue_streamfunction_residual`
 * and `synchronize_streamfunction_residual_report` fail fast with
 * `std::invalid_argument` for a non-isotropic or too-small grid); it is not
 * re-validated here.
 *
 * Allocation/synchronization policy: `prepare()` performs all allocation;
 * after `prepare()`, `prepare_jvp_base()` and `apply()` perform no
 * allocation. Every device operation is stream-ordered on `ctx.cuda_stream()`.
 * `prepare_jvp_base()` issues one documented synchronization (the weighted
 * base-norm evaluation); `apply()` issues two documented synchronizations
 * (the weighted direction-norm evaluation, then the residual evaluator's own
 * `synchronize_streamfunction_residual_report()` call) -- no additional
 * hidden synchronization is added anywhere in this file.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../numerics/blas/reduction_workspace.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../runtime/CudaContext.cuh"
#include "NonlinearSources.cuh"
#include "ResidualEvaluator.cuh"
#include "affine_gauge.cuh"

#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Light, non-owning view pairing two coupled-state components (component 0 =
 * psi1-fluctuation-like, component 1 = psi2-fluctuation-like). Neither span
 * is owned; the caller keeps the referenced buffers alive for the lifetime of
 * any call that consumes this view. See the file header for the exact
 * contiguous-buffer layout convention `split_coupled_vector` implements.
 */
struct CoupledVectorView {
    DeviceSpan<real> c1;
    DeviceSpan<real> c2;
};

/** Read-only counterpart of `CoupledVectorView`. */
struct ConstCoupledVectorView {
    DeviceSpan<const real> c1;
    DeviceSpan<const real> c2;

    ConstCoupledVectorView() = default;
    ConstCoupledVectorView(DeviceSpan<const real> c1_, DeviceSpan<const real> c2_)
        : c1(c1_), c2(c2_) {}
    // Safe non-const -> const conversion, mirroring DeviceSpan's own.
    ConstCoupledVectorView(const CoupledVectorView& view) : c1(view.c1), c2(view.c2) {}
};

/**
 * Splits a contiguous coupled buffer of exactly `2 * n` elements into its two
 * `n`-element component views, per the layout contract documented in the file
 * header (component 0 = `[0, n)`, component 1 = `[n, 2n)`). Throws
 * `std::invalid_argument` if `combined.size() != 2 * n` or `n == 0`. This is
 * the seam a Krylov-style caller uses to present one owned `2n`-length
 * storage buffer as a `CoupledVectorView`/`ConstCoupledVectorView` without any
 * copy.
 */
[[nodiscard]] CoupledVectorView split_coupled_vector(DeviceSpan<real> combined, std::size_t n);
[[nodiscard]] ConstCoupledVectorView split_coupled_vector(DeviceSpan<const real> combined,
                                                          std::size_t n);

/**
 * Forward-difference delta clamp configuration (D2). `delta_min`/`delta_max`
 * must both be finite with `0 < delta_min < delta_max`; validated by
 * `validate_jvp_delta_config` (also called internally by
 * `JvpWorkspace::apply`).
 */
struct JvpDeltaConfig {
    real delta_min{real{1e-12}};
    real delta_max{real{1e2}};
};

/** Throws `std::invalid_argument` if `config` violates the D2 contract. */
void validate_jvp_delta_config(const JvpDeltaConfig& config);

/**
 * Structured per-call outcome of `JvpWorkspace::apply`. `ok` means `jv_out`
 * holds a valid forward-difference Jacobian-vector product; `nonfinite_
 * perturbed_residual` means the perturbed residual evaluation itself came
 * back non-finite (checked against its RMS/Linf fields) -- a recoverable,
 * structured signal for a Krylov caller, not an exception; `jv_out` is left
 * untouched in that case.
 */
enum class JvpApplyStatus { ok, nonfinite_perturbed_residual };

/**
 * Host report for one `JvpWorkspace::apply` call. `weighted_p_norm` is
 * `||p||_w` of the PROJECTED direction (D1); `delta` is the final (possibly
 * clamped) forward-difference step (D2); `delta_min_clamped`/
 * `delta_max_clamped` flag whether THIS call's delta was clamped.
 * `perturbed_rms_f1`/`perturbed_rms_f2`/`perturbed_linf_f1`/
 * `perturbed_linf_f2` echo the perturbed-state residual report's own
 * reductions, regardless of `status` (always populated once the perturbed
 * residual evaluation has run).
 */
struct JvpApplyReport {
    JvpApplyStatus status{JvpApplyStatus::ok};
    real delta{};
    real weighted_p_norm{};
    bool delta_min_clamped{false};
    bool delta_max_clamped{false};

    real perturbed_rms_f1{};
    real perturbed_rms_f2{};
    real perturbed_linf_f1{};
    real perturbed_linf_f2{};
};

/**
 * Owns every scratch buffer needed to evaluate the matrix-free coupled
 * Jacobian-vector product for one grid cell count `n`, reusing exactly one
 * owned `StreamfunctionResidualWorkspace` (D4). Move-only, matching every
 * other accepted streamfunction workspace convention.
 *
 * Usage: `prepare(n)` once per grid size; `prepare_jvp_base(...)` once per
 * base state `Psi` (caches `F(Psi)` and the frozen call identity); then any
 * number of `apply(...)` calls against that SAME cached base, each producing
 * one `J(Psi) p` for a caller-supplied direction `p`. Calling
 * `prepare_jvp_base` again (same or different `Psi`) refreshes the cached
 * base and identity for subsequent `apply` calls.
 */
class JvpWorkspace {
  public:
    JvpWorkspace() = default;
    ~JvpWorkspace() = default;

    JvpWorkspace(const JvpWorkspace&) = delete;
    JvpWorkspace& operator=(const JvpWorkspace&) = delete;
    JvpWorkspace(JvpWorkspace&&) noexcept = default;
    JvpWorkspace& operator=(JvpWorkspace&&) noexcept = default;

    // Allocates/resizes every owned buffer for exactly n cells. Idempotent
    // for an already-prepared n (no device allocation). Invalidates any
    // previously cached base (a fresh prepare_jvp_base call is required
    // before the next apply()); does NOT reset the monotonically increasing
    // evaluation/clamp counters (see the file header, D4).
    void prepare(std::size_t n);

    [[nodiscard]] bool prepared_for(std::size_t n) const noexcept;

    // Evaluates and caches F(Psi) exactly once (D4), plus the weighted
    // base-norm term 1 + ||Psi||_w, and freezes the (eta, source_config,
    // histogram_config, gauge, q, fluctuations) identity every subsequent
    // apply() call reuses. `base` is a CALLER contract: already mean-zero
    // projected (D3); this call does not project it. `base.c1`/`base.c2`
    // must remain valid and unchanged (device contents) for the lifetime of
    // every apply() call made against this cached base. Throws
    // std::logic_error if prepare(n) was not called for
    // grid.num_cells() == n; delegates every other precondition (q/gauge/eta
    // /source_config/histogram_config validity, grid isotropy) to
    // enqueue_streamfunction_residual.
    void prepare_jvp_base(CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q,
                          CoupledVectorView base, const AffineGauge& gauge, real eta,
                          const NonlinearSourceConfig& source_config,
                          const ResidualHistogramConfig& histogram_config);

    // One matrix-free Jacobian-vector product J(Psi) p ~= (F(Psi + delta*p) -
    // F(Psi)) / delta against the base state cached by the most recent
    // prepare_jvp_base call. `direction` is read-only (D3: never mutated);
    // `jv_out` is the caller-owned output, of exactly n elements per
    // component, NOT projected on output (see the file header). Throws
    // std::logic_error if prepare_jvp_base has not been called since the
    // last prepare(); throws std::invalid_argument if delta_config is
    // invalid, if direction/jv_out are not exactly n elements per component,
    // or if the projected direction's weighted norm is zero or non-finite
    // (D2). Performs exactly one perturbed residual evaluation (D4).
    [[nodiscard]] JvpApplyReport apply(CudaContext& ctx, const Grid3D& grid,
                                       ConstCoupledVectorView direction,
                                       const JvpDeltaConfig& delta_config,
                                       CoupledVectorView jv_out);

    // Monotonically increasing, never reset by prepare() (see the file
    // header, D4): base_evaluations() counts prepare_jvp_base() calls;
    // apply_evaluations() counts apply() calls (one residual evaluation
    // each, regardless of the returned status).
    [[nodiscard]] long long base_evaluations() const noexcept { return base_evaluations_; }
    [[nodiscard]] long long apply_evaluations() const noexcept { return apply_evaluations_; }

    // Cumulative D2 clamp-event counters across every apply() call.
    [[nodiscard]] unsigned long long delta_min_clamp_count() const noexcept {
        return delta_min_clamp_total_;
    }
    [[nodiscard]] unsigned long long delta_max_clamp_count() const noexcept {
        return delta_max_clamp_total_;
    }

    // Exact sum of every owned DeviceBuffer capacity across this workspace
    // and its owned sub-workspaces, in bytes. Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Host-only prediction of allocated_device_bytes() after a fresh
    // prepare(n) call; kept colocated with prepare() so it cannot drift.
    [[nodiscard]] static std::size_t estimate_device_bytes(std::size_t n);

  private:
    // One weighted-norm evaluation (D1): two nrm2_device calls into distinct
    // slots of norm_scratch_, one memcpy + one synchronize. Non-const: uses
    // owned scratch/reduction-workspace state.
    [[nodiscard]] real compute_weighted_norm(CudaContext& ctx, DeviceSpan<const real> a,
                                             DeviceSpan<const real> b, real g1, real g2);

    void ensure_prepared() const;
    void ensure_base_prepared() const;

    std::size_t n_ = 0;
    bool base_prepared_ = false;

    // D4: cached F(Psi), one field per component.
    DeviceBuffer<real> base_f1_;
    DeviceBuffer<real> base_f2_;

    // D3: projected-direction workspace copy.
    DeviceBuffer<real> proj_p1_;
    DeviceBuffer<real> proj_p2_;

    // Perturbed state Psi + delta*p, and its residual F(Psi + delta*p).
    DeviceBuffer<real> pert_u1_;
    DeviceBuffer<real> pert_u2_;
    DeviceBuffer<real> pert_f1_;
    DeviceBuffer<real> pert_f2_;

    // D1: fixed 2-slot device scratch for the coupled weighted-norm
    // reduction (component-1 norm, component-2 norm), n-independent.
    DeviceBuffer<real> norm_scratch_;

    StreamfunctionResidualWorkspace residual_workspace_;
    constraints::MeanZeroWorkspace projection_workspace_;
    blas::ReductionWorkspace reduction_workspace_;

    // D4: frozen call identity captured by prepare_jvp_base, reused by apply.
    DeviceSpan<const real> q_{};
    PeriodicStreamfunctionFluctuations base_fluctuations_{};
    AffineGauge gauge_{};
    real eta_{};
    NonlinearSourceConfig source_config_{};
    ResidualHistogramConfig histogram_config_{};
    real weighted_base_term_{}; // 1 + ||Psi||_w

    long long base_evaluations_ = 0;
    long long apply_evaluations_ = 0;
    unsigned long long delta_min_clamp_total_ = 0;
    unsigned long long delta_max_clamp_total_ = 0;
};

} // namespace streamfunctions
} // namespace macroflow3d
