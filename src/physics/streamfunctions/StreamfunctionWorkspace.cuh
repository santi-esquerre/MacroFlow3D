#pragma once

/**
 * @file StreamfunctionWorkspace.cuh
 * @brief Owned streamfunction fields and the unified, reusable solver
 *        workspace (SF-12).
 *
 * This header defines the two owning types of the SF-12 public surface:
 *
 *   - `StreamfunctionFields`: owns `u1`, `u2` (the periodic fluctuations of
 *     `psi1`, `psi2`, see `affine_gauge.cuh`).
 *   - `StreamfunctionWorkspace`: owns every scratch and solver buffer needed
 *     by the accepted primitives from SF-02..SF-11 to assemble and solve the
 *     Lester equation (14) linear subproblem and evaluate its coupled
 *     nonlinear residual and physical diagnostics, sized for one grid and
 *     one `StreamfunctionSolverConfig`.
 *
 * Ownership and allocation contract: `prepare()` performs all allocation.
 * Calling `prepare()` again with an unchanged grid and config performs no
 * device allocation (every owned `DeviceBuffer::resize()` call is a no-op
 * once capacity already covers the requested size; the multigrid hierarchy
 * and preconditioner, which cannot be resized in place, are only
 * reconstructed when the grid or `config.mg` actually changed). After
 * `prepare()`, no use path covered by this workspace allocates.
 *
 * Sequential-only solve: this workspace intentionally owns exactly ONE
 * `multigrid::MGHierarchy` and ONE `solvers::ProjectedPCGWorkspace`. The
 * increment spec mandates sequential block solves by default and exposes no
 * concurrent solve mode; both the `psi1` and `psi2` linear blocks share the
 * same operator `A(q)` and are solved one after another, reusing this single
 * hierarchy/workspace pair. A future increment that wants concurrent block
 * solves must introduce a second hierarchy/workspace pair explicitly; SF-12
 * does not.
 *
 * Move semantics: `StreamfunctionFields` and `StreamfunctionWorkspace` are
 * move-only, matching every accepted sub-workspace's convention. The MG
 * hierarchy is held behind `std::unique_ptr<multigrid::MGHierarchy>` (not by
 * value) because `solvers::ProjectedPositiveMGPreconditioner` stores a raw
 * `MGHierarchy*`; a `unique_ptr` gives the hierarchy a stable heap address
 * across `StreamfunctionWorkspace` moves, so a moved-from/moved-to workspace
 * never invalidates the preconditioner's pointer.
 *
 * Byte accounting: `capacity()`, not `size()`, is the owned-bytes contract
 * throughout this file. The two owned-storage regimes shrink differently on
 * re-preparation: every `DeviceBuffer`-backed member (the scratch fields and
 * every sub-workspace's own buffers) never shrinks capacity nor moves its
 * pointer when re-preparing for a smaller grid, so those buffers individually
 * keep the larger capacity. The MG hierarchy and preconditioner, held behind
 * `unique_ptr`/`optional`, cannot be resized in place; they are destroyed and
 * reconstructed whenever the grid or `config.mg` changes, so their
 * contribution to `allocated_device_bytes()` shrinks (and the AGGREGATE total
 * can shrink) across a smaller-grid re-prepare even though the DeviceBuffer
 * members do not. The memory estimator therefore predicts a FRESH
 * preparation, not a reused/shrunk one; re-preparing back at the identical
 * original grid/config restores the aggregate total exactly. cuBLAS/CUB
 * internal handle memory is excluded from owned-bytes accounting
 * project-wide, consistent with every accepted sub-workspace.
 *
 * `q` is allocated here (one field) but SF-12 does not fill it; computing
 * `q = 1/K` (or `q = exp(-Y)`) from `StreamfunctionProblemView::conductivity`
 * is later increments' job (SF-13).
 *
 * `u_trial1`/`u_trial2` (SF-15) are the adaptive-Picard backtracking trial
 * pair: `solve_streamfunctions` writes each candidate relaxed state
 * `(1-omega_try)*u_i + omega_try*u_hat_i` here, evaluates the trial's
 * residual/diagnostics from these buffers, and only copies them into the
 * caller-owned `StreamfunctionFields::u1_span()`/`u2_span()` once a trial is
 * accepted. The accepted state in `StreamfunctionFields` is therefore NEVER
 * overwritten during backtracking; only these two scratch fields are
 * mutated while a trial is being evaluated. SF-24's Newton line search
 * (`NewtonKrylovSolver.cu`) reuses this SAME `u_trial1`/`u_trial2` pair for
 * its own trials, under the identical "only an accepted trial touches
 * `StreamfunctionFields`" discipline.
 *
 * == SF-24: optional Newton sub-workspace ==
 *
 * `StreamfunctionWorkspace` additionally owns an OPTIONAL Newton-Krylov sub-
 * workspace, allocated ONLY when `config.newton.enabled` at the last
 * `prepare()` call -- the exact `std::optional` lifecycle SF-20's
 * `anderson_` already established (disabling deallocates it; an
 * already-prepared `(n, restart)` pair performs no device allocation;
 * accessors throw `std::logic_error` when disabled). It bundles:
 *
 *   - a `JvpWorkspace` (SF-22) prepared for `n`;
 *   - a `CoupledGmres` (SF-23) prepared for `(n, config.newton.gmres.restart)`;
 *   - a `BlockDiagonalMGPreconditioner` (SF-23) constructed over
 *     `hierarchy()` + `config.mg`. UNLIKE the JVP/GMRES members, this cannot
 *     be resized in place (it holds an `MGHierarchy*`), so it follows the
 *     SAME lifecycle rule as `mg_preconditioner_` below: it is (re)constructed
 *     whenever the MG hierarchy itself is (re)constructed (grid or
 *     `config.mg` change), AND whenever the Newton sub-workspace is newly
 *     created (`config.newton.enabled` transitioning from `false` to `true`
 *     against an already-current hierarchy);
 *   - two dedicated `n`-sized delta buffers (`newton_delta1()`/
 *     `newton_delta2()`), the GMRES solve's output `delta = M^-1 u` for the
 *     Newton line search to read from.
 *
 * No allocation occurs in the Newton solve path after `prepare()` (see
 * `NewtonKrylovSolver.cuh`).
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/solvers/pcg.cuh"
#include "../../numerics/solvers/projected_positive_mg_preconditioner.cuh"
#include "AndersonAccelerator.cuh"
#include "BlockDiagonalMGPreconditioner.cuh"
#include "CoupledGmres.cuh"
#include "Diagnostics.cuh"
#include "JacobianVectorProduct.cuh"
#include "ResidualEvaluator.cuh"
#include "StreamfunctionTypes.hpp"
#include "affine_gauge.cuh"
#include "affine_periodic_rhs.cuh"

#include <cstddef>
#include <memory>
#include <optional>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Owns the periodic fluctuations `u1` (of `psi1`) and `u2` (of `psi2`).
 * Move-only; copy is deleted to keep device ownership unambiguous, matching
 * every other accepted streamfunction workspace.
 */
class StreamfunctionFields {
  public:
    StreamfunctionFields() = default;
    ~StreamfunctionFields() = default;

    StreamfunctionFields(const StreamfunctionFields&) = delete;
    StreamfunctionFields& operator=(const StreamfunctionFields&) = delete;
    StreamfunctionFields(StreamfunctionFields&&) noexcept = default;
    StreamfunctionFields& operator=(StreamfunctionFields&&) noexcept = default;

    // Allocates/resizes u1, u2 to exactly grid.num_cells() elements each.
    // A repeated call for an already-prepared grid performs no allocation.
    void prepare(const Grid3D& grid);

    [[nodiscard]] bool prepared_for(const Grid3D& grid) const noexcept;

    // Fluctuation view for kernels that consume/mutate u1/u2 (see
    // affine_gauge.cuh: PeriodicStreamfunctionFluctuations only ever exposes
    // mutable DeviceSpan<real> members, matching every accepted consumer of
    // this type, e.g. DifferentialOperators.cuh). Throws std::logic_error if
    // called before prepare(). Use u1_span()/u2_span() const below for
    // genuinely read-only, const-correct access.
    [[nodiscard]] PeriodicStreamfunctionFluctuations fluctuations();

    [[nodiscard]] DeviceSpan<real> u1_span() noexcept { return u1_.span(); }
    [[nodiscard]] DeviceSpan<real> u2_span() noexcept { return u2_.span(); }
    [[nodiscard]] DeviceSpan<const real> u1_span() const noexcept { return u1_.span(); }
    [[nodiscard]] DeviceSpan<const real> u2_span() const noexcept { return u2_.span(); }

    // Exact sum of owned DeviceBuffer capacities (u1 + u2), in bytes.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

  private:
    DeviceBuffer<real> u1_;
    DeviceBuffer<real> u2_;
    int nx_ = 0, ny_ = 0, nz_ = 0;
};

/**
 * Owns every scratch and solver buffer required to assemble and solve the
 * Lester equation (14) linear subproblem and evaluate its coupled nonlinear
 * residual and SF-11 physical diagnostics for one grid and one
 * `StreamfunctionSolverConfig`. See the file header for the full ownership,
 * allocation, sequential-solve, and move-semantics contract.
 *
 * Members exposed by `q()`, `rhs1()`, `rhs2()`, `f1()`, `f2()`, `v_psi()`,
 * `residual_workspace()`, `diagnostics_workspace()`,
 * `affine_rhs_workspace()`, `pcg_workspace()`, and `hierarchy()` are the
 * caller-facing handles a future solver implementation (SF-13) uses to drive
 * the accepted primitives without any additional allocation. Every accessor
 * throws `std::logic_error` if called before a successful `prepare()`.
 */
class StreamfunctionWorkspace {
  public:
    StreamfunctionWorkspace() = default;
    ~StreamfunctionWorkspace() = default;

    StreamfunctionWorkspace(const StreamfunctionWorkspace&) = delete;
    StreamfunctionWorkspace& operator=(const StreamfunctionWorkspace&) = delete;
    StreamfunctionWorkspace(StreamfunctionWorkspace&&) noexcept = default;
    StreamfunctionWorkspace& operator=(StreamfunctionWorkspace&&) noexcept = default;

    // Allocates/resizes every owned buffer and, only if the grid or
    // config.mg actually changed since the last successful prepare(),
    // (re)constructs the MG hierarchy and preconditioner. Throws
    // std::invalid_argument if the grid cannot support config.mg.num_levels
    // under the exact multigrid::MGHierarchy coarsening/break rule and
    // multigrid::validate_projected_positive_hierarchy's requirements (the
    // same rule enforced by validate_streamfunction_problem).
    void prepare(const Grid3D& grid, const StreamfunctionSolverConfig& config);

    [[nodiscard]] bool prepared_for(const Grid3D& grid,
                                    const StreamfunctionSolverConfig& config) const noexcept;

    // SF-21: whether this workspace currently holds a valid `q`/MG
    // hierarchy/affine-RHS built by a `CoefficientState::rebuild` call for
    // its currently prepared grid. `prepare()` clears this whenever the grid
    // changes (see StreamfunctionWorkspace.cu); `solve_streamfunctions` sets
    // it after a successful rebuild and requires it before honoring
    // `CoefficientState::reuse`. See `CoefficientState` in
    // StreamfunctionTypes.hpp for the full caller contract.
    [[nodiscard]] bool coefficients_valid() const noexcept { return coefficients_valid_; }
    void mark_coefficients_valid() noexcept { coefficients_valid_ = true; }

    // q = 1/K (or exp(-Y)) storage; SF-12 allocates it but does not fill it.
    [[nodiscard]] DeviceSpan<real> q();
    [[nodiscard]] DeviceSpan<const real> q() const;

    [[nodiscard]] DeviceSpan<real> rhs1();
    [[nodiscard]] DeviceSpan<real> rhs2();
    [[nodiscard]] DeviceSpan<real> f1();
    [[nodiscard]] DeviceSpan<real> f2();

    // SF-15 adaptive-Picard backtracking trial pair; see the file header.
    [[nodiscard]] DeviceSpan<real> u_trial1();
    [[nodiscard]] DeviceSpan<real> u_trial2();
    [[nodiscard]] DeviceSpan<const real> u_trial1() const;
    [[nodiscard]] DeviceSpan<const real> u_trial2() const;

    // Caller-owned CompactMAC output view for the reconstructed velocity
    // v_psi (see Diagnostics.cuh). Every plane, including the periodic
    // duplicate planes, is caller-writable.
    [[nodiscard]] CompactMacVelocityView v_psi();
    [[nodiscard]] CompactMacVelocityConstView v_psi() const;

    [[nodiscard]] StreamfunctionResidualWorkspace& residual_workspace();
    [[nodiscard]] StreamfunctionDiagnosticsWorkspace& diagnostics_workspace();

    // Top-level affine-periodic-RHS workspace, distinct from the private
    // instance owned internally by StreamfunctionResidualWorkspace. This one
    // is for assembling rhs1/rhs2 directly, outside of a full residual
    // evaluation (e.g. the SF-13 Picard block-RHS assembly step).
    [[nodiscard]] AffinePeriodicRhsWorkspace& affine_rhs_workspace();

    // The single, shared projected-PCG workspace used sequentially for both
    // the psi1 and psi2 linear block solves (see the file header).
    [[nodiscard]] solvers::ProjectedPCGWorkspace& pcg_workspace();

    // Non-null only after a successful prepare(). Stable across
    // StreamfunctionWorkspace moves (see the file header).
    [[nodiscard]] multigrid::MGHierarchy& hierarchy();
    [[nodiscard]] const multigrid::MGHierarchy& hierarchy() const;

    [[nodiscard]] solvers::ProjectedPositiveMGPreconditioner& preconditioner();
    [[nodiscard]] const solvers::ProjectedPositiveMGPreconditioner& preconditioner() const;

    // SF-20: non-null only when this workspace was last prepare()'d with
    // config.anderson.enabled == true (allocated ONLY in that case; see the
    // file header and AndersonAccelerator.cuh). Throws std::logic_error if
    // called while disabled (mirroring every other ensure_prepared()
    // accessor's failure mode).
    [[nodiscard]] AndersonAccelerator& anderson();
    [[nodiscard]] bool anderson_enabled() const noexcept;

    // SF-24: non-null only when this workspace was last prepare()'d with
    // config.newton.enabled == true (allocated ONLY in that case; see the
    // file header). Each throws std::logic_error if called while disabled
    // (mirroring anderson()'s failure mode).
    [[nodiscard]] JvpWorkspace& newton_jvp();
    [[nodiscard]] CoupledGmres& newton_gmres();
    [[nodiscard]] BlockDiagonalMGPreconditioner& newton_preconditioner();
    [[nodiscard]] DeviceSpan<real> newton_delta1();
    [[nodiscard]] DeviceSpan<real> newton_delta2();
    [[nodiscard]] bool newton_enabled() const noexcept;

    // Exact sum of every owned DeviceBuffer capacity across this workspace
    // and every sub-workspace/hierarchy/preconditioner it owns, in bytes.
    // Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Categorized report built from this workspace's ACTUAL capacities.
    // fields_bytes is always zero here (StreamfunctionFields is a separate,
    // independently owned object); use make_streamfunction_memory_report()
    // to combine both. grid supplies n for fine_grid_equivalent_fields.
    [[nodiscard]] StreamfunctionMemoryReport memory_report(const Grid3D& grid) const;

  private:
    void ensure_prepared() const;

    DeviceBuffer<real> q_;
    DeviceBuffer<real> rhs1_;
    DeviceBuffer<real> rhs2_;
    DeviceBuffer<real> f1_;
    DeviceBuffer<real> f2_;

    DeviceBuffer<real> u_trial1_;
    DeviceBuffer<real> u_trial2_;

    DeviceBuffer<real> v_psi_u_;
    DeviceBuffer<real> v_psi_v_;
    DeviceBuffer<real> v_psi_w_;

    StreamfunctionResidualWorkspace residual_workspace_;
    StreamfunctionDiagnosticsWorkspace diagnostics_workspace_;
    AffinePeriodicRhsWorkspace affine_rhs_workspace_;
    solvers::ProjectedPCGWorkspace pcg_workspace_;

    // See the file header: exactly one hierarchy, held behind unique_ptr for
    // pointer stability under StreamfunctionWorkspace moves.
    std::unique_ptr<multigrid::MGHierarchy> mg_hierarchy_;
    std::optional<solvers::ProjectedPositiveMGPreconditioner> mg_preconditioner_;

    // SF-20: optional, allocated only when config.anderson.enabled (see the
    // file header and AndersonAccelerator.cuh).
    std::optional<AndersonAccelerator> anderson_;

    // SF-24: bundles the optional Newton-Krylov sub-workspace (see the file
    // header). BlockDiagonalMGPreconditioner is not default-constructible
    // (it binds to *mg_hierarchy_), hence its own nested optional.
    struct NewtonSubWorkspace {
        JvpWorkspace jvp;
        CoupledGmres gmres;
        std::optional<BlockDiagonalMGPreconditioner> preconditioner;
        DeviceBuffer<real> delta1;
        DeviceBuffer<real> delta2;
    };
    std::optional<NewtonSubWorkspace> newton_;

    Grid3D prepared_grid_{};
    multigrid::MGConfig prepared_mg_config_{};
    bool prepared_ = false;

    // SF-21: see coefficients_valid() above.
    bool coefficients_valid_ = false;
};

/**
 * Pure-host, allocation-free prediction of the exact owned-bytes totals a
 * fresh `StreamfunctionFields::prepare(grid)` +
 * `StreamfunctionWorkspace::prepare(grid, config)` pair will produce. Issues
 * the same host-side CUB size queries (`blas::sum_workspace_bytes`) that the
 * corresponding prepare() paths issue, so it never allocates device memory
 * itself. Throws std::invalid_argument under the same conditions as
 * `validate_streamfunction_problem`'s grid/config-only checks (this function
 * does not take a StreamfunctionProblemView, so it cannot check
 * problem-specific preconditions such as conductivity span size).
 */
[[nodiscard]] StreamfunctionMemoryReport estimate_streamfunction_memory(
    const Grid3D& grid, const StreamfunctionSolverConfig& config);

/**
 * Combines an already-prepared StreamfunctionFields and StreamfunctionWorkspace
 * into one categorized report from their ACTUAL capacities. grid must be the
 * grid both were prepared for; this is not re-validated.
 */
[[nodiscard]] StreamfunctionMemoryReport make_streamfunction_memory_report(
    const StreamfunctionFields& fields, const StreamfunctionWorkspace& workspace,
    const Grid3D& grid);

} // namespace streamfunctions
} // namespace macroflow3d
