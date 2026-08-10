#pragma once

/**
 * @file AffinePeriodicFlowSolver.cuh
 * @brief Affine-periodic Darcy solve for the triply periodic Lester benchmark
 *        (SF-19).
 * @ingroup physics_flow
 *
 * This solver constructs a mass-conservative REFERENCE Darcy velocity field
 * on a triply periodic conductivity realization `K(x)`, with a prescribed
 * mean flux, instead of applying incompatible Dirichlet head boundaries. It
 * is a standalone entry point: it does not touch the existing Dirichlet head
 * solver (`solve_head.cuh`), the existing velocity-from-head path
 * (`velocity_from_head.cuh`), or the streamfunction nonlinear solver.
 *
 * ## Mathematical construction (homogenization cell problems)
 *
 * Head is decomposed as `h(x) = -G.x + h_tilde(x)` with `h_tilde` triply
 * periodic. For each unit mean pressure-gradient direction `e_d`
 * (`d = x, y, z`) this module solves the zero-mean periodic corrector cell
 * problem
 *
 *   div(K (grad w_d + e_d)) = 0,   <w_d> = 0
 *
 * i.e. `A_K w_d = div(K e_d)` with `A_K u = -div_h(K grad_h u)`. This is
 * EXACTLY the accepted `operators::LesterPositiveDiffusionOperator` /
 * `solvers::projected_pcg_solve` / multigrid stack used by the Lester
 * streamfunction path (`StreamfunctionSolver.cu`), with the coefficient
 * array filled with `K` ITSELF instead of `q = 1/K`: this coefficient swap
 * is the increment's "explicit sign/coefficient adapter". No numerics or
 * streamfunctions file is modified to achieve this; the operator, RHS
 * assembler, PCG solver, and MG preconditioner are reused unchanged.
 *
 * The right-hand side `div_h(K e_d)` uses the SAME harmonic face-coefficient
 * convention (`2 K_C K_N / (K_C + K_N)`) as the operator, by reusing the SF-06
 * assembler `streamfunctions::assemble_affine_periodic_rhs` per direction
 * with a degenerate gauge pair `(e_d, e_d)` (both outputs are then
 * numerically identical; only the first is kept, the second is scratch).
 * Reusing this exact assembler (rather than hand-writing an equivalent
 * kernel) removes any risk of a convention mismatch between the operator and
 * the RHS -- see the file header of `affine_periodic_rhs.cu` for the shared
 * face formula. Sharing the identical harmonic-face convention across the
 * operator, the RHS, and the velocity reconstruction below is MANDATORY: it
 * is what makes discrete mass conservation (`div F`) reduce exactly to the
 * corrector PCG residuals, and what makes `K_eff` symmetric up to those same
 * residuals (a discrete summation-by-parts identity).
 *
 * ## Effective conductivity tensor
 *
 * Column `d` of `K_eff` is the domain average, over the `nx*ny*nz` UNIQUE
 * periodic faces of direction `i`, of the face flux component
 * `K_f (grad w_d + e_d)_i`. This is computed by evaluating the same
 * affine-aware CompactMAC face-flux formula used for the final velocity
 * (below) with `G = e_d`, `w = w_d`, and averaging over the unique planes of
 * each component with a deterministic two-pass block reduction (see
 * `AffinePeriodicFlowSolver.cu`; the same pattern as
 * `PeriodicGaussianField.cu`'s mean/variance reduction: per-block partial
 * sums are copied to the host and summed there in a fixed, sequential
 * order). `symmetry_defect_rel = max_ij|K_eff - K_eff^T| / max_ij|K_eff|` is
 * reported, never silently symmetrized away, together with the eigenvalues
 * of the symmetric part `(K_eff + K_eff^T)/2` as positivity evidence.
 *
 * ## Mean-flux solve
 *
 * The full (unsymmetrized) 3x3 host system `K_eff G = qbar` is solved in
 * double precision by Cramer's rule; the transverse coupling of a finite
 * realization's tensor is NEVER discarded (rollback rule in the increment
 * spec). Before solving, the symmetric part's eigenvalues are checked for
 * strict positivity; a non-positive-definite symmetric part or a singular
 * `K_eff` throws `std::runtime_error` with a distinct message rather than
 * silently returning a degenerate `G`.
 *
 * ## Velocity reconstruction
 *
 * The final periodic head correction `h_tilde = sum_d G_d w_d` is
 * recombined (three `blas::axpy` calls), and the affine-aware CompactMAC
 * face flux
 *
 *   F_f = K_f ( G.n + d/dn(h_tilde) )
 *
 * is evaluated with the SAME harmonic `K_f` and the SAME face convention as
 * `streamfunctions::Diagnostics.cuh`'s `CompactMacVelocityConstView` /
 * `physics::VelocityField`: cell-centered `idx = i + nx*(j + ny*k)`
 * (x-fastest, triply periodic); `U` dims `(nx+1, ny, nz)`,
 * `idx = i + j*(nx+1) + k*(nx+1)*ny`; `V` dims `(nx, ny+1, nz)`,
 * `idx = i + j*nx + k*nx*(ny+1)`; `W` dims `(nx, ny, nz+1)`,
 * `idx = i + j*nx + k*nx*ny`; the periodic duplicate plane (`U` plane `nx`,
 * `V` plane `ny`, `W` plane `nz`) is always numerically identical to plane 0
 * by construction (same wrapped cell pair), never a separate write. The
 * per-cell discrete divergence `div F = (U[i+1]-U[i])/dx + (V[j+1]-V[j])/dy +
 * (W[k+1]-W[k])/dz` and the achieved mean flux (mean over unique faces, same
 * reduction as `K_eff`) are reported as the mass-conservation and flux
 * diagnostics.
 *
 * ## Determinism
 *
 * Every reduction in this module (unique-face means, divergence max/RMS) is
 * a fixed-shape two-pass block reduction: a `<<<n_blocks, 256>>>` kernel
 * writes deterministic per-block partial results, which are copied to the
 * host and combined there in a single, fixed sequential loop -- the same
 * pattern `PeriodicGaussianField.cu` uses. `n_blocks` depends only on the
 * grid size, never on scheduling, so a repeated solve on the same field and
 * grid is bitwise-identical. `blas::nrm2_host` (CUB) is reused for `div_rms`
 * only, matching the accepted streamfunction diagnostics' own reduction
 * backend.
 *
 * ## Gate 4 interpretation (no transport claim)
 *
 * This module only CONSTRUCTS the smooth, locally isotropic, triply
 * periodic reference Darcy field with a controlled mean flux -- the regime
 * in which Lester theory forbids purely advective transverse
 * macrodispersion. It makes no transport or macrodispersion claim; nothing
 * produced here may later be cited as evidence about transverse dispersion
 * without the Gate-4 controls of the increments that consume this velocity
 * field.
 *
 * ## Memory
 *
 * `AffinePeriodicFlowWorkspace` stores all three correctors `w_x`, `w_y`,
 * `w_z` (required for the final recombination, so none can be discarded
 * after its own cell-problem solve) plus one recombined `h_tilde` field, RHS
 * scratch, CompactMAC flux scratch reused across the three `K_eff` passes,
 * one shared MG hierarchy/preconditioner/PCG workspace (sequential-only, the
 * same convention as `StreamfunctionWorkspace`), and a fixed-size reduction
 * scratch buffer. Every `DeviceBuffer` follows the project's grow-only
 * convention (`DeviceBuffer::resize` never shrinks capacity); `prepare()` is
 * idempotent for an unchanged grid/config.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../multigrid/mg_types.hpp"
#include "../../numerics/solvers/pcg.cuh"
#include "../../numerics/solvers/projected_positive_mg_preconditioner.cuh"
#include "../../runtime/CudaContext.cuh"
#include "../streamfunctions/affine_periodic_rhs.cuh"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

namespace macroflow3d {
namespace physics {

/**
 * Host configuration for one `solve_affine_periodic_flow` call.
 *
 * `qbar` is the target mean Darcy flux `(qbar[0], qbar[1], qbar[2])`;
 * default `(1, 0, 0)` per the increment spec. `linear` controls every
 * corrector's projected-PCG solve (default `rtol = 1e-10`, the
 * `solvers::ProjectedPCGConfig` project default). `mg` controls the shared
 * MG preconditioner hierarchy, consistent with the accepted streamfunction
 * stack's `multigrid::MGConfig` defaults.
 */
struct AffinePeriodicFlowConfig {
    real qbar[3]{real{1}, real{0}, real{0}};
    solvers::ProjectedPCGConfig linear{};
    multigrid::MGConfig mg{};
};

/**
 * Caller-owned CompactMAC velocity view: `u`/`v`/`w` must be exactly
 * `(nx+1)*ny*nz`, `nx*(ny+1)*nz`, `nx*ny*(nz+1)` elements, matching
 * `physics::VelocityField` / `streamfunctions::CompactMacVelocityView`.
 * Every plane, including the periodic duplicate plane, is written by
 * `solve_affine_periodic_flow`.
 */
struct AffinePeriodicVelocityView {
    DeviceSpan<real> u;
    DeviceSpan<real> v;
    DeviceSpan<real> w;
};

/**
 * Categorized, exact-byte device memory report for one prepared
 * `AffinePeriodicFlowWorkspace` (see the file header's Memory section).
 */
struct AffinePeriodicFlowMemoryReport {
    std::size_t correctors_bytes{};
    std::size_t rhs_scratch_bytes{};
    std::size_t flux_scratch_bytes{};
    std::size_t reduction_scratch_bytes{};
    std::size_t affine_rhs_workspace_bytes{};
    std::size_t pcg_workspace_bytes{};
    std::size_t mg_hierarchy_bytes{};
    std::size_t mg_preconditioner_bytes{};
    std::size_t total_bytes{};
};

/**
 * Host-assembled report for one `solve_affine_periodic_flow` call. See the
 * file header for the exact mathematical meaning of every field.
 */
struct AffinePeriodicFlowReport {
    // K_eff[i][d]: component i of the domain-averaged unique-face flux for
    // corrector direction d (see the file header).
    real K_eff[3][3]{};
    real symmetry_defect_rel{};
    // Eigenvalues of (K_eff + K_eff^T)/2, ascending; positivity evidence.
    real eigenvalues_symmetric_part[3]{};

    // Solution of the full (unsymmetrized) K_eff G = qbar system.
    real G[3]{};

    // Mean over unique faces of the final reconstructed velocity, per
    // component; compare against AffinePeriodicFlowConfig::qbar.
    real achieved_mean_flux[3]{};

    // Discrete mass-conservation diagnostics of the reconstructed velocity.
    real div_max_abs{};
    real div_rms{};

    // Per-direction corrector cell-problem PCG results, in x, y, z order.
    solvers::ProjectedPCGResult corrector_results[3]{};

    AffinePeriodicFlowMemoryReport memory{};
};

/**
 * Owns every scratch/solver buffer required by `solve_affine_periodic_flow`
 * for one grid and one `AffinePeriodicFlowConfig::mg`. See the file header's
 * Memory section for the exact ownership and allocation contract; matches
 * `StreamfunctionWorkspace`'s idempotent-`prepare()`, sequential-solve,
 * grow-only-`DeviceBuffer`, move-only conventions.
 */
class AffinePeriodicFlowWorkspace {
  public:
    AffinePeriodicFlowWorkspace() = default;
    ~AffinePeriodicFlowWorkspace() = default;

    AffinePeriodicFlowWorkspace(const AffinePeriodicFlowWorkspace&) = delete;
    AffinePeriodicFlowWorkspace& operator=(const AffinePeriodicFlowWorkspace&) = delete;
    AffinePeriodicFlowWorkspace(AffinePeriodicFlowWorkspace&&) noexcept = default;
    AffinePeriodicFlowWorkspace& operator=(AffinePeriodicFlowWorkspace&&) noexcept = default;

    // Allocates/resizes every owned buffer and, only if the grid or
    // config.mg actually changed since the last successful prepare(),
    // (re)constructs the MG hierarchy and preconditioner. Throws
    // std::invalid_argument for an invalid grid or config.mg (see
    // AffinePeriodicFlowSolver.cu for the exact messages).
    void prepare(const Grid3D& grid, const AffinePeriodicFlowConfig& config);

    [[nodiscard]] bool prepared_for(const Grid3D& grid,
                                    const AffinePeriodicFlowConfig& config) const noexcept;

    // Exact sum of every owned DeviceBuffer capacity across this workspace
    // and every sub-workspace/hierarchy/preconditioner it owns, in bytes.
    // Never allocates.
    [[nodiscard]] std::size_t allocated_device_bytes() const noexcept;

    // Categorized report built from this workspace's ACTUAL capacities.
    [[nodiscard]] AffinePeriodicFlowMemoryReport memory_report() const;

  private:
    void ensure_prepared() const;

    friend AffinePeriodicFlowReport solve_affine_periodic_flow(
        CudaContext&, const Grid3D&, DeviceSpan<const real>, const AffinePeriodicFlowConfig&,
        AffinePeriodicVelocityView, AffinePeriodicFlowWorkspace&);

    // Three zero-mean periodic head correctors w_x, w_y, w_z (in that
    // order) and their affine recombination h_tilde = sum_d G_d w_d.
    std::array<DeviceBuffer<real>, 3> correctors_;
    DeviceBuffer<real> h_tilde_;

    // RHS scratch for streamfunctions::assemble_affine_periodic_rhs, called
    // once per direction with a degenerate (e_d, e_d) gauge pair; rhs_b_ is
    // discarded after every call (see the file header).
    DeviceBuffer<real> rhs_a_;
    DeviceBuffer<real> rhs_b_;

    // CompactMAC flux scratch, reused across the three K_eff column passes;
    // the final velocity is written directly into the caller-owned
    // AffinePeriodicVelocityView instead.
    DeviceBuffer<real> flux_u_;
    DeviceBuffer<real> flux_v_;
    DeviceBuffer<real> flux_w_;

    // Cell-centered discrete divergence of the final reconstructed
    // velocity.
    DeviceBuffer<real> divergence_;

    // Deterministic two-pass block-reduction scratch (see the file header).
    DeviceBuffer<real> reduce_blocks_;

    streamfunctions::AffinePeriodicRhsWorkspace affine_rhs_workspace_;
    solvers::ProjectedPCGWorkspace pcg_workspace_;

    std::unique_ptr<multigrid::MGHierarchy> mg_hierarchy_;
    std::optional<solvers::ProjectedPositiveMGPreconditioner> mg_preconditioner_;

    Grid3D prepared_grid_{};
    multigrid::MGConfig prepared_mg_config_{};
    bool prepared_{false};
};

/**
 * Solve the SF-19 affine-periodic Darcy problem for one triply periodic
 * conductivity field `K`, writing the reconstructed CompactMAC velocity into
 * `velocity` and using `workspace` for every scratch/solver buffer. See the
 * file header for the exact mathematical construction. Throws
 * `std::invalid_argument` for host-detectable grid/config/span misuse and
 * `std::runtime_error` if the effective conductivity tensor's symmetric part
 * is not positive definite or the tensor is singular (see
 * AffinePeriodicFlowSolver.cu for the exact messages).
 */
[[nodiscard]] AffinePeriodicFlowReport solve_affine_periodic_flow(
    CudaContext& context, const Grid3D& grid, DeviceSpan<const real> K,
    const AffinePeriodicFlowConfig& config, AffinePeriodicVelocityView velocity,
    AffinePeriodicFlowWorkspace& workspace);

} // namespace physics
} // namespace macroflow3d
