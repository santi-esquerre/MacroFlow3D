#pragma once

/**
 * @file pcg.cuh
 * @brief Preconditioned Conjugate Gradient solver
 * @ingroup numerics_solvers
 *
 * Legacy correspondence:
 *   - solver_CG(AH, PCCMG_CG, ...) where PCCMG_CG is MG preconditioner
 *   - rz = dot(r, z), NOT dot(r, r) as in unpreconditioned CG
 *
 * Design (HPC):
 *   - Zero allocations in solve loop
 *   - Workspace passed from outside
 *   - Device-first: sync only every check_every iterations
 *   - Preconditioner is template (duck-typing, not virtual)
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../numerics/blas/blas.cuh"
#include "../../runtime/CudaContext.cuh"

#include <cmath>
#include <cstdint>
#include <limits>

namespace macroflow3d {
namespace solvers {

// PCG configuration
struct PCGConfig {
    int max_iter = 1000;
    int check_every = 10; // Check convergence every N iterations
    real rtol = 1e-6;     // Relative tolerance
    bool verbose = false;
};

// PCG result
struct PCGResult {
    int iterations = 0;
    real initial_residual = 0.0;
    real final_residual = 0.0;
    bool converged = false;
};

// Workspace for PCG (pre-allocated buffers)
struct PCGWorkspace {
    DeviceBuffer<real> r;  // Residual
    DeviceBuffer<real> z;  // Preconditioned residual
    DeviceBuffer<real> p;  // Search direction
    DeviceBuffer<real> Ap; // A * p

    // Reduction workspace for dot/nrm2 (persistent, avoids per-solve alloc)
    blas::ReductionWorkspace red;

    void ensure(size_t n) {
        if (r.size() != n) {
            r.resize(n);
            z.resize(n);
            p.resize(n);
            Ap.resize(n);
        }
    }
};

/**
 * @brief Preconditioned Conjugate Gradient solver
 *
 * Solves A*x = b using PCG with preconditioner M.
 *
 * Algorithm:
 *   r0 = b - A*x0
 *   z0 = M^{-1} * r0
 *   p0 = z0
 *   for k = 0, 1, ...
 *       alpha_k = (r_k, z_k) / (p_k, A*p_k)
 *       x_{k+1} = x_k + alpha_k * p_k
 *       r_{k+1} = r_k - alpha_k * A*p_k
 *       if ||r_{k+1}|| / ||r_0|| < rtol: converged
 *       z_{k+1} = M^{-1} * r_{k+1}
 *       beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
 *       p_{k+1} = z_{k+1} + beta_k * p_k
 *
 * Template parameters:
 *   - Operator: must have `apply(ctx, x, Ax)` method
 *   - Preconditioner: must have `apply(ctx, r, z)` method
 *
 * @param ctx    CUDA context
 * @param A      Linear operator with apply(ctx, x, Ax)
 * @param M      Preconditioner with apply(ctx, r, z)
 * @param b      Right-hand side (device)
 * @param x      Initial guess / solution (device, in/out)
 * @param cfg    PCG configuration
 * @param ws     Pre-allocated workspace
 * @return PCGResult with convergence info
 */
template <typename Operator, typename Preconditioner>
PCGResult pcg_solve(CudaContext& ctx, const Operator& A, const Preconditioner& M,
                    DeviceSpan<const real> b, DeviceSpan<real> x, const PCGConfig& cfg,
                    PCGWorkspace& ws) {
    PCGResult result;
    const size_t n = x.size();

    // Ensure workspace
    ws.ensure(n);

    // Use persistent reduction workspace from PCGWorkspace (no allocation per solve)
    auto& red = ws.red;

    // Spans for convenience
    auto r = ws.r.span();
    auto z = ws.z.span();
    auto p = ws.p.span();
    auto Ap = ws.Ap.span();

    // Initial residual: r0 = b - A*x0
    A.apply(ctx, DeviceSpan<const real>(x), Ap);          // Ap = A*x (temporary use)
    blas::copy(ctx, b, r);                                // r = b
    blas::axpy(ctx, -1.0, DeviceSpan<const real>(Ap), r); // r = b - A*x

    // Initial residual norm
    result.initial_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);

    if (result.initial_residual < 1e-15) {
        // Already converged
        result.converged = true;
        result.final_residual = result.initial_residual;
        return result;
    }

    // Apply preconditioner: z0 = M^{-1} * r0
    M.apply(ctx, DeviceSpan<const real>(r), z);

    // p0 = z0
    blas::copy(ctx, DeviceSpan<const real>(z), p);

    // rz_old = (r, z)
    real rz_old = blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), red);

    // PCG iteration
    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        result.iterations = iter + 1;

        // Ap = A * p
        A.apply(ctx, DeviceSpan<const real>(p), Ap);

        // alpha = rz_old / (p, Ap)
        real pAp = blas::dot_host(ctx, DeviceSpan<const real>(p), DeviceSpan<const real>(Ap), red);
        real alpha = rz_old / pAp;

        // x = x + alpha * p
        blas::axpy(ctx, alpha, DeviceSpan<const real>(p), x);

        // r = r - alpha * Ap
        blas::axpy(ctx, -alpha, DeviceSpan<const real>(Ap), r);

        // Check convergence every check_every iterations
        if ((iter + 1) % cfg.check_every == 0 || iter == cfg.max_iter - 1) {
            result.final_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);
            real relative_res = result.final_residual / result.initial_residual;

            if (relative_res < cfg.rtol) {
                result.converged = true;
                return result;
            }
        }

        // z = M^{-1} * r
        M.apply(ctx, DeviceSpan<const real>(r), z);

        // rz_new = (r, z)
        real rz_new =
            blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), red);

        // beta = rz_new / rz_old
        real beta = rz_new / rz_old;

        // p = z + beta * p
        blas::axpby(ctx, 1.0, DeviceSpan<const real>(z), beta, p);

        rz_old = rz_new;
    }

    // Did not converge within max_iter
    result.final_residual = blas::nrm2_host(ctx, DeviceSpan<const real>(r), red);
    return result;
}

enum class ProjectedPCGStatus {
    converged,
    max_iterations,
    invalid_configuration,
    size_mismatch,
    aliasing,
    breakdown_pAp,
    breakdown_rz,
    nonfinite_value,
};

struct ProjectedPCGConfig {
    int max_iter = 1000;
    int check_every = 10;
    real rtol = 1e-10;
};

struct ProjectedPCGResult {
    int iterations = 0;
    real raw_rhs_mean = 0.0;
    real raw_rhs_l2_norm = 0.0;
    real raw_rhs_compatibility_defect = 0.0;
    real initial_projected_residual = 0.0;
    real final_projected_residual = 0.0;
    real relative_projected_residual = 0.0;
    real final_field_mean = 0.0;
    bool converged = false;
    ProjectedPCGStatus status = ProjectedPCGStatus::invalid_configuration;
};

struct ProjectedPCGWorkspace {
    PCGWorkspace pcg;
    DeviceBuffer<real> projected_rhs;
    constraints::MeanZeroWorkspace mean_zero;

    void prepare(size_t n) {
        pcg.ensure(n);
        projected_rhs.resize(n);
        mean_zero.prepare(n);
    }

    [[nodiscard]] bool prepared_for(size_t n) const noexcept {
        return pcg.r.size() == n && pcg.z.size() == n && pcg.p.size() == n &&
               pcg.Ap.size() == n && pcg.red.d_scalar.size() >= 1 &&
               projected_rhs.size() == n && mean_zero.prepared_for(n);
    }
};

namespace detail {

inline bool spans_overlap(DeviceSpan<const real> left, DeviceSpan<const real> right) noexcept {
    if (left.empty() || right.empty()) {
        return false;
    }
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
    const auto left_end = left_begin + left.size() * sizeof(real);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
    const auto right_end = right_begin + right.size() * sizeof(real);
    return left_begin < right_end && right_begin < left_end;
}

inline bool aliases_projected_workspace(DeviceSpan<const real> field,
                                        const ProjectedPCGWorkspace& workspace) noexcept {
    return spans_overlap(field, workspace.pcg.r.span()) ||
           spans_overlap(field, workspace.pcg.z.span()) ||
           spans_overlap(field, workspace.pcg.p.span()) ||
           spans_overlap(field, workspace.pcg.Ap.span()) ||
           spans_overlap(field, workspace.projected_rhs.span()) ||
           spans_overlap(field, workspace.pcg.red.d_scalar.span());
}

} // namespace detail

/**
 * Solve a periodic singular SPD system in the explicit mean-zero quotient.
 *
 * The caller must prepare `workspace` for the exact input size before this
 * call.  The raw right-hand side is copied before projection and is never
 * modified.  Host reductions are deliberately performed only for diagnostics,
 * scalar recurrence checks, and requested true-residual checks.
 */
template <typename Operator, typename Preconditioner>
ProjectedPCGResult projected_pcg_solve(CudaContext& ctx, const Operator& A, const Preconditioner& M,
                                       DeviceSpan<const real> b, DeviceSpan<real> x,
                                       const ProjectedPCGConfig& cfg,
                                       const constraints::MeanZeroProjector& projector,
                                       ProjectedPCGWorkspace& workspace) {
    ProjectedPCGResult result;
    const size_t n = x.size();

    if (n == 0 || b.size() != n || !workspace.prepared_for(n)) {
        result.status = ProjectedPCGStatus::size_mismatch;
        return result;
    }
    if (cfg.max_iter < 0 || cfg.check_every <= 0 || cfg.rtol < real(0.0) ||
        !std::isfinite(cfg.rtol)) {
        result.status = ProjectedPCGStatus::invalid_configuration;
        return result;
    }
    if (detail::spans_overlap(b, DeviceSpan<const real>(x)) ||
        detail::aliases_projected_workspace(b, workspace) ||
        detail::aliases_projected_workspace(DeviceSpan<const real>(x), workspace)) {
        result.status = ProjectedPCGStatus::aliasing;
        return result;
    }

    auto& pcg = workspace.pcg;
    auto b_hat = workspace.projected_rhs.span();
    auto r = pcg.r.span();
    auto z = pcg.z.span();
    auto p = pcg.p.span();
    auto Ap = pcg.Ap.span();

    result.raw_rhs_mean = projector.mean_host(ctx, b, workspace.mean_zero);
    result.raw_rhs_l2_norm = blas::nrm2_host(ctx, b, pcg.red);
    if (!std::isfinite(result.raw_rhs_mean) || !std::isfinite(result.raw_rhs_l2_norm)) {
        result.status = ProjectedPCGStatus::nonfinite_value;
        return result;
    }
    result.raw_rhs_compatibility_defect =
        result.raw_rhs_l2_norm > real(0.0)
            ? std::abs(result.raw_rhs_mean) /
                  (result.raw_rhs_l2_norm / std::sqrt(static_cast<real>(n)))
            : real(0.0);
    if (!std::isfinite(result.raw_rhs_compatibility_defect)) {
        result.status = ProjectedPCGStatus::nonfinite_value;
        return result;
    }

    blas::copy(ctx, b, b_hat);
    projector.project(ctx, b_hat, workspace.mean_zero);
    projector.project(ctx, x, workspace.mean_zero);

    const auto recompute_true_residual = [&]() {
        A.apply(ctx, DeviceSpan<const real>(x), Ap);
        blas::copy(ctx, DeviceSpan<const real>(b_hat), r);
        blas::axpy(ctx, real(-1.0), DeviceSpan<const real>(Ap), r);
        projector.project(ctx, r, workspace.mean_zero);
        return blas::nrm2_host(ctx, DeviceSpan<const real>(r), pcg.red);
    };
    const auto record_final_diagnostics = [&]() {
        result.final_field_mean =
            projector.mean_host(ctx, DeviceSpan<const real>(x), workspace.mean_zero);
        if (!std::isfinite(result.final_field_mean)) {
            result.status = ProjectedPCGStatus::nonfinite_value;
            result.converged = false;
        }
    };
    const auto set_relative_residual = [&](real residual) {
        result.final_projected_residual = residual;
        result.relative_projected_residual =
            result.initial_projected_residual == real(0.0)
                ? (residual == real(0.0) ? real(0.0) : std::numeric_limits<real>::infinity())
                : residual / result.initial_projected_residual;
    };

    result.initial_projected_residual = recompute_true_residual();
    set_relative_residual(result.initial_projected_residual);
    if (!std::isfinite(result.initial_projected_residual) ||
        !std::isfinite(result.relative_projected_residual)) {
        result.status = ProjectedPCGStatus::nonfinite_value;
        record_final_diagnostics();
        return result;
    }
    if (result.relative_projected_residual <= cfg.rtol) {
        result.status = ProjectedPCGStatus::converged;
        result.converged = true;
        record_final_diagnostics();
        return result;
    }
    if (cfg.max_iter == 0) {
        result.status = ProjectedPCGStatus::max_iterations;
        record_final_diagnostics();
        return result;
    }

    M.apply(ctx, DeviceSpan<const real>(r), z);
    projector.project(ctx, z, workspace.mean_zero);
    blas::copy(ctx, DeviceSpan<const real>(z), p);
    projector.project(ctx, p, workspace.mean_zero);
    real rz_old = blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), pcg.red);
    if (!std::isfinite(rz_old)) {
        result.status = ProjectedPCGStatus::nonfinite_value;
        record_final_diagnostics();
        return result;
    }
    if (rz_old <= real(0.0)) {
        result.status = ProjectedPCGStatus::breakdown_rz;
        record_final_diagnostics();
        return result;
    }

    bool residual_is_current = true;
    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        result.iterations = iter + 1;
        A.apply(ctx, DeviceSpan<const real>(p), Ap);
        const real pAp =
            blas::dot_host(ctx, DeviceSpan<const real>(p), DeviceSpan<const real>(Ap), pcg.red);
        if (!std::isfinite(pAp)) {
            result.status = ProjectedPCGStatus::nonfinite_value;
            break;
        }
        if (pAp <= real(0.0)) {
            result.status = ProjectedPCGStatus::breakdown_pAp;
            break;
        }
        const real alpha = rz_old / pAp;
        if (!std::isfinite(alpha)) {
            result.status = ProjectedPCGStatus::nonfinite_value;
            break;
        }

        blas::axpy(ctx, alpha, DeviceSpan<const real>(p), x);
        projector.project(ctx, x, workspace.mean_zero);
        blas::axpy(ctx, -alpha, DeviceSpan<const real>(Ap), r);
        projector.project(ctx, r, workspace.mean_zero);
        residual_is_current = false;

        if ((iter + 1) % cfg.check_every == 0 || iter == cfg.max_iter - 1) {
            const real residual = recompute_true_residual();
            residual_is_current = true;
            set_relative_residual(residual);
            if (!std::isfinite(residual) || !std::isfinite(result.relative_projected_residual)) {
                result.status = ProjectedPCGStatus::nonfinite_value;
                break;
            }
            if (result.relative_projected_residual <= cfg.rtol) {
                result.status = ProjectedPCGStatus::converged;
                result.converged = true;
                break;
            }
            if (iter == cfg.max_iter - 1) {
                result.status = ProjectedPCGStatus::max_iterations;
                break;
            }
        }

        M.apply(ctx, DeviceSpan<const real>(r), z);
        projector.project(ctx, z, workspace.mean_zero);
        const real rz_new =
            blas::dot_host(ctx, DeviceSpan<const real>(r), DeviceSpan<const real>(z), pcg.red);
        if (!std::isfinite(rz_new)) {
            result.status = ProjectedPCGStatus::nonfinite_value;
            break;
        }
        if (rz_new <= real(0.0)) {
            result.status = ProjectedPCGStatus::breakdown_rz;
            break;
        }
        const real beta = rz_new / rz_old;
        if (!std::isfinite(beta)) {
            result.status = ProjectedPCGStatus::nonfinite_value;
            break;
        }
        blas::axpby(ctx, real(1.0), DeviceSpan<const real>(z), beta, p);
        projector.project(ctx, p, workspace.mean_zero);
        rz_old = rz_new;
    }

    if (!result.converged && result.status == ProjectedPCGStatus::invalid_configuration) {
        result.status = ProjectedPCGStatus::max_iterations;
    }
    if (!residual_is_current) {
        set_relative_residual(recompute_true_residual());
    }
    if (!std::isfinite(result.final_projected_residual) ||
        !std::isfinite(result.relative_projected_residual)) {
        result.status = ProjectedPCGStatus::nonfinite_value;
        result.converged = false;
    }
    record_final_diagnostics();
    return result;
}

} // namespace solvers
} // namespace macroflow3d
