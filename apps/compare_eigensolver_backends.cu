#ifdef MACROFLOW3D_HAS_PETSC

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/blas/reduction_workspace.cuh"
#include "src/physics/common/fields.cuh"
#include "src/physics/common/physics_config.hpp"
#include "src/physics/common/workspaces.cuh"
#include "src/physics/flow/solve_head.cuh"
#include "src/physics/flow/velocity_from_head.cuh"
#include "src/physics/particles/pspta/invariants/EigensolverBackend.cuh"
#include "src/physics/particles/pspta/invariants/TransportOperator3D.cuh"
#include "src/physics/stochastic/stochastic.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/PetscSlepcInit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;
namespace blas = macroflow3d::blas;

namespace {

struct FieldDiagnostics {
    double norm_psi = 0.0;
    double norm_Dpsi = 0.0;
    double rayleigh = 0.0;
    double direct_residual = 0.0;
};

struct BackendRun {
    std::string backend_name;
    EigensolverResult result;
    std::vector<DeviceBuffer<real>> eigenvectors;
    std::vector<FieldDiagnostics> mode_diags;
    std::vector<double> mode_captures;
};

struct CaseSpec {
    std::string name;
    Grid3D grid;
    VelocityField velocity;
    bool has_expected_yz_subspace = false;

    explicit CaseSpec(const std::string& case_name, const Grid3D& g)
        : name(case_name), grid(g), velocity(g) {}
};

static void fill_uniform_velocity(VelocityField& vel, real vx, real vy, real vz) {
    {
        std::vector<real> h(vel.size_U(), vx);
        cudaMemcpy(vel.U.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
    {
        std::vector<real> h(vel.size_V(), vy);
        cudaMemcpy(vel.V.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
    {
        std::vector<real> h(vel.size_W(), vz);
        cudaMemcpy(vel.W.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
}

static void fill_layered_x_velocity(VelocityField& vel) {
    std::vector<real> u(vel.size_U(), 0.0);
    std::vector<real> v(vel.size_V(), 0.0);
    std::vector<real> w(vel.size_W(), 0.0);

    for (int k = 0; k < vel.nz; ++k) {
        for (int j = 0; j < vel.ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * static_cast<double>(vel.dy);
            const double z = (static_cast<double>(k) + 0.5) * static_cast<double>(vel.dz);
            const double amp =
                1.0 + 0.25 * std::sin(2.0 * M_PI * y) + 0.15 * std::cos(2.0 * M_PI * z);
            for (int i = 0; i <= vel.nx; ++i) {
                u[vel.idx_U(i, j, k)] = static_cast<real>(amp);
            }
        }
    }

    cudaMemcpy(vel.U.data(), u.data(), u.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.V.data(), v.data(), v.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.W.data(), w.data(), w.size() * sizeof(real), cudaMemcpyHostToDevice);
}

static std::vector<real> make_probe_field(const Grid3D& grid, const std::string& probe_name) {
    std::vector<real> h(grid.num_cells(), 0.0);
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = static_cast<size_t>(i + grid.nx * (j + grid.ny * k));
                const double x = (static_cast<double>(i) + 0.5) * static_cast<double>(grid.dx);
                const double y = (static_cast<double>(j) + 0.5) * static_cast<double>(grid.dy);
                const double z = (static_cast<double>(k) + 0.5) * static_cast<double>(grid.dz);

                if (probe_name == "sin_y") {
                    h[c] = std::sin(2.0 * M_PI * y / static_cast<double>(grid.Ly()));
                } else if (probe_name == "cos_y") {
                    h[c] = std::cos(2.0 * M_PI * y / static_cast<double>(grid.Ly()));
                } else if (probe_name == "sin_z") {
                    h[c] = std::sin(2.0 * M_PI * z / static_cast<double>(grid.Lz()));
                } else if (probe_name == "cos_z") {
                    h[c] = std::cos(2.0 * M_PI * z / static_cast<double>(grid.Lz()));
                } else if (probe_name == "x") {
                    h[c] = x;
                }
            }
        }
    }
    return h;
}

static FieldDiagnostics analyze_field(const DeviceBuffer<real>& psi_buf, CombinedOperatorA& A,
                                      TransportOperator3D& D, CudaContext& cuda_ctx,
                                      double lambda_hint) {
    FieldDiagnostics diag;
    const size_t n = psi_buf.size();

    DeviceBuffer<real> d_Dpsi(n), d_Apsi(n), d_resid(n);
    DeviceSpan<const real> psi(psi_buf.data(), n);
    blas::ReductionWorkspace ws;

    D.apply_D(psi, d_Dpsi.span(), cuda_ctx.cuda_stream());
    A.apply_A(psi, d_Apsi.span(), cuda_ctx.cuda_stream());
    cudaStreamSynchronize(cuda_ctx.cuda_stream());

    diag.norm_psi = blas::nrm2_host(cuda_ctx, psi, ws);
    diag.norm_Dpsi = blas::nrm2_host(cuda_ctx, DeviceSpan<const real>(d_Dpsi.data(), n), ws);

    const double dot_psi_Apsi =
        blas::dot_host(cuda_ctx, psi, DeviceSpan<const real>(d_Apsi.data(), n), ws);
    diag.rayleigh = (diag.norm_psi > 1e-30) ? dot_psi_Apsi / (diag.norm_psi * diag.norm_psi) : 0.0;

    blas::copy(cuda_ctx, DeviceSpan<const real>(d_Apsi.data(), n), d_resid.span());
    blas::axpy(cuda_ctx, -lambda_hint, psi, d_resid.span());
    diag.direct_residual =
        blas::nrm2_host(cuda_ctx, DeviceSpan<const real>(d_resid.data(), n), ws) /
        std::max(diag.norm_psi, 1e-30);

    return diag;
}

static double subspace_capture(CudaContext& cuda_ctx, const DeviceBuffer<real>& psi_buf,
                               const std::vector<DeviceBuffer<real>>& basis) {
    if (basis.empty())
        return -1.0;

    blas::ReductionWorkspace ws;
    DeviceSpan<const real> psi(psi_buf.data(), psi_buf.size());
    const double norm_psi = blas::nrm2_host(cuda_ctx, psi, ws);
    if (norm_psi <= 1e-30)
        return 0.0;

    double capture = 0.0;
    for (const auto& basis_vec : basis) {
        DeviceSpan<const real> bj(basis_vec.data(), basis_vec.size());
        const double norm_b = blas::nrm2_host(cuda_ctx, bj, ws);
        if (norm_b <= 1e-30)
            continue;
        const double overlap = blas::dot_host(cuda_ctx, psi, bj, ws) / (norm_psi * norm_b);
        capture += overlap * overlap;
    }
    return capture;
}

static double eigenspace_similarity(CudaContext& cuda_ctx, const std::vector<DeviceBuffer<real>>& a,
                                    const std::vector<DeviceBuffer<real>>& b) {
    if (a.size() < 2 || b.size() < 2)
        return -1.0;

    blas::ReductionWorkspace ws;
    double sum_sq = 0.0;
    for (int i = 0; i < 2; ++i) {
        const DeviceSpan<const real> ai(a[i].data(), a[i].size());
        const double ni = blas::nrm2_host(cuda_ctx, ai, ws);
        for (int j = 0; j < 2; ++j) {
            const DeviceSpan<const real> bj(b[j].data(), b[j].size());
            const double nj = blas::nrm2_host(cuda_ctx, bj, ws);
            if (ni <= 1e-30 || nj <= 1e-30)
                continue;
            const double cij = blas::dot_host(cuda_ctx, ai, bj, ws) / (ni * nj);
            sum_sq += cij * cij;
        }
    }
    return 0.5 * sum_sq;
}

static std::vector<DeviceBuffer<real>> make_expected_yz_basis(const Grid3D& grid) {
    std::vector<DeviceBuffer<real>> basis;
    for (const std::string& probe_name : {"sin_y", "cos_y", "sin_z", "cos_z"}) {
        DeviceBuffer<real> probe(grid.num_cells());
        const std::vector<real> h_probe = make_probe_field(grid, probe_name);
        cudaMemcpy(probe.data(), h_probe.data(), h_probe.size() * sizeof(real),
                   cudaMemcpyHostToDevice);
        basis.push_back(std::move(probe));
    }
    return basis;
}

static CaseSpec make_uniform_case() {
    Grid3D grid(16, 16, 16, 1.0, 1.0, 1.0);
    CaseSpec spec("uniform_x", grid);
    fill_uniform_velocity(spec.velocity, 1.0, 0.0, 0.0);
    spec.has_expected_yz_subspace = true;
    return spec;
}

static CaseSpec make_layered_case() {
    Grid3D grid(16, 16, 16, 1.0, 1.0, 1.0);
    CaseSpec spec("layered_x", grid);
    fill_layered_x_velocity(spec.velocity);
    spec.has_expected_yz_subspace = true;
    return spec;
}

static BCSpec make_darcy_bc() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, 1.0);
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = BCFace(BCType::Periodic, 0.0);
    bc.ymax = BCFace(BCType::Periodic, 0.0);
    bc.zmin = BCFace(BCType::Periodic, 0.0);
    bc.zmax = BCFace(BCType::Periodic, 0.0);
    return bc;
}

static CaseSpec make_small_darcy_case(CudaContext& ctx) {
    Grid3D grid(12, 12, 12, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0);
    CaseSpec spec("darcy_small", grid);

    StochasticConfig stoch_cfg;
    stoch_cfg.sigma2 = 0.25;
    stoch_cfg.corr_length = 0.25;
    stoch_cfg.n_modes = 256;
    stoch_cfg.seed = 20260416;
    stoch_cfg.K_geometric_mean = 1.0;

    HeadSolveConfig head_cfg;
    head_cfg.solver_type = HeadSolverType::PCG_MG;
    head_cfg.mg_levels = 3;
    head_cfg.mg_pre_smooth = 2;
    head_cfg.mg_post_smooth = 2;
    head_cfg.mg_max_cycles = 20;
    head_cfg.mg_coarse_iters = 32;
    head_cfg.cg_max_iter = 400;
    head_cfg.cg_check_every = 5;
    head_cfg.cg_rtol = 1e-8;
    head_cfg.rtol = 1e-6;

    const BCSpec bc = make_darcy_bc();

    KField K(grid);
    HeadField head(grid);
    StochasticWorkspace stoch_ws;
    stoch_ws.allocate(grid, stoch_cfg);
    FlowWorkspace flow_ws;
    flow_ws.allocate(grid, head_cfg.mg_levels);

    generate_K_field(K.span(), stoch_ws, grid, stoch_cfg, ctx);
    init_head_guess(head.span(), grid, bc, ctx);
    const HeadSolveResult solve_result =
        solve_head(head.span(), K.span(), grid, bc, head_cfg, ctx, flow_ws);
    if (!solve_result.converged) {
        throw std::runtime_error("small Darcy control case did not converge");
    }

    compute_velocity_from_head(spec.velocity, head, K, grid, bc, ctx);
    spec.has_expected_yz_subspace = false;
    return spec;
}

static BackendRun run_backend(const std::string& backend_name, CombinedOperatorA& A,
                              TransportOperator3D& D, const Grid3D& grid, bool expected_yz,
                              CudaContext& ctx) {
    BackendRun run;
    run.backend_name = backend_name;

    auto backend = create_eigensolver_backend(backend_name);
    if (!backend) {
        run.result.message = "backend unavailable";
        return run;
    }

    EigensolverConfig cfg;
    cfg.n_eigenvectors = 2;
    cfg.tolerance = 1.0e-8;
    cfg.max_iterations = 500;
    cfg.verbose = false;

    run.result = backend->solve(A, cfg, ctx, run.eigenvectors);

    std::vector<DeviceBuffer<real>> yz_basis;
    if (expected_yz) {
        yz_basis = make_expected_yz_basis(grid);
    }

    for (size_t i = 0; i < run.eigenvectors.size(); ++i) {
        run.mode_diags.push_back(
            analyze_field(run.eigenvectors[i], A, D, ctx, run.result.eigenvalues[i]));
        if (expected_yz)
            run.mode_captures.push_back(subspace_capture(ctx, run.eigenvectors[i], yz_basis));
    }

    return run;
}

static void print_run_summary(const BackendRun& run) {
    std::printf("  backend=%s  success=%s  iters=%d  elapsed=%.1f ms\n", run.backend_name.c_str(),
                run.result.success ? "yes" : "no", run.result.iterations, run.result.elapsed_ms);
    for (size_t i = 0; i < run.result.eigenvalues.size(); ++i) {
        const double d_ratio = run.mode_diags[i].norm_psi > 0.0
                                   ? run.mode_diags[i].norm_Dpsi / run.mode_diags[i].norm_psi
                                   : 0.0;
        std::printf("    λ[%zu]=%.8e  residual=%.3e  ||Dψ||/||ψ||=%.3e  direct=%.3e\n", i,
                    run.result.eigenvalues[i], run.result.residual_norms[i], d_ratio,
                    run.mode_diags[i].direct_residual);
    }
    if (!run.mode_captures.empty()) {
        for (size_t i = 0; i < run.mode_captures.size(); ++i) {
            std::printf("    capture[%zu](expected yz subspace)=%.6f\n", i, run.mode_captures[i]);
        }
    }
}

} // namespace

int main() {
    runtime::PetscSlepcInit::ensure();
    CudaContext ctx(0);

    const double mu = 1.0e-4;
    std::vector<CaseSpec> cases;
    cases.push_back(make_uniform_case());
    cases.push_back(make_layered_case());
    cases.push_back(make_small_darcy_case(ctx));

    bool all_cases_aligned = true;
    bool any_case_needs_gate3_work = false;

    std::printf("=== Eigensolver Backend Comparison ===\n");
    std::printf("  reference backend : slepc_validation\n");
    std::printf("  production backend: slepc\n");
    std::printf("  diagnostic assembly: probing only, not a correctness reference\n\n");

    for (auto& case_spec : cases) {
        std::printf("=== Case: %s ===\n", case_spec.name.c_str());

        TransportOperatorConfig D_cfg;
        D_cfg.x_bc = TransportXBoundary::OneSided;
        TransportOperator3D D(&case_spec.velocity, case_spec.grid, D_cfg);
        LaplacianOperator3D L(case_spec.grid, LaplacianOperator3D::XBoundary::Neumann);
        CombinedOperatorA A(&D, &L, mu);

        const BackendRun validation = run_backend("slepc_validation", A, D, case_spec.grid,
                                                  case_spec.has_expected_yz_subspace, ctx);
        const BackendRun production =
            run_backend("slepc", A, D, case_spec.grid, case_spec.has_expected_yz_subspace, ctx);

        print_run_summary(validation);
        print_run_summary(production);

        bool aligned = validation.result.success && production.result.success;
        bool gate3_ready = validation.result.success && production.result.success;
        const int n_modes = std::min<int>(validation.result.eigenvalues.size(),
                                          production.result.eigenvalues.size());

        std::printf("  comparison:\n");
        for (int i = 0; i < n_modes; ++i) {
            const double lv = validation.result.eigenvalues[i];
            const double lp = production.result.eigenvalues[i];
            const double rel =
                (std::fabs(lv) > 1e-30) ? std::fabs(lp - lv) / std::fabs(lv) : std::fabs(lp - lv);
            const double val_d =
                validation.mode_diags[i].norm_psi > 0.0
                    ? validation.mode_diags[i].norm_Dpsi / validation.mode_diags[i].norm_psi
                    : 0.0;
            const double prod_d =
                production.mode_diags[i].norm_psi > 0.0
                    ? production.mode_diags[i].norm_Dpsi / production.mode_diags[i].norm_psi
                    : 0.0;
            std::printf("    mode %d: λ_val=%.8e  λ_prod=%.8e  relΔ=%.3e  D-ratio(val/prod)=%.3e / "
                        "%.3e  direct(val/prod)=%.3e / %.3e\n",
                        i, lv, lp, rel, val_d, prod_d, validation.mode_diags[i].direct_residual,
                        production.mode_diags[i].direct_residual);
            aligned = aligned && (rel < 2.5e-1);
            aligned = aligned && (validation.mode_diags[i].direct_residual < 1e-3);
            aligned = aligned && (production.mode_diags[i].direct_residual < 1e-3);
            gate3_ready = gate3_ready && (val_d < 1e-2);
            gate3_ready = gate3_ready && (prod_d < 1e-2);
        }

        if (case_spec.has_expected_yz_subspace) {
            for (size_t i = 0; i < validation.mode_captures.size(); ++i) {
                std::printf("    expected yz capture[%zu]: val=%.6f  prod=%.6f\n", i,
                            validation.mode_captures[i], production.mode_captures[i]);
                aligned = aligned && (validation.mode_captures[i] > 0.95);
                aligned = aligned && (production.mode_captures[i] > 0.95);
            }
        } else {
            const double subspace =
                eigenspace_similarity(ctx, validation.eigenvectors, production.eigenvectors);
            std::printf("    eigenspace similarity = %.6f\n", subspace);
            aligned = aligned && (subspace > 0.95);
        }

        if (!gate3_ready) {
            std::printf("    note: backend agreement holds, but absolute invariance quality still "
                        "needs Gate 3 work.\n");
        }

        std::printf("  verdict: %s\n\n", aligned ? "aligned" : "MISMATCH");
        all_cases_aligned = all_cases_aligned && aligned;
        any_case_needs_gate3_work = any_case_needs_gate3_work || !gate3_ready;
    }

    std::printf("=== Overall backend verdict: %s ===\n",
                all_cases_aligned ? "production aligned with validation reference"
                                  : "production NOT aligned with validation reference");
    std::printf("=== Gate 3 readiness note: %s ===\n",
                any_case_needs_gate3_work
                    ? "backend alignment established, but invariant quality work remains open"
                    : "no immediate Gate 3 quality warning from these cases");
    return all_cases_aligned ? 0 : 1;
}

#else

#include <cstdio>
int main() {
    std::printf("compare_eigensolver_backends: PETSc not enabled. Skipping.\n");
    return 0;
}

#endif
