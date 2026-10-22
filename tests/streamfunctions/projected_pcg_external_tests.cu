/**
 * @file projected_pcg_external_tests.cu
 * @brief SF-40 L1: `projected_pcg_solve_external` is bit-for-bit identical to
 *        `projected_pcg_solve` on the SF-19 periodic Darcy corrector problem, with
 *        z == Ap == MG level-0 b, b == MG level-0 x, and b_hat in either an own device
 *        buffer or pinned mapped host memory; plus the aliasing / size guards.
 *
 * Fast mode (default, < 60 s locally): 32^3, 64x16x16, 16^3 guards.
 * `--full` (V100 only): the same comparisons at 256^3 and 512x64x64, eps 2.
 * Output: one `CHECK <name> PASS|FAIL` line per check, then `ALL_PASS` / `FAILED n`.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/multigrid/coefficient_hierarchy.cuh"
#include "src/multigrid/mg_types.hpp"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/numerics/solvers/projected_positive_mg_preconditioner.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/physics/streamfunctions/affine_periodic_rhs.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace macroflow3d;
using solvers::ProjectedPCGConfig;
using solvers::ProjectedPCGExternalVectors;
using solvers::ProjectedPCGResult;
using solvers::ProjectedPCGStatus;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
int g_failures = 0;

void check(bool ok, const std::string& name) {
    std::printf("CHECK %s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

bool bits_eq(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Compare every field of the result bitwise; returns the first differing field name or "".
std::string result_diff(const ProjectedPCGResult& a, const ProjectedPCGResult& b) {
    if (a.iterations != b.iterations) return "iterations";
    if (a.status != b.status) return "status";
    if (a.converged != b.converged) return "converged";
    if (!bits_eq(a.raw_rhs_mean, b.raw_rhs_mean)) return "raw_rhs_mean";
    if (!bits_eq(a.raw_rhs_l2_norm, b.raw_rhs_l2_norm)) return "raw_rhs_l2_norm";
    if (!bits_eq(a.raw_rhs_compatibility_defect, b.raw_rhs_compatibility_defect))
        return "raw_rhs_compatibility_defect";
    if (!bits_eq(a.initial_projected_residual, b.initial_projected_residual))
        return "initial_projected_residual";
    if (!bits_eq(a.final_projected_residual, b.final_projected_residual))
        return "final_projected_residual";
    if (!bits_eq(a.relative_projected_residual, b.relative_projected_residual))
        return "relative_projected_residual";
    if (!bits_eq(a.final_field_mean, b.final_field_mean)) return "final_field_mean";
    return "";
}

std::vector<real> download(CudaContext& ctx, DeviceSpan<const real> s) {
    ctx.synchronize(); // ctx-stream work must finish before a legacy-stream D2H
    std::vector<real> h(s.size());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), s.data(), s.size() * sizeof(real),
                                      cudaMemcpyDeviceToHost));
    return h;
}

// SF-19 corrector problem: K = exp(eps*Y), MG hierarchy of K, Lester operator, RHS div(K e_d).
struct Problem {
    Grid3D grid;
    size_t n = 0;
    DeviceBuffer<real> K, rhs_a, rhs_b;
    multigrid::MGConfig mg;
    std::unique_ptr<multigrid::MGHierarchy> hierarchy;
    std::unique_ptr<solvers::ProjectedPositiveMGPreconditioner> M;
    std::unique_ptr<operators::LesterPositiveDiffusionOperator> A;

    Problem(CudaContext& ctx, int nx, int ny, int nz, double eps, int d) {
        const int nmin = std::min(nx, std::min(ny, nz));
        const double h = 1.0 / nmin; // isotropic spacing; domain (nx/nmin) x 1 x 1
        grid = Grid3D(nx, ny, nz, h, h, h);
        n = grid.num_cells();
        const double Lx = nx * h, Ly = ny * h, Lz = nz * h;
        std::vector<real> hostK(n);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double x = (i + 0.5) * h / Lx, y = (j + 0.5) * h / Ly,
                                 z = (k + 0.5) * h / Lz;
                    const double Y =
                        0.7 * std::sin(kTwoPi * x + 0.4) * std::cos(kTwoPi * y) +
                        0.5 * std::cos(kTwoPi * 2.0 * y + 1.1) * std::sin(kTwoPi * z + 0.2) +
                        0.3 * std::sin(kTwoPi * (x - y + 2.0 * z) + 0.9);
                    hostK[grid.idx(i, j, k)] = std::exp(eps * Y);
                }
        K.resize(n);
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(K.data(), hostK.data(), n * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize()); // legacy-stream upload before ctx work

        // Same level rule as slab_auto_mg_levels, applied to the smallest dimension.
        int levels = 1, m = nmin;
        while (m % 2 == 0 && m / 2 >= 4 && (m / 2) % 2 == 0) {
            m /= 2;
            ++levels;
        }
        mg.num_levels = levels;
        hierarchy = std::make_unique<multigrid::MGHierarchy>(grid, levels);
        multigrid::populate_coefficient_hierarchy(ctx, *hierarchy, DeviceSpan<const real>(K.span()));
        M = std::make_unique<solvers::ProjectedPositiveMGPreconditioner>(*hierarchy, mg);
        A = std::make_unique<operators::LesterPositiveDiffusionOperator>(
            grid, DeviceSpan<const real>(K.span()));

        rhs_a.resize(n);
        rhs_b.resize(n);
        streamfunctions::AffineGauge gauge;
        const streamfunctions::AffineGradient e = {real(d == 0), real(d == 1), real(d == 2)};
        gauge.psi1_gradient = e;
        gauge.psi2_gradient = e;
        streamfunctions::AffinePeriodicRhsWorkspace ws;
        ws.prepare(n);
        (void)streamfunctions::assemble_affine_periodic_rhs(ctx, grid, DeviceSpan<const real>(K.span()),
                                                           gauge, rhs_a.span(), rhs_b.span(), ws);
        ctx.synchronize();
    }
};

void zero(CudaContext& ctx, DeviceBuffer<real>& buf) {
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(buf.data(), 0, buf.size() * sizeof(real), ctx.cuda_stream()));
    ctx.synchronize();
}

void run_case(CudaContext& ctx, const std::string& name, int nx, int ny, int nz, double eps, int d,
              int max_iter) {
    Problem P(ctx, nx, ny, nz, eps, d);
    ProjectedPCGConfig cfg;
    cfg.rtol = 1e-10;
    if (max_iter > 0) cfg.max_iter = max_iter;
    constraints::MeanZeroProjector projector;

    // Reference: legacy entry point with its own workspace.
    DeviceBuffer<real> x_ref(P.n);
    zero(ctx, x_ref);
    solvers::ProjectedPCGWorkspace ws;
    ws.prepare(P.n);
    const ProjectedPCGResult ref = solvers::projected_pcg_solve(
        ctx, *P.A, *P.M, DeviceSpan<const real>(P.rhs_a.span()), x_ref.span(), cfg, projector, ws);
    const std::vector<real> sol_ref = download(ctx, DeviceSpan<const real>(x_ref.span()));
    std::printf("INFO %s ref status=%d iterations=%d rel=%.3e\n", name.c_str(),
                static_cast<int>(ref.status), ref.iterations, ref.relative_projected_residual);

    for (int mode = 0; mode < 2; ++mode) {
        const bool mapped = mode == 1;
        const std::string tag = name + (mapped ? "/b_hat=mapped" : "/b_hat=device");

        auto& lvl0 = P.hierarchy->levels.front();
        // b lives in MG level-0 x; z = Ap = MG level-0 b (exact identity).
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(lvl0.x.data(), P.rhs_a.data(), P.n * sizeof(real),
                                               cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
        ctx.synchronize();
        DeviceBuffer<real> x_var(P.n), r(P.n), p(P.n), b_hat_dev;
        zero(ctx, x_var);
        real* b_hat_host = nullptr;
        real* b_hat_ptr = nullptr;
        if (mapped) {
            MACROFLOW3D_CUDA_CHECK(
                cudaHostAlloc(&b_hat_host, P.n * sizeof(real), cudaHostAllocMapped));
            MACROFLOW3D_CUDA_CHECK(
                cudaHostGetDevicePointer(reinterpret_cast<void**>(&b_hat_ptr), b_hat_host, 0));
        } else {
            b_hat_dev.resize(P.n);
            b_hat_ptr = b_hat_dev.data();
        }
        blas::ReductionWorkspace red;
        constraints::MeanZeroWorkspace mz;
        mz.prepare(P.n);

        ProjectedPCGExternalVectors v;
        v.r = r.span();
        v.p = p.span();
        v.z = lvl0.b.span();
        v.Ap = lvl0.b.span();
        v.b_hat = DeviceSpan<real>(b_hat_ptr, P.n);
        v.red = &red;
        v.mean_zero = &mz;

        const ProjectedPCGResult var = solvers::projected_pcg_solve_external(
            ctx, *P.A, *P.M, DeviceSpan<const real>(lvl0.x.span()), x_var.span(), cfg, projector, v);
        const std::vector<real> sol_var = download(ctx, DeviceSpan<const real>(x_var.span()));

        const std::string diff = result_diff(ref, var);
        check(diff.empty(), tag + " result_fields_bitwise" + (diff.empty() ? "" : " (" + diff + ")"));
        check(sol_ref.size() == sol_var.size() &&
                  std::memcmp(sol_ref.data(), sol_var.data(), sol_ref.size() * sizeof(real)) == 0,
              tag + " solution_bitwise");
        check(var.status == ref.status && var.iterations > 0, tag + " nontrivial_solve");
        if (mapped) {
            MACROFLOW3D_CUDA_CHECK(cudaFreeHost(b_hat_host));
        }
    }
}

void guard_cases(CudaContext& ctx) {
    Problem P(ctx, 16, 16, 16, 1.0, 0);
    const size_t n = P.n;
    ProjectedPCGConfig cfg;
    constraints::MeanZeroProjector projector;
    DeviceBuffer<real> b(n), x(n), r(n), p(n), z(n), Ap(n), bh(n), big(n + 1);
    zero(ctx, x);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(b.data(), P.rhs_a.data(), n * sizeof(real),
                                           cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
    ctx.synchronize();
    blas::ReductionWorkspace red;
    constraints::MeanZeroWorkspace mz;
    mz.prepare(n);

    const auto base = [&]() {
        ProjectedPCGExternalVectors v;
        v.r = r.span();
        v.p = p.span();
        v.z = z.span();
        v.Ap = Ap.span();
        v.b_hat = bh.span();
        v.red = &red;
        v.mean_zero = &mz;
        return v;
    };
    const auto status_of = [&](DeviceSpan<const real> bs, DeviceSpan<real> xs,
                               ProjectedPCGExternalVectors v) {
        return solvers::projected_pcg_solve_external(ctx, *P.A, *P.M, bs, xs, cfg, projector, v)
            .status;
    };
    const DeviceSpan<const real> B(b.span());

    {
        auto v = base();
        v.z = DeviceSpan<real>(big.data(), n);
        v.Ap = DeviceSpan<real>(big.data() + 1, n); // partial overlap, not identical
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::aliasing, "guard z_Ap_partial_overlap");
    }
    {
        auto v = base();
        v.z = DeviceSpan<real>(big.data(), n);
        v.Ap = DeviceSpan<real>(big.data(), n - 1); // same pointer, different size: not identity
        // size differs from n -> size_mismatch is the correct (earlier) guard
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::size_mismatch, "guard z_Ap_wrong_size");
    }
    {
        auto v = base();
        v.Ap = v.z; // exact identity accepted
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::converged, "guard z_eq_Ap_accepted");
    }
    {
        auto v = base();
        v.p = v.r; // overlap among distinct members
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::aliasing, "guard r_p_overlap");
    }
    {
        auto v = base();
        v.b_hat = v.Ap;
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::aliasing, "guard b_hat_Ap_overlap");
    }
    {
        auto v = base();
        check(status_of(DeviceSpan<const real>(x.span()), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_x");
    }
    {
        auto v = base();
        check(status_of(DeviceSpan<const real>(v.r), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_r");
        check(status_of(DeviceSpan<const real>(v.p), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_p");
        check(status_of(DeviceSpan<const real>(v.z), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_z");
        check(status_of(DeviceSpan<const real>(v.Ap), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_Ap");
        check(status_of(DeviceSpan<const real>(v.b_hat), x.span(), v) == ProjectedPCGStatus::aliasing,
              "guard b_overlaps_b_hat");
    }
    {
        // d_scalar has 1 element, so overlap it from a size-n vector that starts at it.
        auto v = base();
        v.r = DeviceSpan<real>(red.d_scalar.data(), n);
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::aliasing, "guard r_overlaps_d_scalar");
    }
    {
        auto v = base();
        check(status_of(B, v.r, v) == ProjectedPCGStatus::aliasing, "guard x_overlaps_r");
        check(status_of(B, v.z, v) == ProjectedPCGStatus::aliasing, "guard x_overlaps_z");
    }
    {
        auto v = base();
        v.red = nullptr;
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::size_mismatch, "guard red_null");
        v = base();
        v.mean_zero = nullptr;
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::size_mismatch, "guard mean_zero_null");
        v = base();
        v.b_hat = DeviceSpan<real>(bh.data(), n - 1);
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::size_mismatch, "guard b_hat_wrong_size");
    }
    {
        auto v = base();
        cfg.check_every = 0;
        check(status_of(B, x.span(), v) == ProjectedPCGStatus::invalid_configuration,
              "guard invalid_configuration");
        cfg.check_every = 10;
    }
}

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) full = true;
    }
    // Needed on non-UVA setups for cudaHostAllocMapped; harmless if the context already exists.
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaGetLastError();

    CudaContext ctx(0);
    if (full) {
        run_case(ctx, "256x256x256_eps2_e0", 256, 256, 256, 2.0, 0, 0);
        run_case(ctx, "512x64x64_eps2_e0", 512, 64, 64, 2.0, 0, 0);
    } else {
        run_case(ctx, "32x32x32_eps1_e0", 32, 32, 32, 1.0, 0, 0);
        run_case(ctx, "32x32x32_eps2_e1", 32, 32, 32, 2.0, 1, 0);
        run_case(ctx, "64x16x16_eps2_e0", 64, 16, 16, 2.0, 0, 0);
        run_case(ctx, "32x32x32_eps2_e0_maxiter7", 32, 32, 32, 2.0, 0, 7);
    }
    guard_cases(ctx);

    if (g_failures == 0) {
        std::printf("ALL_PASS\n");
        return 0;
    }
    std::printf("FAILED %d\n", g_failures);
    return 1;
}
