/**
 * @file validate_slepc_eigensolver.cu
 * @brief End-to-end validation of the SLEPc eigensolver backend.
 *
 * Creates a synthetic uniform-flow velocity field (vx = 1, vy = vz = 0)
 * on a small 3D grid, builds the combined operator A = D†D + μL,
 * solves for the 2 smallest non-trivial eigenvalues using SLEPcBackend,
 * and prints:
 *   - Eigenvalues and residual norms
 *   - GPU execution evidence (VecType, MatType, KSP, PC, EPS)
 *   - ||D ψ_i|| for each eigenvector (should be small → invariant)
 *
 * Usage:
 *   ./validate_slepc_eigensolver [-eps_monitor] [-log_view]
 */

#ifdef MACROFLOW3D_HAS_PETSC

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/blas/reduction_workspace.cuh"
#include "src/physics/common/fields.cuh"
#include "src/physics/particles/pspta/invariants/EigensolverBackend.cuh"
#include "src/physics/particles/pspta/invariants/PsptaInvariantField.cuh"
#include "src/physics/particles/pspta/invariants/SLEPcBackend.cuh"
#include "src/physics/particles/pspta/invariants/TransportOperator3D.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/PetscSlepcInit.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;
namespace blas = macroflow3d::blas;

/// Fill U (x-face velocity) with a uniform value on the host, then copy.
static void fill_uniform_velocity(VelocityField& vel, real vx, real vy, real vz) {
    const int nx = vel.nx, ny = vel.ny, nz = vel.nz;

    // U: (nx+1) * ny * nz  faces in x
    {
        std::vector<real> h(vel.size_U(), vx);
        cudaMemcpy(vel.U.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
    // V: nx * (ny+1) * nz  faces in y
    {
        std::vector<real> h(vel.size_V(), vy);
        cudaMemcpy(vel.V.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
    // W: nx * ny * (nz+1)  faces in z
    {
        std::vector<real> h(vel.size_W(), vz);
        cudaMemcpy(vel.W.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
    }
}

int main(int argc, char** argv) {
    // ------------------------------------------------------------------
    // 0. Initialization
    // ------------------------------------------------------------------
    runtime::PetscSlepcInit::ensure();

    CudaContext cuda_ctx(0);

    // ------------------------------------------------------------------
    // 1. Problem setup
    // ------------------------------------------------------------------
    constexpr int NX = 16, NY = 16, NZ = 16;
    constexpr real LX = 1.0, LY = 1.0, LZ = 1.0;
    Grid3D grid(NX, NY, NZ, LX, LY, LZ);

    std::printf("=== SLEPc Eigensolver Validation ===\n");
    std::printf("  Grid: %d x %d x %d   (N = %zu)\n", NX, NY, NZ, grid.num_cells());

    // Uniform flow: vx = 1, vy = vz = 0
    // Expected: y and z are exact invariants → Dψ = 0 for ψ = y or ψ = z
    VelocityField vel(grid);
    fill_uniform_velocity(vel, 1.0, 0.0, 0.0);

    std::printf("  Velocity: uniform vx = 1.0, vy = vz = 0.0\n");

    // ------------------------------------------------------------------
    // 2. Build operators
    // ------------------------------------------------------------------
    TransportOperatorConfig D_cfg;
    D_cfg.x_bc = TransportXBoundary::OneSided;

    TransportOperator3D D(&vel, grid, D_cfg);
    LaplacianOperator3D L(grid, LaplacianOperator3D::XBoundary::Neumann);
    CombinedOperatorA A(&D, &L, /*mu=*/1.0e-4);

    std::printf("  Operators: D (v·∇), L (-∇²), A = D†D + μL, μ = %.1e\n", A.mu());

    // ------------------------------------------------------------------
    // 3. Run SLEPc eigensolver
    // ------------------------------------------------------------------
    EigensolverConfig config;
    config.n_eigenvectors = 2;
    config.tolerance = 1.0e-8;
    config.max_iterations = 500;
    config.verbose = true;

    auto backend = create_eigensolver_backend("slepc_validation");
    if (!backend) {
        std::fprintf(stderr, "ERROR: SLEPc backend not available.\n");
        return 1;
    }

    std::printf("  Backend: %s\n\n", backend->name().c_str());

    // Quick operator sanity check: A(ones) should NOT be zero
    {
        DeviceBuffer<real> x(grid.num_cells()), y(grid.num_cells());
        std::vector<real> hx(grid.num_cells(), 1.0);
        cudaMemcpy(x.data(), hx.data(), hx.size() * sizeof(real), cudaMemcpyHostToDevice);
        A.apply_A(x.span(), y.span(), cuda_ctx.cuda_stream());
        cudaStreamSynchronize(cuda_ctx.cuda_stream());
        blas::ReductionWorkspace ws2;
        double norm_y = blas::nrm2_host(cuda_ctx, DeviceSpan<const real>(y.data(), y.size()), ws2);
        double norm_x = blas::nrm2_host(cuda_ctx, DeviceSpan<const real>(x.data(), x.size()), ws2);
        std::printf("  Operator check: ||A(1)||/||1|| = %.6e\n", norm_y / norm_x);
    }

    std::vector<DeviceBuffer<real>> eigenvectors;
    EigensolverResult result = backend->solve(A, config, cuda_ctx, eigenvectors);

    // ------------------------------------------------------------------
    // 4. Post-solution analysis
    // ------------------------------------------------------------------
    std::printf("\n=== Post-Solution Analysis ===\n");
    std::printf("  Status: %s\n", result.message.c_str());

    // Compute ||D ψ_i|| for each eigenvector
    DeviceBuffer<real> d_Dpsi(grid.num_cells());
    blas::ReductionWorkspace ws;
    for (int i = 0; i < static_cast<int>(eigenvectors.size()); ++i) {
        DeviceSpan<const real> psi(eigenvectors[i].data(), eigenvectors[i].size());
        D.apply_D(psi, d_Dpsi.span(), cuda_ctx.cuda_stream());
        cudaStreamSynchronize(cuda_ctx.cuda_stream());

        double norm_Dpsi =
            blas::nrm2_host(cuda_ctx, DeviceSpan<const real>(d_Dpsi.data(), d_Dpsi.size()), ws);
        double norm_psi = blas::nrm2_host(
            cuda_ctx, DeviceSpan<const real>(eigenvectors[i].data(), eigenvectors[i].size()), ws);
        std::printf("  ψ[%d]:  λ = %14.8e   ||Dψ|| = %10.3e   ||Dψ||/||ψ|| = %10.3e\n", i,
                    result.eigenvalues[i], norm_Dpsi, norm_psi > 0 ? norm_Dpsi / norm_psi : 0.0);
    }

    // ------------------------------------------------------------------
    // 5. Ingest eigenvectors into PsptaInvariantField
    // ------------------------------------------------------------------
    if (eigenvectors.size() >= 2) {
        std::printf("\n=== Eigenvector → PsptaInvariantField ===\n");
        PsptaInvariantField inv;
        inv.resize(grid);
        inv.ingest_eigenvectors(eigenvectors[0], eigenvectors[1], result, A.mu(), backend->name(),
                                cuda_ctx, cuda_ctx.cuda_stream());

        const auto& info = inv.construction_info();
        std::printf("  method       = StrategyA\n");
        std::printf("  backend      = %s\n", info.eigensolver_backend.c_str());
        std::printf("  mu           = %.2e\n", info.mu);
        std::printf("  eigenvalues  =");
        for (double lam : info.eigenvalues)
            std::printf(" %.8e", lam);
        std::printf("\n");
        std::printf("  iterations   = %d\n", info.eigensolver_iterations);
        std::printf("  elapsed_ms   = %.1f\n", info.construction_time_ms);
        std::printf("  psi1 valid   = %s\n", inv.psi1_ptr() ? "yes" : "no");
        std::printf("  psi2 valid   = %s\n", inv.psi2_ptr() ? "yes" : "no");
    } else {
        std::printf("\n=== Eigenvector → PsptaInvariantField ===\n");
        std::printf("  SKIPPED: need >= 2 eigenvectors, got %zu\n", eigenvectors.size());
    }

    // ------------------------------------------------------------------
    // 6. Verdict
    // ------------------------------------------------------------------
    std::printf("\n=== Verdict ===\n");
    bool pass = true;
    if (!result.success) {
        std::printf("  FAIL: eigensolver did not converge.\n");
        pass = false;
    }
    for (int i = 0; i < static_cast<int>(result.eigenvalues.size()); ++i) {
        if (result.residual_norms[i] > 1.0e-5) {
            std::printf("  FAIL: residual[%d] = %.3e > 1e-5\n", i, result.residual_norms[i]);
            pass = false;
        }
    }
    if (pass) {
        std::printf("  PASS: All eigenpairs converged with small residuals.\n");
    }
    std::printf("  Elapsed: %.1f ms\n", result.elapsed_ms);

    return pass ? 0 : 1;
}

#else // !MACROFLOW3D_HAS_PETSC

#include <cstdio>
int main() {
    std::fprintf(stderr, "ERROR: This executable requires MACROFLOW3D_HAS_PETSC.\n"
                         "Rebuild with -DMACROFLOW3D_ENABLE_PETSC=ON.\n");
    return 1;
}

#endif // MACROFLOW3D_HAS_PETSC
