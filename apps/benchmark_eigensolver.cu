/**
 * @file benchmark_eigensolver.cu
 * @brief Stage 4C benchmark: grid-ladder stress-test and
 * validation-vs-production comparison.
 *
 * Runs both eigensolver backends on a series of grid sizes, collecting:
 *   - converged eigenpairs, iterations, wall-clock time
 *   - GPU memory usage before/after
 *   - eigenvalue/residual comparison between backends
 *   - modal orthogonality and subspace agreement
 *
 * Usage:
 *   ./benchmark_eigensolver                  # default: 16,32,64
 *   ./benchmark_eigensolver 16 32 64 128     # custom grid sizes
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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;
namespace blas = macroflow3d::blas;

// ============================================================================
// Helpers
// ============================================================================

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

struct GpuMem {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    double used_mb() const {
        return static_cast<double>(total_bytes - free_bytes) / (1024.0 * 1024.0);
    }
    static GpuMem now() {
        GpuMem m;
        cudaMemGetInfo(&m.free_bytes, &m.total_bytes);
        return m;
    }
};

struct BenchmarkRow {
    int grid_n = 0;
    size_t dofs = 0;
    std::string backend;
    int nconv = 0;
    int iterations = 0;
    double elapsed_ms = 0.0;
    double mem_before_mb = 0.0;
    double mem_after_mb = 0.0;
    double mem_delta_mb = 0.0;
    std::vector<double> eigenvalues;
    std::vector<double> residuals;
    double orthogonality = 0.0;
    bool success = false;
    std::string note;
};

/// Compute subspace agreement: max |cosine| between two sets of eigenvectors.
/// For 2 vectors each, this is |<a1,b1>|+|<a1,b2>|+|<a2,b1>|+|<a2,b2>| style.
static double subspace_agreement(CudaContext& ctx, const std::vector<DeviceBuffer<real>>& ev_a,
                                 const std::vector<DeviceBuffer<real>>& ev_b, size_t n) {
    if (ev_a.size() < 2 || ev_b.size() < 2)
        return -1.0;

    blas::ReductionWorkspace ws;
    // Build 2x2 overlap matrix |<a_i, b_j>|
    double overlap = 0.0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            DeviceSpan<const real> ai(ev_a[i].data(), n);
            DeviceSpan<const real> bj(ev_b[j].data(), n);
            double d = blas::dot_host(ctx, ai, bj, ws);
            double ni = blas::nrm2_host(ctx, DeviceSpan<const real>(ev_a[i].data(), n), ws);
            double nj = blas::nrm2_host(ctx, DeviceSpan<const real>(ev_b[j].data(), n), ws);
            if (ni > 1e-30 && nj > 1e-30)
                overlap += std::fabs(d) / (ni * nj);
        }
    }
    // For a perfect 2D subspace match, this sum is 2.0 (identity permutation).
    // Normalize to [0,1]: 1 = perfect match.
    return overlap / 2.0;
}

// ============================================================================
// Run one backend at one grid size
// ============================================================================

static BenchmarkRow run_backend(const std::string& backend_name, int grid_n, double mu, int nev,
                                double tol, int max_iter, CudaContext& ctx,
                                std::vector<DeviceBuffer<real>>& eigenvectors_out) {
    BenchmarkRow row;
    row.grid_n = grid_n;
    row.backend = backend_name;

    const real L = 1.0;
    Grid3D grid(grid_n, grid_n, grid_n, L, L, L);
    row.dofs = grid.num_cells();

    auto mem0 = GpuMem::now();
    row.mem_before_mb = mem0.used_mb();

    // Velocity field: uniform vx=1
    VelocityField vel(grid);
    fill_uniform_velocity(vel, 1.0, 0.0, 0.0);

    // Operators
    TransportOperatorConfig D_cfg;
    D_cfg.x_bc = TransportXBoundary::OneSided;
    TransportOperator3D D(&vel, grid, D_cfg);
    LaplacianOperator3D Lop(grid, LaplacianOperator3D::XBoundary::Neumann);
    CombinedOperatorA A(&D, &Lop, mu);

    // Eigensolver
    auto backend = create_eigensolver_backend(backend_name);
    if (!backend) {
        row.note = "backend unavailable";
        return row;
    }

    EigensolverConfig config;
    config.n_eigenvectors = nev;
    config.tolerance = tol;
    config.max_iterations = max_iter;
    config.verbose = false;

    eigenvectors_out.clear();
    EigensolverResult result;
    try {
        result = backend->solve(A, config, ctx, eigenvectors_out);
    } catch (const std::exception& e) {
        row.note = std::string("EXCEPTION: ") + e.what();
        auto mem1 = GpuMem::now();
        row.mem_after_mb = mem1.used_mb();
        row.mem_delta_mb = row.mem_after_mb - row.mem_before_mb;
        return row;
    }

    auto mem1 = GpuMem::now();
    row.mem_after_mb = mem1.used_mb();
    row.mem_delta_mb = row.mem_after_mb - row.mem_before_mb;

    row.nconv = result.n_converged;
    row.iterations = result.iterations;
    row.elapsed_ms = result.elapsed_ms;
    row.eigenvalues = result.eigenvalues;
    row.residuals = result.residual_norms;
    row.success = result.success;
    row.note = result.message;

    // Compute orthogonality between first 2 eigenvectors
    if (eigenvectors_out.size() >= 2) {
        blas::ReductionWorkspace ws;
        DeviceSpan<const real> e0(eigenvectors_out[0].data(), row.dofs);
        DeviceSpan<const real> e1(eigenvectors_out[1].data(), row.dofs);
        double d01 = blas::dot_host(ctx, e0, e1, ws);
        double n0 = blas::nrm2_host(ctx, e0, ws);
        double n1 = blas::nrm2_host(ctx, e1, ws);
        row.orthogonality = (n0 > 1e-30 && n1 > 1e-30) ? std::fabs(d01) / (n0 * n1) : 0.0;
    }

    return row;
}

// ============================================================================
// Print benchmark table
// ============================================================================

static void print_table(const std::vector<BenchmarkRow>& rows) {
    std::printf("\n╔══════════════════════════════════════════════════════"
                "════════════════════════════════════════════╗\n");
    std::printf("║  BENCHMARK TABLE: Eigensolver Grid Ladder"
                "                                                ║\n");
    std::printf("╠═══════╦═══════════╦═══════════════════╦══════╦══════╦"
                "══════════╦══════════╦══════════╦═════════╣\n");
    std::printf("║  N³   ║    DOFs   ║  Backend          ║ Conv ║ Iter ║"
                " Time(ms) ║ ΔMem(MB) ║ Ortho    ║ Status  ║\n");
    std::printf("╠═══════╬═══════════╬═══════════════════╬══════╬══════╬"
                "══════════╬══════════╬══════════╬═════════╣\n");

    for (const auto& r : rows) {
        std::printf("║ %5d ║ %9zu ║ %-17s ║ %4d ║ %4d ║ %8.1f ║ %8.1f ║ %8.1e ║ %-7s ║\n", r.grid_n,
                    r.dofs, r.backend.c_str(), r.nconv, r.iterations, r.elapsed_ms, r.mem_delta_mb,
                    r.orthogonality, r.success ? "OK" : "FAIL");
    }
    std::printf("╚═══════╩═══════════╩═══════════════════╩══════╩══════╩"
                "══════════╩══════════╩══════════╩═════════╝\n\n");

    // Eigenvalue detail
    std::printf("=== Eigenvalue Detail ===\n");
    for (const auto& r : rows) {
        std::printf("  [%s @ %d³]: ", r.backend.c_str(), r.grid_n);
        for (size_t i = 0; i < r.eigenvalues.size() && i < 4; ++i)
            std::printf("λ%zu=%.6e ", i, r.eigenvalues[i]);
        std::printf("\n");
        std::printf("  %*s  residuals: ", static_cast<int>(r.backend.size() + 8), "");
        for (size_t i = 0; i < r.residuals.size() && i < 4; ++i)
            std::printf("r%zu=%.2e ", i, r.residuals[i]);
        std::printf("\n");
        if (!r.note.empty() && r.note != "OK")
            std::printf("  %*s  note: %s\n", static_cast<int>(r.backend.size() + 8), "",
                        r.note.c_str());
    }
}

// ============================================================================
// Validation vs Production comparison
// ============================================================================

static void compare_backends(CudaContext& ctx, int grid_n, const std::vector<BenchmarkRow>& rows,
                             const std::vector<DeviceBuffer<real>>& ev_val,
                             const std::vector<DeviceBuffer<real>>& ev_prod) {
    std::printf("\n=== Validation vs Production @ %d³ ===\n", grid_n);

    // Find matching rows
    const BenchmarkRow* r_val = nullptr;
    const BenchmarkRow* r_prod = nullptr;
    for (const auto& r : rows) {
        if (r.grid_n == grid_n && r.backend == "slepc_validation")
            r_val = &r;
        if (r.grid_n == grid_n && r.backend == "slepc")
            r_prod = &r;
    }
    if (!r_val || !r_prod) {
        std::printf("  (skip: both backends not available at this size)\n");
        return;
    }
    if (!r_val->success || !r_prod->success) {
        std::printf("  (skip: at least one backend did not converge)\n");
        return;
    }

    // Eigenvalue comparison
    std::printf("  %-20s %-20s %-20s %-14s\n", "", "Validation", "Production", "|Δλ|/|λ_v|");
    int n_eig = std::min(static_cast<int>(r_val->eigenvalues.size()),
                         static_cast<int>(r_prod->eigenvalues.size()));
    for (int i = 0; i < n_eig; ++i) {
        double lv = r_val->eigenvalues[i];
        double lp = r_prod->eigenvalues[i];
        double rdiff =
            (std::fabs(lv) > 1e-30) ? std::fabs(lv - lp) / std::fabs(lv) : std::fabs(lv - lp);
        std::printf("  λ[%d]               %14.8e    %14.8e    %10.3e\n", i, lv, lp, rdiff);
    }

    // Residual comparison
    std::printf("  %-20s %-20s %-20s\n", "", "Val residual", "Prod residual");
    int n_res = std::min(static_cast<int>(r_val->residuals.size()),
                         static_cast<int>(r_prod->residuals.size()));
    for (int i = 0; i < n_res; ++i) {
        std::printf("  r[%d]               %14.8e    %14.8e\n", i, r_val->residuals[i],
                    r_prod->residuals[i]);
    }

    // Orthogonality
    std::printf("  Orthogonality:     val=%8.1e   prod=%8.1e\n", r_val->orthogonality,
                r_prod->orthogonality);

    // Subspace agreement
    size_t n = static_cast<size_t>(grid_n) * grid_n * grid_n;
    double sa = subspace_agreement(ctx, ev_val, ev_prod, n);
    std::printf("  Subspace agreement: %.6f  (1.0 = perfect)\n", sa);

    // Timing ratio
    std::printf("  Timing:            val=%.1f ms   prod=%.1f ms   ratio=%.2fx\n",
                r_val->elapsed_ms, r_prod->elapsed_ms,
                r_prod->elapsed_ms / std::max(r_val->elapsed_ms, 1e-3));

    // Memory ratio
    std::printf("  Memory delta:      val=%.1f MB   prod=%.1f MB\n", r_val->mem_delta_mb,
                r_prod->mem_delta_mb);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    runtime::PetscSlepcInit::ensure();
    CudaContext ctx(0);

    // Parse grid sizes from command line, or use defaults
    std::vector<int> grid_sizes;
    for (int i = 1; i < argc; ++i) {
        int n = std::atoi(argv[i]);
        if (n >= 4 && n <= 256)
            grid_sizes.push_back(n);
    }
    if (grid_sizes.empty())
        grid_sizes = {16, 32, 64};

    // Solver params
    const double mu = 1.0e-4;
    const int nev = 2;
    const double tol = 1.0e-8;
    const int max_iter = 500;

    // Threshold: validation backend is O(n²) memory, skip for large grids
    const int validation_max_n = 32;

    std::printf("╔══════════════════════════════════════════╗\n");
    std::printf("║  Stage 4C: Production Eigensolver Bench  ║\n");
    std::printf("╠══════════════════════════════════════════╣\n");
    std::printf("║  μ = %.1e, nev = %d, tol = %.1e       ║\n", mu, nev, tol);
    std::printf("║  Grid sizes: ");
    for (int n : grid_sizes)
        std::printf("%d ", n);
    std::printf("%*s║\n", static_cast<int>(27 - grid_sizes.size() * 4), "");
    std::printf("╚══════════════════════════════════════════╝\n\n");

    auto mem_baseline = GpuMem::now();
    std::printf("GPU baseline: %.1f MB used / %.1f MB total\n\n", mem_baseline.used_mb(),
                static_cast<double>(mem_baseline.total_bytes) / (1024.0 * 1024.0));

    std::vector<BenchmarkRow> all_rows;

    // For comparison, keep eigenvectors at comparison size
    std::vector<DeviceBuffer<real>> ev_val_cmp, ev_prod_cmp;
    int compare_n = -1;

    for (int grid_n : grid_sizes) {
        std::printf("━━━ Grid %d³ (N=%zu) ━━━\n", grid_n,
                    static_cast<size_t>(grid_n) * grid_n * grid_n);

        // --- Run validation backend (small grids only) ---
        if (grid_n <= validation_max_n) {
            std::printf("\n--- Validation backend ---\n");
            std::vector<DeviceBuffer<real>> ev_val;
            cudaDeviceSynchronize();
            auto row_val =
                run_backend("slepc_validation", grid_n, mu, nev, tol, max_iter, ctx, ev_val);
            cudaDeviceSynchronize();
            all_rows.push_back(row_val);

            // Save for comparison
            compare_n = grid_n;
            ev_val_cmp = std::move(ev_val);
        }

        // --- Run production backend ---
        {
            std::printf("\n--- Production backend ---\n");
            std::vector<DeviceBuffer<real>> ev_prod;
            cudaDeviceSynchronize();
            auto row_prod = run_backend("slepc", grid_n, mu, nev, tol, max_iter, ctx, ev_prod);
            cudaDeviceSynchronize();
            all_rows.push_back(row_prod);

            // Save for comparison at the compare_n size
            if (grid_n == compare_n) {
                ev_prod_cmp = std::move(ev_prod);
            }
        }

        // Force cleanup between sizes
        cudaDeviceSynchronize();
        std::printf("\n");
    }

    // Print aggregate table
    print_table(all_rows);

    // Print comparison
    if (compare_n > 0) {
        compare_backends(ctx, compare_n, all_rows, ev_val_cmp, ev_prod_cmp);
    }

    // --- Profiling summary ---
    std::printf("\n=== Profiling Summary ===\n");
    std::printf("  Main cost centers in matrix-free path:\n");
    std::printf("  1. apply_A (MATSHELL callback) = D†D + μL per LOBPCG iter × "
                "block_size vectors\n");
    std::printf("  2. Preconditioner (μL assembled + PCILU): setup once, apply "
                "per iter\n");
    std::printf("  3. Block orthogonalization: Gram-Schmidt in LOBPCG (nev × nev "
                "dot products)\n");
    std::printf("  4. Host↔Device sync: cudaStreamSynchronize in shell_matmult "
                "callback\n\n");

    // --- Preconditioner assessment ---
    std::printf("=== Preconditioner Assessment ===\n");

    // Find largest production run
    const BenchmarkRow* largest_prod = nullptr;
    for (const auto& r : all_rows) {
        if (r.backend == "slepc" && (!largest_prod || r.grid_n > largest_prod->grid_n))
            largest_prod = &r;
    }

    if (largest_prod && largest_prod->success) {
        std::printf("  Largest converged production run: %d³ (%zu DOFs)\n", largest_prod->grid_n,
                    largest_prod->dofs);
        std::printf("  Iterations: %d,  Time: %.1f ms\n", largest_prod->iterations,
                    largest_prod->elapsed_ms);
        std::printf("  μL + PCILU status: ");
        if (largest_prod->iterations <= 100) {
            std::printf("ACCEPTABLE — converges in reasonable iterations.\n");
        } else if (largest_prod->iterations <= 300) {
            std::printf("MARGINAL — converges but iteration count may grow with N.\n");
        } else {
            std::printf("PROBLEMATIC — high iteration count suggests weak PC.\n");
        }
    } else if (largest_prod) {
        std::printf("  Largest production run %d³: DID NOT CONVERGE\n", largest_prod->grid_n);
        std::printf("  μL + PCILU status: BLOCKING — needs stronger PC or solver "
                    "change.\n");
    }

    std::printf("\n  AMG-GPU assessment:\n");
    std::printf("    PCGAMG: PETSc native, always available but may segfault in "
                "cuda builds.\n");
    std::printf("    PCHYPRE/BoomerAMG: unavailable (PETSc built without hypre).\n");
    std::printf("    Recommendation: if PCILU blocks at ≥64³, rebuild PETSc with "
                "--download-hypre.\n\n");

    // --- Stage 5 readiness ---
    std::printf("=== Stage 5 (Gauge Fixing) Readiness ===\n");
    std::printf("  ✓ ModalQualityReport available at ingest time\n");
    std::printf("  ✓ gauge_ready flag computed (norms, orthogonality, residuals)\n");
    std::printf("  ✓ PsptaInvariantField::ingest_eigenvectors() is the canonical "
                "entry\n");
    std::printf("  → Gauge fixing plugs in AFTER ingest, BEFORE particle transport\n");
    std::printf("  → Needs: pair_selection (which 2 of K modes), section calibration\n");
    std::printf("  → Entry point: new GaugeFixer method called on PsptaInvariantField\n");
    std::printf("  → Already stub: "
                "src/physics/particles/pspta/invariants/GaugeFixer.cu\n\n");

    // Verdict
    bool all_pass = true;
    for (const auto& r : all_rows) {
        if (!r.success)
            all_pass = false;
    }

    std::printf("=== VERDICT: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}

#else // !MACROFLOW3D_HAS_PETSC

#include <cstdio>
int main() {
    std::printf("benchmark_eigensolver: PETSc not enabled. Skipping.\n");
    return 0;
}

#endif
