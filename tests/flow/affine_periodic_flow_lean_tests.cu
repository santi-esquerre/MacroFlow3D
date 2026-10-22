/**
 * @file affine_periodic_flow_lean_tests.cu
 * @brief SF-40 L2: `solve_affine_periodic_flow_lean` is bit-for-bit identical to
 *        `solve_affine_periodic_flow` (h_tilde, u, v, w, K_eff, G, every corrector
 *        ProjectedPCGResult field, symmetry/eigenvalue/flux/divergence diagnostics, inlet U
 *        plane) in both b_hat modes (pinned mapped host = level S+, device = level S); its S2
 *        memory report equals the closed form of the pre-registration (RR3) computed here
 *        independently from the geometry; and only h_tilde remains allocated on return.
 *
 * Fast mode (default, < 60 s locally): 32^3 eps 1 and 2, 64^3 eps 2, 64x16x16 eps 2.
 * `--full` (V100 only): 256^3 eps 2 and 512x64x64 eps 2.
 * Output: one `CHECK <name> PASS|FAIL` line per check, then `ALL_PASS` / `FAILED n`.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/sum.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using solvers::ProjectedPCGResult;

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

bool vec_bits_eq(const std::vector<real>& a, const std::vector<real>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0);
}

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

std::vector<real> download(CudaContext& ctx, const real* dev, std::size_t n) {
    ctx.synchronize(); // ctx-stream work must finish before a legacy-stream D2H
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), dev, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

std::size_t device_used() {
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    std::size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return t - f;
}

struct Problem {
    Grid3D grid;
    std::size_t n = 0;
    int levels = 1;
    std::vector<real> K;

    Problem(int nx, int ny, int nz, double eps) {
        const int nmin = std::min(nx, std::min(ny, nz));
        const double h = 1.0 / nmin; // isotropic spacing; domain (nx/nmin) x 1 x 1
        grid = Grid3D(nx, ny, nz, h, h, h);
        n = grid.num_cells();
        const double Lx = nx * h, Ly = ny * h, Lz = nz * h;
        K.resize(n);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double x = (i + 0.5) * h / Lx, y = (j + 0.5) * h / Ly,
                                 z = (k + 0.5) * h / Lz;
                    const double Y =
                        0.7 * std::sin(kTwoPi * x + 0.4) * std::cos(kTwoPi * y) +
                        0.5 * std::cos(kTwoPi * 2.0 * y + 1.1) * std::sin(kTwoPi * z + 0.2) +
                        0.3 * std::sin(kTwoPi * (x - y + 2.0 * z) + 0.9);
                    K[grid.idx(i, j, k)] = std::exp(eps * Y);
                }
        // slab_auto_mg_levels(N) loop applied to the smallest extent (valid for all extents).
        int m = nmin;
        while (m % 2 == 0 && m / 2 >= 4 && (m / 2) % 2 == 0) {
            m /= 2;
            ++levels;
        }
    }
};

AffinePeriodicFlowConfig make_config(int levels) {
    AffinePeriodicFlowConfig cfg;
    cfg.qbar[0] = 1.0;
    cfg.qbar[1] = 0.0;
    cfg.qbar[2] = 0.0;
    cfg.linear.rtol = 1e-10;
    cfg.mg.num_levels = levels;
    return cfg;
}

// Everything the reference solver produces, on the host.
struct Reference {
    AffinePeriodicFlowReport report;
    std::vector<real> h_tilde, u, v, w;
};

// Reference run in its own scope: all its device memory is released before returning.
Reference run_reference(CudaContext& ctx, const Problem& P, const AffinePeriodicFlowConfig& cfg) {
    DeviceBuffer<real> K(P.n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(K.data(), P.K.data(), P.n * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    const Grid3D& g = P.grid;
    const std::size_t nu = static_cast<std::size_t>(g.nx + 1) * g.ny * g.nz;
    const std::size_t nv = static_cast<std::size_t>(g.nx) * (g.ny + 1) * g.nz;
    const std::size_t nw = static_cast<std::size_t>(g.nx) * g.ny * (g.nz + 1);
    DeviceBuffer<real> u(nu), v(nv), w(nw);
    auto ws = std::make_unique<AffinePeriodicFlowWorkspace>();
    Reference R;
    R.report = solve_affine_periodic_flow(ctx, g, DeviceSpan<const real>(K.span()), cfg,
                                          AffinePeriodicVelocityView{u.span(), v.span(), w.span()},
                                          *ws);
    ctx.synchronize();
    R.h_tilde = download(ctx, ws->potential_fluctuation().data(), P.n);
    R.u = download(ctx, u.data(), nu);
    R.v = download(ctx, v.data(), nv);
    R.w = download(ctx, w.data(), nw);
    return R;
}

std::vector<real> reference_inlet_plane(const Problem& P, const std::vector<real>& U) {
    // Exactly the extraction of SlabProductionSetup.cu (build_production_stage), generalized to
    // ny x nz transverse extents.
    const int nx = P.grid.nx, ny = P.grid.ny, nz = P.grid.nz;
    std::vector<real> plane(static_cast<std::size_t>(ny) * nz, 0.0);
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
            plane[static_cast<std::size_t>(k) + static_cast<std::size_t>(nz) * j] =
                U[static_cast<std::size_t>(j) * (nx + 1) +
                  static_cast<std::size_t>(k) * (nx + 1) * ny];
    return plane;
}

// Independent closed form of the S2 items (C-MEMPHASE names and size rules) from the geometry.
std::vector<std::pair<std::string, std::size_t>> expected_items(const Problem& P, bool bhat_device) {
    const std::size_t S = sizeof(real);
    const Grid3D& g = P.grid;
    const std::size_t nu = static_cast<std::size_t>(g.nx + 1) * g.ny * g.nz;
    const std::size_t nv = static_cast<std::size_t>(g.nx) * (g.ny + 1) * g.nz;
    const std::size_t nw = static_cast<std::size_t>(g.nx) * g.ny * (g.nz + 1);
    const std::size_t m = std::max(nu, std::max(nv, nw));
    std::vector<std::size_t> nl;
    for (int l = 0; l < P.levels; ++l) {
        nl.push_back(static_cast<std::size_t>(g.nx >> l) * (g.ny >> l) * (g.nz >> l));
    }
    std::vector<std::pair<std::string, std::size_t>> it;
    for (int l = 0; l < P.levels; ++l) {
        const std::string p = "mg_l" + std::to_string(l) + "_";
        it.push_back({p + "x", S * (l == 0 ? m : nl[l])});
        it.push_back({p + "b", S * nl[l]});
        it.push_back({p + "r", S * nl[l]});
        it.push_back({p + "coef", S * nl[l]});
    }
    it.push_back({"pcg_r", S * m});
    it.push_back({"pcg_p", S * m});
    it.push_back({"corrector", S * m});
    if (bhat_device) it.push_back({"bhat_device", S * P.n});
    it.push_back({"scratch_pcg_d_scalar", S});
    it.push_back({"scratch_pcg_mean_zero", blas::sum_workspace_bytes(P.n) + S});
    it.push_back({"scratch_rhs_mean_zero", blas::sum_workspace_bytes(P.n) + S});
    it.push_back({"scratch_rhs_means", 4 * S});
    for (int l = 0; l < P.levels; ++l) {
        it.push_back({"scratch_mg_mean_zero_l" + std::to_string(l),
                      blas::sum_workspace_bytes(nl[l]) + S});
    }
    it.push_back({"scratch_reduce_blocks", 256 * S});
    return it;
}

// RR3: 8 [7n + 4 sum_{l>=1} n_l] + 8*4*(m - n) + S_scr (+ 8n with b_hat on device).
std::size_t closed_form_total(const Problem& P, bool bhat_device) {
    const Grid3D& g = P.grid;
    const std::size_t nu = static_cast<std::size_t>(g.nx + 1) * g.ny * g.nz;
    const std::size_t nv = static_cast<std::size_t>(g.nx) * (g.ny + 1) * g.nz;
    const std::size_t nw = static_cast<std::size_t>(g.nx) * g.ny * (g.nz + 1);
    const std::size_t m = std::max(nu, std::max(nv, nw));
    std::size_t coarse = 0, scr = 8 + 2 * (blas::sum_workspace_bytes(P.n) + 8) + 32 + 2048;
    for (int l = 0; l < P.levels; ++l) {
        const std::size_t nl = static_cast<std::size_t>(g.nx >> l) * (g.ny >> l) * (g.nz >> l);
        scr += blas::sum_workspace_bytes(nl) + 8;
        if (l >= 1) coarse += nl;
    }
    return 8 * (7 * P.n + 4 * coarse) + 32 * (m - P.n) + scr + (bhat_device ? 8 * P.n : 0);
}

void run_case(CudaContext& ctx, const std::string& name, int nx, int ny, int nz, double eps) {
    const Problem P(nx, ny, nz, eps);
    const AffinePeriodicFlowConfig cfg = make_config(P.levels);
    const Reference R = run_reference(ctx, P, cfg);
    std::printf("INFO %s levels=%d n=%zu iters=%d,%d,%d status=%d,%d,%d G=%.17g,%.17g,%.17g\n",
                name.c_str(), P.levels, P.n, R.report.corrector_results[0].iterations,
                R.report.corrector_results[1].iterations, R.report.corrector_results[2].iterations,
                static_cast<int>(R.report.corrector_results[0].status),
                static_cast<int>(R.report.corrector_results[1].status),
                static_cast<int>(R.report.corrector_results[2].status), R.report.G[0],
                R.report.G[1], R.report.G[2]);
    check(R.report.corrector_results[0].converged && R.report.corrector_results[1].converged &&
              R.report.corrector_results[2].converged,
          name + " reference_converged");
    const std::vector<real> ref_plane = reference_inlet_plane(P, R.u);

    for (int mode = 0; mode < 2; ++mode) {
        const bool mapped = mode == 0;
        const std::string tag = name + (mapped ? "/b_hat=mapped" : "/b_hat=device");
        AffinePeriodicFlowLeanOptions opt;
        opt.bhat_mapped_host = mapped;
        opt.keep_face_fluxes_host = true;

        const std::size_t used_before = device_used();
        AffinePeriodicFlowLeanResult L = solve_affine_periodic_flow_lean(ctx, P.grid, P.K, cfg, opt);
        const std::size_t used_after = device_used();
        const AffinePeriodicFlowReport& a = R.report;
        const AffinePeriodicFlowReport& b = L.report;

        // ---- bitwise equality with the current solver ------------------------------------
        check(vec_bits_eq(R.h_tilde, download(ctx, L.h_tilde.data(), P.n)) && L.h_tilde.size() == P.n,
              tag + " h_tilde_bitwise");
        check(vec_bits_eq(R.u, L.face_u), tag + " face_u_bitwise");
        check(vec_bits_eq(R.v, L.face_v), tag + " face_v_bitwise");
        check(vec_bits_eq(R.w, L.face_w), tag + " face_w_bitwise");
        check(vec_bits_eq(ref_plane, L.inlet_u_plane), tag + " inlet_u_plane_bitwise");
        bool keff = true, g3 = true, eig = true, mf = true;
        for (int i = 0; i < 3; ++i) {
            g3 = g3 && bits_eq(a.G[i], b.G[i]);
            eig = eig && bits_eq(a.eigenvalues_symmetric_part[i], b.eigenvalues_symmetric_part[i]);
            mf = mf && bits_eq(a.achieved_mean_flux[i], b.achieved_mean_flux[i]);
            for (int j = 0; j < 3; ++j) keff = keff && bits_eq(a.K_eff[i][j], b.K_eff[i][j]);
        }
        check(keff, tag + " K_eff_bitwise");
        check(g3, tag + " G_bitwise");
        check(eig, tag + " eigenvalues_bitwise");
        check(mf, tag + " achieved_mean_flux_bitwise");
        check(bits_eq(a.symmetry_defect_rel, b.symmetry_defect_rel), tag + " symmetry_defect_bitwise");
        check(bits_eq(a.div_max_abs, b.div_max_abs), tag + " div_max_abs_bitwise");
        check(bits_eq(a.div_rms, b.div_rms), tag + " div_rms_bitwise");
        for (int d = 0; d < 3; ++d) {
            const std::string diff = result_diff(a.corrector_results[d], b.corrector_results[d]);
            check(diff.empty(), tag + " corrector" + std::to_string(d) + "_result_bitwise" +
                                    (diff.empty() ? "" : " (" + diff + ")"));
        }

        // ---- exact memory report ------------------------------------------------------------
        const AffinePeriodicFlowLeanMemoryReport& M = L.memory;
        const auto exp = expected_items(P, !mapped);
        bool items_ok = M.items.size() == exp.size();
        std::string first_bad;
        for (std::size_t i = 0; items_ok && i < exp.size(); ++i) {
            if (M.items[i].name != exp[i].first || M.items[i].bytes != exp[i].second) {
                items_ok = false;
                first_bad = " (item " + M.items[i].name + "=" + std::to_string(M.items[i].bytes) +
                            " expected " + exp[i].first + "=" + std::to_string(exp[i].second) + ")";
            }
        }
        check(items_ok, tag + " memory_items_equal_geometric_sizes" + first_bad);
        std::size_t sum = 0;
        for (const auto& it : M.items) sum += it.bytes;
        const std::size_t closed = closed_form_total(P, !mapped);
        check(sum == M.total_device_bytes && M.total_device_bytes == closed,
              tag + " memory_total_equals_closed_form");
        std::printf("INFO %s total_device_bytes=%zu closed_form=%zu diff=%lld device_used_at_peak=%zu "
                    "face_max=%zu bhat_host_bytes=%zu pcg_red_temp=%zu seconds=%.3f\n",
                    tag.c_str(), M.total_device_bytes, closed,
                    static_cast<long long>(M.total_device_bytes) - static_cast<long long>(closed),
                    M.device_used_at_peak, M.face_max, M.bhat_host_bytes, M.pcg_red_temp_bytes,
                    M.seconds);
        check(M.pcg_red_temp_bytes == 0, tag + " pcg_reduction_workspace_has_no_temp");
        check(M.bhat_host_bytes == (mapped ? 8 * P.n : 0), tag + " bhat_host_bytes");
        const AffinePeriodicFlowMemoryReport& C = M.categories;
        check(C.correctors_bytes + C.rhs_scratch_bytes + C.flux_scratch_bytes +
                      C.reduction_scratch_bytes + C.affine_rhs_workspace_bytes +
                      C.pcg_workspace_bytes + C.mg_hierarchy_bytes + C.mg_preconditioner_bytes ==
                  C.total_bytes && C.total_bytes == M.total_device_bytes,
              tag + " memory_categories_sum_to_total");
        check(M.seconds > 0.0 && M.device_used_at_peak > 0, tag + " memory_report_timing_and_peak");

        // ---- after the call only h_tilde (capacity m) remains allocated ----------------------------
        const long long delta = static_cast<long long>(used_after) - static_cast<long long>(used_before);
        const long long expected = static_cast<long long>(8 * M.face_max);
        std::printf("INFO %s residual_after_call=%lld expected_8m=%lld (diff %lld)\n", tag.c_str(),
                    delta, expected, delta - expected);
        check(L.h_tilde.capacity() == M.face_max, tag + " h_tilde_capacity_is_m");
        check(std::llabs(delta - expected) <= 8LL * 1024 * 1024, tag + " residual_allocation_is_8m");
    }
}

void validation_cases(CudaContext& ctx) {
    const Problem P(16, 16, 16, 1.0);
    const AffinePeriodicFlowConfig cfg = make_config(P.levels);
    const AffinePeriodicFlowLeanOptions opt;
    {
        bool threw = false;
        std::vector<real> bad(P.n - 1, 1.0);
        try {
            (void)solve_affine_periodic_flow_lean(ctx, P.grid, bad, cfg, opt);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validation K_size_mismatch_throws");
    }
    {
        bool threw = false;
        try {
            (void)solve_affine_periodic_flow_lean(ctx, Grid3D(16, 16, 16, 1.0, 1.0, 2.0), P.K, cfg,
                                                  opt);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "validation anisotropic_spacing_throws");
    }
}

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) full = true;
    }
    // No cudaSetDeviceFlags here: the library relies on UVA for the mapped b_hat.
    CudaContext ctx(0);

    // Warm-up (loads kernels / cuBLAS / CUB state) so the residual-allocation deltas below
    // measure the lean call, not lazy module loading.
    {
        const Problem W(16, 16, 16, 1.0);
        (void)run_reference(ctx, W, make_config(W.levels));
        AffinePeriodicFlowLeanOptions o;
        o.keep_face_fluxes_host = true;
        (void)solve_affine_periodic_flow_lean(ctx, W.grid, W.K, make_config(W.levels), o);
    }

    if (full) {
        run_case(ctx, "256x256x256_eps2", 256, 256, 256, 2.0);
        run_case(ctx, "512x64x64_eps2", 512, 64, 64, 2.0);
    } else {
        run_case(ctx, "32x32x32_eps1", 32, 32, 32, 1.0);
        run_case(ctx, "32x32x32_eps2", 32, 32, 32, 2.0);
        run_case(ctx, "64x64x64_eps2", 64, 64, 64, 2.0);
        run_case(ctx, "64x16x16_eps2", 64, 16, 16, 2.0);
    }
    validation_cases(ctx);

    if (g_failures == 0) {
        std::printf("ALL_PASS\n");
        return 0;
    }
    std::printf("FAILED %d\n", g_failures);
    return 1;
}
