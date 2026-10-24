/**
 * @file dirichlet_medium_tests.cu
 * @brief SF-38 N2: DirichletMedium (medium (b), contract C-DIRICHLET) and EulerianFaceStats (E1, RR5).
 *
 *  1. k = 1 (Y = 0), 64x16x16, G1 = 1: converged; every U = 1 (<= 1e-8), every V, W <= 1e-8; balance
 *     <= 1e-10; E1 interior var 0 (<= 1e-14), qbar = 1, thetas 0.
 *  2. Layered Y(x1) = 0.5 cos(2 pi x1 / L1), eps = 1: every U equals the exact discrete 1-D flux of
 *     solve_head's scheme (derivation below) to <= 1e-8 relative; V, W ~ 0.
 *  3. Periodic repack: the device periodic arrays equal the CompactMAC arrays without the outlet plane
 *     (bitwise), on a smooth 3-D field.
 *  4. Memory: the FlowWorkspace is freed before the face arrays are allocated (phase records and
 *     cudaMemGetInfo deltas), at 512x32x32.
 *  5. E1 on a hand-built CompactMAC array with known per-plane mean/variance (exact); input checks.
 *  6. build_dirichlet_medium input validation.
 *  --full (V100 mode): smooth field, eps 2, 256x64x64, auto MG levels: rel <= 1e-10, balance <= 1e-8.
 *
 * Exact discrete 1-D flux (layered K_i, cell i = 0..nx-1, x only). solve_head's scheme: interior face
 * flux U = -Kh (H_r - H_l)/h with Kh = 2 Ka Kb/(Ka + Kb); west Dirichlet face
 * U = -K_0 (H_0 - H_west)/(h/2) (velocity_from_head.cu kernel_U_west; the operator's boundary term
 * 2 K_0/h^2 is the same half cell), east face U = -K_{nx-1} (H_east - H_{nx-1})/(h/2). The 1-D discrete
 * problem is a series of resistances, flux q the same on every face:
 *   q = (H_west - H_east) / R,  H_west - H_east = G1 L1,
 *   R = (h/2)/K_0 + sum_{i=0}^{nx-2} h / Kh(K_i, K_{i+1}) + (h/2)/K_{nx-1}.
 *
 * Usage: dirichlet_medium_tests [--full]. Prints `CHECK <name> PASS|FAIL` and `ALL_PASS` / `FAILED n`.
 * Compiled with --fmad=false like the other lagrangian tests.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/particles/lagrangian_labels/DirichletMedium.cuh"
#include "src/physics/particles/lagrangian_labels/EulerianFaceStats.hpp"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
int g_failed = 0;

void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("CHECK %s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
                detail.c_str());
    std::fflush(stdout);
    if (!ok)
        ++g_failed;
}

std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

/// Deepest hierarchy by the slab_auto_mg_levels loop condition on all three extents.
int auto_levels(int nx, int ny, int nz) {
    auto ok = [](int n) { return n % 2 == 0 && n / 2 >= 4 && (n / 2) % 2 == 0; };
    int l = 1;
    while (ok(nx) && ok(ny) && ok(nz)) {
        nx /= 2;
        ny /= 2;
        nz /= 2;
        ++l;
    }
    return l;
}

std::vector<real> smooth_field(int nx, int ny, int nz) {
    const double h = 1.0 / ny, L1 = nx * h;
    std::vector<real> Y((size_t)nx * ny * nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = (i + 0.5) * h, y = (j + 0.5) * h, z = (k + 0.5) * h;
                Y[i + (size_t)nx * (j + (size_t)ny * k)] =
                    0.6 * std::sin(kTwoPi * y) * std::cos(kTwoPi * z) +
                    0.4 * std::sin(kTwoPi * (3.0 * x / L1 + y)) + 0.3 * std::cos(kTwoPi * (z + 2.0 * x / L1));
            }
    return Y;
}

ll::DirichletMediumSpec make_spec(int nx, int ny, int nz, double eps, double rtol) {
    ll::DirichletMediumSpec s;
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    s.eps = eps;
    s.G1 = 1.0;
    s.mg_levels = auto_levels(nx, ny, nz);
    s.cg_rtol = rtol;
    s.cg_max_iter = 2000;
    s.keep_host_faces = true;
    s.build_periodic_faces = true;
    return s;
}

std::size_t used_now() {
    std::size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return t - f;
}

void test_k1(CudaContext& ctx) {
    const int nx = 64, ny = 16, nz = 16;
    const std::vector<real> Y((size_t)nx * ny * nz, 0.0);
    const auto M = ll::build_dirichlet_medium(ctx, make_spec(nx, ny, nz, 1.0, 1e-12), Y);
    const auto& R = M.report;
    check("k1_converged", R.converged,
          "iters=" + std::to_string(R.iterations) + " r0=" + fmt("%.3e", R.initial_residual) +
              " rf=" + fmt("%.3e", R.final_residual));
    double eu = 0, ev = 0, ew = 0;
    for (double u : M.face_u)
        eu = std::max(eu, std::abs(u - 1.0));
    for (double v : M.face_v)
        ev = std::max(ev, std::abs(v));
    for (double w : M.face_w)
        ew = std::max(ew, std::abs(w));
    check("k1_U_is_1", eu <= 1e-8, "max|U-1|=" + fmt("%.3e", eu));
    check("k1_V_W_zero", ev <= 1e-8 && ew <= 1e-8, "max|V|=" + fmt("%.3e", ev) + " max|W|=" + fmt("%.3e", ew));
    check("k1_balance", R.balance <= 1e-10, "balance=" + fmt("%.3e", R.balance) +
                                                 " qbar=" + fmt("%.12f", R.qbar) +
                                                 " max_div=" + fmt("%.3e", R.max_div));
    const auto E = ll::eulerian_face_stats(M.face_u, M.face_v, M.face_w, nx, ny, nz, M.L1, 0.25);
    check("k1_E1", E.V1 <= 1e-14 && std::abs(E.qbar - 1.0) <= 1e-8 && std::abs(E.theta_2) <= 1e-8 &&
                       std::abs(E.theta_3) <= 1e-8 && E.var_v2 <= 1e-14 && E.var_v3 <= 1e-14,
          "V1=" + fmt("%.3e", E.V1) + " qbar-1=" + fmt("%.3e", E.qbar - 1.0) +
              " th2=" + fmt("%.3e", E.theta_2) + " th3=" + fmt("%.3e", E.theta_3));
}

void test_layered(CudaContext& ctx) {
    const int nx = 64, ny = 16, nz = 16;
    const double h = 1.0 / ny, L1 = nx * h;
    std::vector<real> Y((size_t)nx * ny * nz);
    std::vector<double> K1(nx);
    for (int i = 0; i < nx; ++i) {
        const double y = 0.5 * std::cos(kTwoPi * (i + 0.5) * h / L1);
        K1[i] = std::exp(y);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                Y[i + (size_t)nx * (j + (size_t)ny * k)] = y;
    }
    double R = 0.5 * h / K1[0] + 0.5 * h / K1[nx - 1];
    for (int i = 0; i + 1 < nx; ++i)
        R += h / (2.0 * K1[i] * K1[i + 1] / (K1[i] + K1[i + 1]));
    const double q = 1.0 * L1 / R; // G1 = 1
    const auto M = ll::build_dirichlet_medium(ctx, make_spec(nx, ny, nz, 1.0, 1e-12), Y);
    double eu = 0, ev = 0, ew = 0;
    for (double u : M.face_u)
        eu = std::max(eu, std::abs(u - q) / q);
    for (double v : M.face_v)
        ev = std::max(ev, std::abs(v));
    for (double w : M.face_w)
        ew = std::max(ew, std::abs(w));
    check("layered_converged", M.report.converged, "rel=" + fmt("%.3e", M.report.rel));
    check("layered_U_exact_1d", eu <= 1e-8, "q=" + fmt("%.12f", q) + " max rel err=" + fmt("%.3e", eu));
    check("layered_V_W_zero", ev <= 1e-8 * q && ew <= 1e-8 * q,
          "max|V|=" + fmt("%.3e", ev) + " max|W|=" + fmt("%.3e", ew));
    check("layered_qbar", std::abs(M.report.qbar - q) <= 1e-8 * q, "qbar=" + fmt("%.12f", M.report.qbar));
}

void test_repack(CudaContext& ctx) {
    const int nx = 64, ny = 16, nz = 16;
    const auto M = ll::build_dirichlet_medium(ctx, make_spec(nx, ny, nz, 1.0, 1e-10), smooth_field(nx, ny, nz));
    check("repack_available", M.periodic_available() && M.report.converged);
    const auto V = M.periodic_view();
    const size_t n = (size_t)nx * ny * nz;
    const auto pu = [&] { std::vector<real> a(n); MACROFLOW3D_CUDA_CHECK(cudaMemcpy(a.data(), V.u, n * 8, cudaMemcpyDeviceToHost)); return a; }();
    const auto pv = [&] { std::vector<real> a(n); MACROFLOW3D_CUDA_CHECK(cudaMemcpy(a.data(), V.v, n * 8, cudaMemcpyDeviceToHost)); return a; }();
    const auto pw = [&] { std::vector<real> a(n); MACROFLOW3D_CUDA_CHECK(cudaMemcpy(a.data(), V.w, n * 8, cudaMemcpyDeviceToHost)); return a; }();
    bool same = true;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t p = i + (size_t)nx * (j + (size_t)ny * k);
                same = same && std::memcmp(&pu[p], &M.face_u[i + (size_t)(nx + 1) * (j + (size_t)ny * k)], 8) == 0 &&
                       std::memcmp(&pv[p], &M.face_v[i + (size_t)nx * (j + (size_t)(ny + 1) * k)], 8) == 0 &&
                       std::memcmp(&pw[p], &M.face_w[p], 8) == 0;
            }
    check("repack_bitwise", same);
    check("repack_view_geometry", V.nx == nx && V.ny == ny && V.nz == nz && V.dx == M.h &&
                                      V.Lx == nx * M.h && V.Ly == 1.0 && V.Lz == 1.0);
}

void test_memory(CudaContext& ctx) {
    const int nx = 512, ny = 32, nz = 32;
    const size_t n = (size_t)nx * ny * nz;
    ll::DirichletMediumSpec s = make_spec(nx, ny, nz, 1.0, 1e-8);
    s.keep_host_faces = false;
    const std::vector<real> Y = smooth_field(nx, ny, nz);
    const size_t base = used_now();
    const auto M = ll::build_dirichlet_medium(ctx, s, Y);
    const size_t after = used_now();
    const ll::DirichletPhaseRecord *pK = nullptr, *pS = nullptr, *pR = nullptr, *pV = nullptr, *pF = nullptr;
    for (const auto& p : M.phases) {
        if (p.phase == "K") pK = &p;
        if (p.phase == "solve") pS = &p;
        if (p.phase == "release") pR = &p;
        if (p.phase == "velocity") pV = &p;
        if (p.phase == "faces") pF = &p;
    }
    check("mem_phases", pK && pS && pR && pV && pF && M.phases.size() == 5);
    if (!(pK && pS && pR && pV && pF))
        return;
    size_t cells_levels = 0;
    for (int l = 0; l < s.mg_levels; ++l)
        cells_levels += (size_t)(nx >> l) * (ny >> l) * (nz >> l);
    const size_t ws_expect = (n + 4 * cells_levels + 4 * n) * sizeof(real);
    check("mem_workspace_bytes_exact", M.report.workspace_bytes == ws_expect,
          "ws=" + std::to_string(M.report.workspace_bytes) + " expect=" + std::to_string(ws_expect) +
              " (" + fmt("%.2f", (double)ws_expect / n) + " B/cell)");
    const size_t MiB = 1u << 20;
    const size_t solve_peak = pS->device_used, rel_after = pR->device_used;
    check("mem_workspace_freed", solve_peak >= rel_after + ws_expect - 4 * MiB,
          "solve=" + std::to_string(solve_peak) + " after_release=" + std::to_string(rel_after) +
              " ws=" + std::to_string(ws_expect));
    check("mem_back_to_K_head", rel_after <= pK->device_used + 16 * MiB,
          "after_release=" + std::to_string(rel_after) + " K_phase=" + std::to_string(pK->device_used));
    check("mem_faces_after_release", pV->device_used_start <= rel_after + 4 * MiB,
          "velocity start=" + std::to_string(pV->device_used_start) + " after_release=" + std::to_string(rel_after));
    const size_t compact = ((size_t)(nx + 1) * ny * nz + (size_t)nx * (ny + 1) * nz + (size_t)nx * ny * (nz + 1)) * 8;
    check("mem_faces_phase_exact", pF->exact_bytes() == compact + 3 * n * sizeof(real),
          "faces exact=" + std::to_string(pF->exact_bytes()) + " B/cell=" + fmt("%.3f", (double)pF->exact_bytes() / n));
    check("mem_final_only_periodic", M.device_bytes() == 3 * n * sizeof(real) &&
                                         after <= base + 3 * n * sizeof(real) + 16 * MiB,
          "after-base=" + std::to_string((long long)after - (long long)base) +
              " periodic=" + std::to_string(M.device_bytes()));
    for (const auto& l : M.mem_lines())
        std::printf("%s\n", l.c_str());
}

void test_e1_hand() {
    const int nx = 8, ny = 2, nz = 2;
    const double L1 = 4.0; // h = 0.5
    const double sgn[4] = {-1, 1, 1, -1}; // zero mean over a plane, face index m = j + 2k
    std::vector<real> U((nx + 1) * ny * nz), V(nx * (ny + 1) * nz, 1e6), W(nx * ny * (nz + 1), 1e6);
    for (int i = 0; i <= nx; ++i)
        for (int m = 0; m < 4; ++m)
            U[i + (size_t)(nx + 1) * m] = (1.0 + i) + (i / 8.0) * sgn[m];
    for (int i = 0; i < nx; ++i)
        for (int m = 0; m < 4; ++m) {
            const int j = m % ny, k = m / ny;
            V[i + (size_t)nx * (j + (size_t)(ny + 1) * k)] = 0.25 * i + 0.5 * sgn[m]; // duplicate j = ny stays 1e6
            W[i + (size_t)nx * m] = -0.125 * i + (0.125 * i) * sgn[m];                // duplicate k = nz stays 1e6
        }
    const auto E = ll::eulerian_face_stats(U, V, W, nx, ny, nz, L1, 0.2); // interior [1, 3]
    bool prof = true;
    for (int i = 0; i <= nx; ++i)
        prof = prof && E.u_mean[i] == 1.0 + i && E.u_var[i] == (double)i * i / 64.0;
    for (int i = 0; i < nx; ++i)
        prof = prof && E.v_mean[i] == 0.25 * i && E.v_var[i] == 0.25 && E.w_mean[i] == -0.125 * i &&
               E.w_var[i] == (0.125 * i) * (0.125 * i);
    check("e1_profiles_exact", prof);
    check("e1_interior_ranges", E.plane_lo == 2 && E.plane_hi == 6 && E.layer_lo == 2 && E.layer_hi == 5 &&
                                    E.qbar == 1.0);
    const double V1 = (4 + 9 + 16 + 25 + 36) / 64.0 / 5.0;
    const double th2 = (0.5 + 0.75 + 1.0 + 1.25) / 4.0, th3 = -(0.25 + 0.375 + 0.5 + 0.625) / 4.0;
    check("e1_primary", std::abs(E.V1 - V1) <= 1e-15 && std::abs(E.theta_2 - th2) <= 1e-15 &&
                            std::abs(E.theta_3 - th3) <= 1e-15 && std::abs(E.var_v2 - 0.25) <= 1e-15 &&
                            std::abs(E.var_v3 - (0.125 * 0.125) * (4 + 9 + 16 + 25) / 4.0) <= 1e-15,
          "V1=" + fmt("%.16g", E.V1) + " vs " + fmt("%.16g", V1) + " th2=" + fmt("%.16g", E.theta_2) + " vs " +
              fmt("%.16g", th2) + " th3=" + fmt("%.16g", E.theta_3) + " vs " + fmt("%.16g", th3));
    auto throws = [&](auto f) {
        try {
            f();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check("e1_throws", throws([&] { ll::eulerian_face_stats(U, V, W, nx, ny, nz, L1, 0.45); }) && // empty interior
                           throws([&] { ll::eulerian_face_stats(V, V, W, nx, ny, nz, L1, 0.2); }) &&
                           throws([&] { ll::eulerian_face_stats(U, V, W, nx, ny, nz, L1, 0.0); }));
}

void test_inputs(CudaContext& ctx) {
    const std::vector<real> Y(64 * 16 * 16, 0.0);
    auto throws = [&](ll::DirichletMediumSpec s, const std::vector<real>& y) {
        try {
            ll::build_dirichlet_medium(ctx, s, y);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    ll::DirichletMediumSpec s = make_spec(64, 16, 16, 1.0, 1e-8);
    auto bad = s;
    bad.nz = 8;
    const bool a = throws(bad, Y);
    bad = s;
    bad.mg_levels = 0;
    const bool b = throws(bad, Y);
    bad = s;
    bad.G1 = 0.0;
    const bool c = throws(bad, Y);
    bad = s;
    bad.mg_levels = 5; // 16 / 16 = 1 < 2
    const bool d = throws(bad, Y);
    const bool e = throws(s, std::vector<real>(10, 0.0));
    check("inputs_invalid_argument", a && b && c && d && e);
}

void test_full(CudaContext& ctx) {
    const int nx = 256, ny = 64, nz = 64;
    ll::DirichletMediumSpec s = make_spec(nx, ny, nz, 2.0, 1e-10);
    s.keep_host_faces = false;
    const auto M = ll::build_dirichlet_medium(ctx, s, smooth_field(nx, ny, nz));
    const auto& R = M.report;
    check("full_converged", R.converged && R.rel <= 1e-10,
          "levels=" + std::to_string(s.mg_levels) + " iters=" + std::to_string(R.iterations) +
              " rel=" + fmt("%.3e", R.rel));
    check("full_balance", R.balance <= 1e-8, "balance=" + fmt("%.3e", R.balance) + " max_div=" +
                                                  fmt("%.3e", R.max_div) + " qbar=" + fmt("%.6f", R.qbar));
}

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else {
            std::fprintf(stderr, "usage: %s [--full]\n", argv[0]);
            return 2;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    CudaContext ctx(0);
    test_e1_hand();
    test_inputs(ctx);
    test_k1(ctx);
    test_layered(ctx);
    test_repack(ctx);
    test_memory(ctx);
    if (full)
        test_full(ctx);
    std::printf("total_seconds=%.2f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::printf(g_failed == 0 ? "ALL_PASS\n" : "FAILED %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
