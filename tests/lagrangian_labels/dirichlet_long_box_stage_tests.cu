/**
 * @file dirichlet_long_box_stage_tests.cu
 * @brief SF-41 N2: DirichletLongBoxStage (lean medium (b) stage, contract C-STAGE flow part), the external-vector
 *        non-projected PCG, and DirichletMedium's opt-in `keep_host_head`.
 *
 * Fast tier (default; every box has <= 16384 cells, i.e. <= 32^3 = 32768):
 *  1. pcg_solve_external == pcg_solve bitwise on a diagonal SPD system (z == Ap aliased), alias guard.
 *  2. D fast: SF-18 Gaussian Y (unit variance, normalize_variance, eps 2, ell 0.125, seed 4001) on 64x16x16 and
 *     32x16x16: lean stage vs DirichletMedium(keep_host_head): head memcmp, B_SOLVER text, device head and
 *     mapped-host head; f = ((-H) - G1 x) + G1 L1 reproduced bitwise on the host.
 *  3. C5 fast: C1 (Y = 0, G1 = 1, 64x16x16: initial residual exactly 0, rel := 0, balance <= 1e-8) and the C2
 *     layered medium (ln k = 0.5 cos(2 pi 0.6 x1 + 0.3), eps = 1) on 32x16x16 and 64x16x16: rel <= 1e-10,
 *     balance <= 1e-8; max relative difference to the exact discrete 1-D head recorded.
 *  4. Exact bytes: S2 == 8 [ 7 n + 4 sum_{l>=1} n_l ] + scratch (6 n with the head in mapped host memory),
 *     scratch = reduction scalar + CUB temporary; every phase's exact bytes <= S2's.
 *  5. Flag off: DirichletMedium with keep_host_head = false returns exactly what it returned with it on
 *     (report, phase bytes, faces) and no head.
 *  6. Input validation.
 *
 * Modes (V100, run by the orchestrator): `--only D` (RR6: 128x32x32, 512x64x64 with G1 = 1 and 3328x256x256 with
 * G1 = G[0] of build_long_box_stage on the same Y; the mapped-host head at the two small sizes) and `--only C5`
 * (RR5: C1 on 64x16x16 and 128x16x16; C2 on 64x32x32 and 128x32x32).
 *
 * Exact discrete 1-D head (layered K_i, i = 0..nx-1; solve_head's scheme with Dirichlet faces at h/2):
 *   q = G1 L1 / R,  R = (h/2)/K_0 + sum_{i=0}^{nx-2} h / Kh(K_i, K_{i+1}) + (h/2)/K_{nx-1},  Kh = 2 K_a K_b/(K_a + K_b),
 *   H_0 = G1 L1 - q (h/2)/K_0,  H_{i+1} = H_i - q h / Kh(K_i, K_{i+1}).
 *
 * Machine-readable lines: `D_CHECK size=... head_identical=<0|1> bsolver_identical=<0|1> mapped_identical=<0|1|na>`,
 * `C5_CHECK ...`, `MEM_PHASE ...`, `D_LEAN`/`D_REF` (the B_SOLVER lines), a final `VERDICT ...`. Prints
 * `CHECK <name> PASS|FAIL` and `ALL_PASS` / `FAILED n`; exit 0 iff all gated checks of the selected mode pass.
 * Compiled with --fmad=false (CMakeLists.txt) like the other lagrangian tests.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/multigrid/coefficient_hierarchy.cuh"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/operators/varcoeff_laplacian.cuh"
#include "src/numerics/solvers/mg_preconditioner.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/physics/particles/lagrangian_labels/DirichletLongBoxStage.cuh"
#include "src/physics/particles/lagrangian_labels/DirichletMedium.cuh"
#include "src/physics/particles/lagrangian_labels/LongBoxStage.cuh"
#include "src/physics/common/fields.cuh"
#include "src/physics/flow/rhs_head.cuh"
#include "src/physics/flow/solve_head.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
int g_failed = 0, g_passed = 0;
unsigned long long g_diag_seed = 4001ULL; // --diag nx,ny,nz[,levels[,seed]]

void check(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("CHECK %s %s%s%s\n", name.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ",
                detail.c_str());
    std::fflush(stdout);
    ok ? ++g_passed : ++g_failed;
}

std::string fmt(const char* f, double v) {
    char b[96];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

std::string size_str(int nx, int ny, int nz) {
    return std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);
}

bool same_bytes(const std::vector<real>& a, const std::vector<real>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0);
}

unsigned long long fnv1a_host(const std::vector<real>& v) {
    unsigned long long x = 1469598103934665603ULL;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(real); ++i)
        x = (x ^ b[i]) * 1099511628211ULL;
    return x;
}

/// Size-aware comparison of two host heads (x fastest): prints `<tag> sizes=a/b differing=.. first=(i,j,k)
/// maxabs=..` and returns true iff both have `n` elements and are bitwise equal. A size mismatch is a failure
/// (never a vacuous "no differing cell").
bool describe_diff(const std::string& tag, const std::vector<real>& a, const std::vector<real>& b, int nx, int ny,
                   std::size_t n) {
    std::size_t cnt = 0, first = (std::size_t)-1;
    double mx = 0;
    if (a.size() == b.size())
        for (std::size_t c = 0; c < a.size(); ++c)
            if (std::memcmp(&a[c], &b[c], 8) != 0) {
                ++cnt;
                if (first == (std::size_t)-1)
                    first = c;
                mx = std::max(mx, std::abs(a[c] - b[c]));
            }
    const bool sized = a.size() == n && b.size() == n;
    std::printf("DIAG %s sizes=%zu/%zu(expected %zu) differing=%zu first=%lld (i=%lld j=%lld k=%lld) maxabs=%.3e\n",
                tag.c_str(), a.size(), b.size(), n, cnt, (long long)first,
                first == (std::size_t)-1 ? -1LL : (long long)(first % nx),
                first == (std::size_t)-1 ? -1LL : (long long)((first / nx) % ny),
                first == (std::size_t)-1 ? -1LL : (long long)(first / ((std::size_t)nx * ny)), mx);
    return sized && cnt == 0;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

/// SF-18 field exactly as the stage's phase F (and LongBoxStage phase F) calls it.
std::vector<real> gaussian_Y(CudaContext& ctx, int nx, int ny, int nz, double sigma2, double ell,
                             unsigned long long seed) {
    const double h = 1.0 / ny;
    const Grid3D grid(nx, ny, nz, h, h, h);
    const std::size_t n = (std::size_t)nx * ny * nz;
    physics::PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = sigma2;
    cfg.corr_length = ell;
    cfg.seed = seed;
    cfg.normalize_variance = true;
    DeviceBuffer<real> y(n);
    physics::PeriodicGaussianFieldWorkspace ws;
    physics::generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
    ctx.synchronize();
    return download(y.data(), n);
}

ll::DirichletMediumSpec ref_spec(int nx, int ny, int nz, double eps, double G1, int levels) {
    ll::DirichletMediumSpec s;
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    s.eps = eps;
    s.G1 = G1;
    s.mg_levels = levels;
    s.cg_rtol = 1e-10;
    s.cg_max_iter = 2000;
    s.keep_host_head = true;
    return s;
}

ll::DirichletLongBoxSpec lean_spec(int nx, int ny, int nz, double eps, double G1) {
    ll::DirichletLongBoxSpec s;
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    s.eps = eps;
    s.G1 = G1;
    s.mg_levels = 0;
    s.cg_rtol = 1e-10;
    s.cg_max_iter = 2000;
    s.keep_host_head = true;
    return s;
}

// ---- 1. pcg_solve_external ------------------------------------------------------------------------------

struct DiagOp {
    const real* d;
    std::size_t n;
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const;
};

__global__ void kernel_diag_apply(const real* d, const real* x, real* y, std::size_t n) {
    for (std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; i < n;
         i += (std::size_t)gridDim.x * blockDim.x)
        y[i] = d[i] * x[i];
}

void DiagOp::apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
    kernel_diag_apply<<<(int)((n + 255) / 256), 256, 0, ctx.cuda_stream()>>>(d, x.data(), y.data(), n);
}

struct IdentityPrec {
    void apply(CudaContext& ctx, DeviceSpan<const real> r, DeviceSpan<real> z) const { blas::copy(ctx, r, z); }
};

void test_pcg_external(CudaContext& ctx) {
    const std::size_t n = 4096;
    std::vector<real> d(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = 1.0 + (double)(i % 7);
        b[i] = std::sin(0.37 * (double)i) + 0.25;
    }
    DeviceBuffer<real> dd(n), db(n), x1(n), x2(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dd.data(), d.data(), n * 8, cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(db.data(), b.data(), n * 8, cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemset(x1.data(), 0, n * 8));
    MACROFLOW3D_CUDA_CHECK(cudaMemset(x2.data(), 0, n * 8));
    DiagOp A{dd.data(), n};
    IdentityPrec M;
    solvers::PCGConfig cfg;
    cfg.max_iter = 50;
    cfg.check_every = 1;
    cfg.rtol = 1e-12;

    solvers::PCGWorkspace ws;
    const solvers::PCGResult ra = solvers::pcg_solve(ctx, A, M, db.span(), x1.span(), cfg, ws);

    DeviceBuffer<real> r(n), p(n), zAp(n);
    blas::ReductionWorkspace red;
    solvers::PCGExternalVectors v;
    v.r = r.span();
    v.p = p.span();
    v.z = zAp.span();
    v.Ap = zAp.span();
    v.red = &red;
    const solvers::PCGResult rb = solvers::pcg_solve_external(ctx, A, M, db.span(), x2.span(), cfg, v);
    ctx.synchronize();
    const std::vector<real> h1 = download(x1.data(), n), h2 = download(x2.data(), n);
    check("pcg_external_bitwise", same_bytes(h1, h2) && ra.iterations == rb.iterations &&
                                      ra.converged == rb.converged && ra.initial_residual == rb.initial_residual &&
                                      ra.final_residual == rb.final_residual,
          "iters=" + std::to_string(ra.iterations) + "/" + std::to_string(rb.iterations) +
              " final=" + fmt("%.3e", ra.final_residual));
    check("pcg_external_converged", ra.converged && rb.converged);

    auto throws = [&](solvers::PCGExternalVectors vv, DeviceSpan<const real> bb, DeviceSpan<real> xx) {
        try {
            solvers::pcg_solve_external(ctx, A, M, bb, xx, cfg, vv);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    DeviceBuffer<real> big(3 * n), x3(n);
    solvers::PCGExternalVectors w = v;
    bool all = true;
    w = v; w.p = r.span();                                         // r == p
    all = all && throws(w, db.span(), x3.span());
    w = v; w.Ap = DeviceSpan<real>(zAp.data() + 1, n - 1);          // size mismatch
    all = all && throws(w, db.span(), x3.span());
    w = v; w.Ap = DeviceSpan<real>(big.data(), n); w.z = DeviceSpan<real>(big.data() + n / 2, n); // partial z/Ap overlap
    all = all && throws(w, db.span(), x3.span());
    w = v;                                                          // b == x
    all = all && throws(w, db.span(), DeviceSpan<real>(db.data(), n));
    w = v; w.r = DeviceSpan<real>(zAp.data(), n);                   // r overlaps z/Ap
    all = all && throws(w, db.span(), x3.span());
    w = v; w.red = nullptr;
    all = all && throws(w, db.span(), x3.span());
    check("pcg_external_alias_guard", all);
}

// ---- 2. D ------------------------------------------------------------------------------------------------

struct DResult {
    bool head_identical = false, bsolver_identical = false, mapped_identical = false, mapped_run = false;
};

void check_exact_bytes(const std::string& tag, const ll::DirichletLongBoxStage& S, int nx, int ny, int nz,
                       bool head_on_device) {
    const std::size_t n = (std::size_t)nx * ny * nz;
    std::size_t scratch = 0, coarse = 0;
    for (const auto& it : S.phase_S2.items)
        if (it.first.rfind("scratch_", 0) == 0)
            scratch += it.second;
    for (int l = 1; l < S.mg_levels; ++l)
        coarse += (std::size_t)(nx >> l) * (ny >> l) * (nz >> l);
    const std::size_t closed = 8 * ((head_on_device ? 7 : 6) * n + 4 * coarse) + scratch;
    const std::size_t s2 = S.phase_S2.exact_bytes();
    check(tag + "_S2_closed_form", s2 == closed && s2 == ll::dirichlet_s2_closed_form_bytes(nx, ny, nz, S.mg_levels,
                                                                                            head_on_device, scratch),
          "S2=" + std::to_string(s2) + " closed=" + std::to_string(closed) + " (" +
              fmt("%.3f", (double)s2 / (double)n) + " B/cell, scratch=" + std::to_string(scratch) + ")");
    bool le = true;
    std::string det;
    for (const ll::LongBoxPhaseRecord* r : {&S.phase_F, &S.phase_S1, &S.phase_V, &S.phase_H}) {
        if (!r->present)
            continue;
        le = le && r->exact_bytes() <= s2;
        det += " " + r->phase + "=" + std::to_string(r->exact_bytes());
    }
    check(tag + "_phases_le_S2", le, det);
    check(tag + "_exact_is_item_sum", S.phase_V.exact_bytes() > 0 && S.phase_H.exact_bytes() == 8 * n);
}

DResult run_D_case(CudaContext& ctx, int nx, int ny, int nz, double G1_in, bool use_cells, bool do_mapped,
                   bool print_mem) {
    const double eps = 2.0, ell = 0.125;
    const unsigned long long seed = 4001ULL;
    const int levels = ll::box_auto_mg_levels(nx, ny, nz);
    const std::size_t n = (std::size_t)nx * ny * nz;
    const std::string sz = size_str(nx, ny, nz);
    std::vector<real> Y = gaussian_Y(ctx, nx, ny, nz, 1.0, ell, seed);

    double G1 = G1_in;
    if (G1 < 0.0) { // G[0] of build_long_box_stage on the same Y, eps, automatic hierarchy
        ll::LongBoxSpec ls;
        ls.nx = nx;
        ls.ny = ny;
        ls.nz = nz;
        ls.source = ll::LongBoxSourceKind::cells;
        ls.cells = std::move(Y);
        ls.eps = eps;
        const ll::LongBoxStage LS = ll::build_long_box_stage(ctx, ls);
        Y = std::move(ls.cells);
        G1 = LS.G[0];
        std::printf("D_G1 size=%s G1=%.17g darcy_converged=%d\n", sz.c_str(), G1, LS.report.darcy_converged ? 1 : 0);
    }

    const unsigned long long y_hash0 = fnv1a_host(Y);
    ll::DirichletMedium ref = ll::build_dirichlet_medium(ctx, ref_spec(nx, ny, nz, eps, G1, levels), Y);
    const std::string ref_line = ll::format_b_solver(0, ll::make_dirichlet_solver_record(ref.report));
    std::printf("D_REF %s size=%s\n", ref_line.c_str(), sz.c_str());
    // A vacuous comparison (empty reference head) must never pass or be mistaken for a difference of the stage.
    check("D_" + sz + "_ref_head_present", ref.host_head.size() == n,
          "ref.host_head.size()=" + std::to_string(ref.host_head.size()) + " expected " + std::to_string(n) +
              " sizeof(DirichletMedium) test=" + std::to_string(sizeof(ll::DirichletMedium)) + " lib=" +
              std::to_string(ll::dirichlet_medium_object_bytes()));
    check("D_" + sz + "_abi_dirichlet_medium", ll::dirichlet_medium_object_bytes() == sizeof(ll::DirichletMedium));
    check("D_" + sz + "_Y_unchanged_by_ref", fnv1a_host(Y) == y_hash0);

    auto run_lean = [&](bool mapped, DResult* /*unused*/) {
        ll::DirichletLongBoxSpec s = lean_spec(nx, ny, nz, eps, G1);
        s.head_mapped_host = mapped;
        if (use_cells) {
            s.source = ll::LongBoxSourceKind::cells;
            s.cells = std::move(Y);
        } else {
            s.source = ll::LongBoxSourceKind::gaussian;
            s.sigma2 = 1.0;
            s.ell = ell;
            s.seed = seed;
            s.normalize_variance = true;
        }
        ll::DirichletLongBoxStage L = ll::build_dirichlet_long_box_stage(ctx, s);
        if (use_cells)
            Y = std::move(s.cells);
        return L;
    };

    DResult out;
    {
        ll::DirichletLongBoxStage L = run_lean(false, nullptr);
        const std::string lean_line = ll::format_b_solver(0, L.solver);
        std::printf("D_LEAN %s size=%s\n", lean_line.c_str(), sz.c_str());
        out.head_identical = describe_diff("D_" + sz + "_device_vs_ref", L.host_head, ref.host_head, nx, ny, n);
        out.bsolver_identical = lean_line == ref_line;
        check("D_" + sz + "_Y_unchanged_by_device_lean", fnv1a_host(Y) == y_hash0);
        check("D_" + sz + "_head_bitwise", out.head_identical);
        check("D_" + sz + "_bsolver_text", out.bsolver_identical);
        check_exact_bytes("D_" + sz, L, nx, ny, nz, true);
        if (print_mem)
            for (const std::string& l : L.mem_lines())
                std::printf("%s\n", l.c_str());
        if (n <= 16384) { // f = ((-H) - G1 x) + G1 L1 on the host, each operation rounded
            const std::vector<real> f = download(L.f.data(), n);
            bool same = f.size() == n;
            volatile double t1, t2, t3;
            for (std::size_t c = 0; c < n && same; ++c) {
                const int i = (int)(c % nx);
                t1 = ((double)i + 0.5) * L.h;
                t2 = G1 * t1;
                t3 = (-L.host_head[c]) - t2;
                const double e = t3 + L.head_drop;
                same = std::memcmp(&e, &f[c], 8) == 0;
            }
            check("D_" + sz + "_f_formula_bitwise", same);
            check("D_" + sz + "_host_lnk", L.host_lnk.size() == n);
            double mx = 0;
            for (std::size_t c = 0; c < n; ++c)
                mx = std::max(mx, std::abs(L.host_lnk[c] - eps * Y[c]));
            check("D_" + sz + "_host_lnk_exact", mx == 0.0);
        }
        check("D_" + sz + "_inlet_plane", L.inlet_u_plane.size() == (std::size_t)ny * nz);
    }
    if (do_mapped) {
        out.mapped_run = true;
        ll::DirichletLongBoxStage L = run_lean(true, nullptr);
        const std::string lean_line = ll::format_b_solver(0, L.solver);
        out.mapped_identical = describe_diff("D_" + sz + "_mapped_vs_ref", L.host_head, ref.host_head, nx, ny, n) &&
                               lean_line == ref_line;
        if (lean_line != ref_line)
            std::printf("D_LEAN_MAPPED %s size=%s\n", lean_line.c_str(), sz.c_str());
        check("D_" + sz + "_Y_unchanged_by_mapped_lean", fnv1a_host(Y) == y_hash0);
        check("D_" + sz + "_mapped_head_bsolver", out.mapped_identical);
        check_exact_bytes("D_" + sz + "_mapped", L, nx, ny, nz, false);
        if (print_mem)
            for (const std::string& l : L.mem_lines())
                std::printf("%s\n", l.c_str());
    }
    std::printf("D_CHECK size=%s head_identical=%d bsolver_identical=%d mapped_identical=%s\n", sz.c_str(),
                out.head_identical ? 1 : 0, out.bsolver_identical ? 1 : 0,
                out.mapped_run ? (out.mapped_identical ? "1" : "0") : "na");
    std::fflush(stdout);
    return out;
}

// ---- 3. C5 -----------------------------------------------------------------------------------------------

/// Exact discrete 1-D head for the layered K1[i] (see the file header).
std::vector<double> exact_head_1d(const std::vector<double>& K1, double h, double G1) {
    const int nx = (int)K1.size();
    const double L1 = nx * h;
    auto Kh = [&](int i) { return 2.0 * K1[i] * K1[i + 1] / (K1[i] + K1[i + 1]); };
    double R = 0.5 * h / K1[0] + 0.5 * h / K1[nx - 1];
    for (int i = 0; i + 1 < nx; ++i)
        R += h / Kh(i);
    const double q = G1 * L1 / R;
    std::vector<double> H(nx);
    H[0] = G1 * L1 - q * 0.5 * h / K1[0];
    for (int i = 0; i + 1 < nx; ++i)
        H[i + 1] = H[i] - q * h / Kh(i);
    return H;
}

void c5_case(CudaContext& ctx, const std::string& name, int nx, int ny, int nz, bool layered) {
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0, eps = 1.0;
    const std::string sz = size_str(nx, ny, nz);
    std::vector<real> Y((std::size_t)nx * ny * nz, 0.0);
    std::vector<double> K1(nx, 1.0), Hex(nx);
    for (int i = 0; i < nx; ++i) {
        const double x = (i + 0.5) * h;
        const double y = layered ? 0.5 * std::cos(kTwoPi * 0.6 * x + 0.3) : 0.0;
        K1[i] = std::exp(eps * y);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                Y[i + (std::size_t)nx * (j + (std::size_t)ny * k)] = y;
    }
    if (layered) {
        Hex = exact_head_1d(K1, h, G1);
    } else {
        for (int i = 0; i < nx; ++i)
            Hex[i] = G1 * (L1 - (i + 0.5) * h);
    }
    ll::DirichletLongBoxSpec s = lean_spec(nx, ny, nz, eps, G1);
    s.source = ll::LongBoxSourceKind::cells;
    s.cells = Y;
    const ll::DirichletLongBoxStage S = ll::build_dirichlet_long_box_stage(ctx, s);
    const auto& R = S.solver;
    double dmax = 0, hmax = 0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double hv = S.host_head[i + (std::size_t)nx * (j + (std::size_t)ny * k)];
                dmax = std::max(dmax, std::abs(hv - Hex[i]));
                hmax = std::max(hmax, std::abs(Hex[i]));
            }
    const double head_rel = dmax / hmax;
    bool pass = R.converged && R.rel <= 1e-10 && R.balance <= 1e-8;
    if (!layered)
        pass = pass && R.initial == 0.0 && R.rel == 0.0 && R.iterations == 0;
    std::printf("C5_CHECK case=%s size=%s converged=%d iterations=%d initial=%.6e final=%.6e rel=%.6e balance=%.6e "
                "qbar=%.12e max_head_rel_diff=%.6e pass=%d\n",
                name.c_str(), sz.c_str(), R.converged ? 1 : 0, R.iterations, R.initial, R.final, R.rel, R.balance,
                R.qbar, head_rel, pass ? 1 : 0);
    check("C5_" + name + "_" + sz, pass,
          "rel=" + fmt("%.3e", R.rel) + " balance=" + fmt("%.3e", R.balance) + " head_rel_diff=" +
              fmt("%.3e", head_rel));
    if (!layered) {
        const std::vector<real> f = download(S.f.data(), Y.size());
        bool zero = true;
        for (double v : f)
            zero = zero && v == 0.0;
        check("C5_C1_" + sz + "_f_exactly_zero", zero);
        check("C5_C1_" + sz + "_head_exact", dmax == 0.0, "max|H-H_exact|=" + fmt("%.3e", dmax));
    }
}

// ---- 5, 6 ------------------------------------------------------------------------------------------------

void test_flag_off(CudaContext& ctx) {
    const int nx = 32, ny = 16, nz = 16;
    const std::vector<real> Y = gaussian_Y(ctx, nx, ny, nz, 1.0, 0.125, 4001ULL);
    ll::DirichletMediumSpec on = ref_spec(nx, ny, nz, 2.0, 1.0, ll::box_auto_mg_levels(nx, ny, nz));
    on.keep_host_faces = true;
    ll::DirichletMediumSpec off = on;
    off.keep_host_head = false;
    const ll::DirichletMedium A = ll::build_dirichlet_medium(ctx, on, Y);
    const ll::DirichletMedium B = ll::build_dirichlet_medium(ctx, off, Y);
    const auto& a = A.report;
    const auto& b = B.report;
    bool same = a.converged == b.converged && a.iterations == b.iterations &&
                std::memcmp(&a.initial_residual, &b.initial_residual, 8) == 0 &&
                std::memcmp(&a.final_residual, &b.final_residual, 8) == 0 && std::memcmp(&a.rel, &b.rel, 8) == 0 &&
                std::memcmp(&a.qbar, &b.qbar, 8) == 0 && std::memcmp(&a.Q_in, &b.Q_in, 8) == 0 &&
                std::memcmp(&a.Q_out, &b.Q_out, 8) == 0 && std::memcmp(&a.balance, &b.balance, 8) == 0 &&
                std::memcmp(&a.max_div, &b.max_div, 8) == 0 && a.workspace_bytes == b.workspace_bytes;
    same = same && same_bytes(A.face_u, B.face_u) && same_bytes(A.face_v, B.face_v) && same_bytes(A.face_w, B.face_w);
    bool phases = A.phases.size() == B.phases.size();
    for (std::size_t i = 0; phases && i < A.phases.size(); ++i)
        phases = A.phases[i].phase == B.phases[i].phase && A.phases[i].exact_bytes() == B.phases[i].exact_bytes();
    check("flag_off_identical_outputs", same && phases);
    check("flag_off_no_head", B.host_head.empty() && A.host_head.size() == Y.size());
}

void test_inputs(CudaContext& ctx) {
    auto throws = [&](ll::DirichletLongBoxSpec s) {
        try {
            ll::build_dirichlet_long_box_stage(ctx, s);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    ll::DirichletLongBoxSpec s = lean_spec(32, 16, 16, 1.0, 1.0);
    s.source = ll::LongBoxSourceKind::cells;
    s.cells.assign((std::size_t)32 * 16 * 16, 0.0);
    bool all = true;
    auto bad = s; bad.nz = 8;           all = all && throws(bad);
    bad = s;      bad.G1 = 0.0;         all = all && throws(bad);
    bad = s;      bad.mg_levels = 6;    all = all && throws(bad); // 16 / 32 < 2
    bad = s;      bad.cells.resize(10); all = all && throws(bad);
    bad = s;      bad.cg_max_iter = 0;  all = all && throws(bad);
    bad = s;      bad.nx = 31;          all = all && throws(bad);
    check("inputs_invalid_argument", all);
}


/// Diagnostic (no gate): DirichletMedium twice and the lean stage twice on the same Y; pairwise head comparison
/// and the first differing cell. Usage: --diag nx,ny,nz[,mg_levels]
void poison_pool(std::size_t bytes, int byte) {
    unsigned char* p = nullptr;
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&p, bytes));
    MACROFLOW3D_CUDA_CHECK(cudaMemset(p, byte, bytes));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(cudaFree(p));
}


/// Reference replica for the trace: DirichletMedium's K phase and solve_head's PCG_MG branch step by step
/// (public APIs, the same order), with the same hashes/apply wrappers as the lean stage's debug trace.
/// Returns the final head (host).
std::vector<real> trace_reference(CudaContext& ctx, const std::vector<real>& Y, int nx, int ny, int nz, double eps,
                                  double G1, int levels, ll::debug::Trace& tr) {
    const double h = 1.0 / ny, L1 = nx * h;
    const Grid3D grid(nx, ny, nz, h, h, h);
    const std::size_t n = (std::size_t)nx * ny * nz;
    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, G1 * L1);
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = bc.ymax = BCFace(BCType::Periodic, 0.0);
    bc.zmin = bc.zmax = BCFace(BCType::Periodic, 0.0);
    physics::KField Kf;
    physics::HeadField head;
    Kf.resize(grid);
    head.resize(grid);
    std::vector<real> K(n);
    for (std::size_t c = 0; c < n; ++c) {
        const real lnK = eps * Y[c];
        K[c] = std::exp(lnK);
    }
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(Kf.data.data(), K.data(), n * 8, cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    physics::HeadSolveConfig cfg;
    cfg.solver_type = physics::HeadSolverType::PCG_MG;
    cfg.mg_levels = levels;
    cfg.cg_rtol = 1e-10;
    cfg.cg_max_iter = 2000;
    cfg.pin.mode = PinMode::Off;
    multigrid::MGHierarchy hier(grid, levels);
    multigrid::MGConfig mg_cfg;
    mg_cfg.num_levels = hier.num_levels();
    mg_cfg.pre_smooth = cfg.mg_pre_smooth;
    mg_cfg.post_smooth = cfg.mg_post_smooth;
    mg_cfg.coarse_solve_iters = cfg.mg_coarse_iters;
    multigrid::populate_coefficient_hierarchy(ctx, hier, Kf.data.span());
    for (std::size_t l = 0; l < hier.levels.size(); ++l)
        tr.emplace_back("coef_l" + std::to_string(l), ll::debug::fnv1a_device(hier.levels[l].coefficient.data(),
                                                                          hier.levels[l].coefficient.size()));
    physics::build_rhs_head(DeviceSpan<real>(hier.levels[0].b.data(), n),
                            DeviceSpan<const real>(hier.levels[0].coefficient.data(), n), grid, bc, ctx);
    physics::init_head_guess(head.span(), grid, bc, ctx);
    tr.emplace_back("rhs", ll::debug::fnv1a_device(hier.levels[0].b.data(), n));
    tr.emplace_back("head_init", ll::debug::fnv1a_device(head.data.data(), n));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(hier.levels[0].x.data(), head.data.data(), n * 8, cudaMemcpyDeviceToDevice,
                                           ctx.cuda_stream()));
    const PinSpec pin_spec(pin_enabled(PinMode::Off, bc));
    operators::VarCoeffLaplacian A(grid, Kf.data.span(), bc, pin_spec);
    solvers::MultigridPreconditioner M(hier, bc, mg_cfg, pin_spec);
    solvers::PCGWorkspace pws;
    pws.ensure(n);
    DeviceBuffer<real> rhs(n);
    blas::copy(ctx, DeviceSpan<const real>(hier.levels[0].b.span()), rhs.span());
    solvers::PCGConfig pc;
    pc.max_iter = cfg.cg_max_iter;
    pc.check_every = cfg.cg_check_every;
    pc.rtol = cfg.cg_rtol;
    const ll::debug::TracedOp<operators::VarCoeffLaplacian> tA{A, &tr, "A#", 0};
    const ll::debug::TracedOp<solvers::MultigridPreconditioner> tM{M, &tr, "M#", 0};
    solvers::pcg_solve(ctx, tA, tM, DeviceSpan<const real>(rhs.span()), head.span(), pc, pws);
    ctx.synchronize();
    tr.emplace_back("head_final", ll::debug::fnv1a_device(head.data.data(), n));
    return download(head.data.data(), n);
}

void print_first_trace_divergence(const char* nm, const ll::debug::Trace& a, const ll::debug::Trace& b) {
    std::size_t i = 0;
    for (; i < a.size() && i < b.size(); ++i)
        if (a[i].first != b[i].first || a[i].second != b[i].second)
            break;
    if (i == a.size() && i == b.size()) {
        std::printf("DIAG_TRACE %s identical entries=%zu\n", nm, a.size());
        return;
    }
    std::printf("DIAG_TRACE %s FIRST_DIFF index=%zu a=%s:%016llx b=%s:%016llx (sizes %zu/%zu)\n", nm, i,
                i < a.size() ? a[i].first.c_str() : "-", i < a.size() ? a[i].second : 0ULL,
                i < b.size() ? b[i].first.c_str() : "-", i < b.size() ? b[i].second : 0ULL, a.size(), b.size());
}

void run_diag(CudaContext& ctx, int nx, int ny, int nz, int levels_in) {
    const int levels = levels_in > 0 ? levels_in : ll::box_auto_mg_levels(nx, ny, nz);
    std::vector<real> Y = gaussian_Y(ctx, nx, ny, nz, 1.0, 0.125, g_diag_seed);
    ll::DirichletMedium r1 = ll::build_dirichlet_medium(ctx, ref_spec(nx, ny, nz, 2.0, 1.0, levels), Y);
    poison_pool(((std::size_t)nx * ny * nz * 80 + (1u << 20)), 0x7f); // recycled memory = garbage doubles
    ll::DirichletMedium r2 = ll::build_dirichlet_medium(ctx, ref_spec(nx, ny, nz, 2.0, 1.0, levels), Y);
    std::vector<std::vector<real>> L;
    for (int rep = 0; rep < 6; ++rep) {
        poison_pool(((std::size_t)nx * ny * nz * 80 + (1u << 20)), rep == 0 ? 0xff : 0x42);
        ll::DirichletLongBoxSpec s = lean_spec(nx, ny, nz, 2.0, 1.0);
        s.source = ll::LongBoxSourceKind::cells;
        s.cells = Y;
        s.mg_levels = levels_in;
        L.push_back(ll::build_dirichlet_long_box_stage(ctx, s).host_head);
    }
    auto cmp = [&](const char* nm, const std::vector<real>& a, const std::vector<real>& b) {
        describe_diff(nm, a, b, nx, ny, (std::size_t)nx * ny * nz);
    };
    std::printf("DIAG size=%s levels=%d\n", size_str(nx, ny, nz).c_str(), levels);
    {
        ll::debug::Trace tr_ref, tr_lean;
        const std::vector<real> hr = trace_reference(ctx, Y, nx, ny, nz, 2.0, 1.0, levels, tr_ref);
        ll::DirichletLongBoxSpec s = lean_spec(nx, ny, nz, 2.0, 1.0);
        s.source = ll::LongBoxSourceKind::cells;
        s.cells = Y;
        s.mg_levels = levels_in;
        s.debug_trace = &tr_lean;
        const ll::DirichletLongBoxStage T = ll::build_dirichlet_long_box_stage(ctx, s);
        describe_diff("replica_vs_DirichletMedium", hr, r1.host_head, nx, ny, (std::size_t)nx * ny * nz);
        std::printf("DIAG trace_entries ref=%zu lean=%zu\n", tr_ref.size(), tr_lean.size());
        print_first_trace_divergence("replica_vs_lean", tr_ref, tr_lean);
    }
    cmp("ref1_vs_ref2", r1.host_head, r2.host_head);
    cmp("lean1_vs_lean2", L[0], L[1]);
    cmp("ref1_vs_lean1", r1.host_head, L[0]);
    cmp("ref2_vs_lean1", r2.host_head, L[0]);
    for (std::size_t q = 1; q < L.size(); ++q)
        cmp(("lean0_vs_lean" + std::to_string(q)).c_str(), L[0], L[q]);
}

} // namespace

int main(int argc, char** argv) {
    std::string only;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        } else if (std::strcmp(argv[i], "--diag") == 0 && i + 1 < argc) {
            int a = 0, b = 0, c = 0, l = 0;
            std::sscanf(argv[++i], "%d,%d,%d,%d,%llu", &a, &b, &c, &l, &g_diag_seed);
            CudaContext dctx(0);
            run_diag(dctx, a, b, c, l);
            return 0;
        } else {
            std::fprintf(stderr, "usage: %s [--only D|C5]\n", argv[0]);
            return 2;
        }
    }
    if (!only.empty() && only != "D" && only != "C5") {
        std::fprintf(stderr, "unknown --only %s\n", only.c_str());
        return 2;
    }
    const auto t0 = std::chrono::steady_clock::now();
    CudaContext ctx(0);
    try {
        if (only == "D") {
            run_D_case(ctx, 128, 32, 32, 1.0, true, true, true);
            run_D_case(ctx, 512, 64, 64, 1.0, true, true, true);
            run_D_case(ctx, 3328, 256, 256, -1.0, true, false, true);
        } else if (only == "C5") {
            c5_case(ctx, "C1", 64, 16, 16, false);
            c5_case(ctx, "C1", 128, 16, 16, false);
            c5_case(ctx, "C2", 64, 32, 32, true);
            c5_case(ctx, "C2", 128, 32, 32, true);
        } else {
            test_pcg_external(ctx);
            test_inputs(ctx);
            test_flag_off(ctx);
            run_D_case(ctx, 64, 16, 16, 1.0, false, true, true);
            run_D_case(ctx, 32, 16, 16, 1.0, true, true, false);
            run_D_case(ctx, 32, 32, 32, 1.0, true, true, false); // 4 MG levels, the --only D path (cells, device then mapped)
            c5_case(ctx, "C1", 64, 16, 16, false);
            c5_case(ctx, "C2", 32, 16, 16, true);
            c5_case(ctx, "C2", 64, 16, 16, true);
        }
    } catch (const std::exception& e) {
        check("no_exception", false, e.what());
    }
    std::printf("total_seconds=%.2f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::printf("VERDICT mode=%s checks_pass=%d checks_fail=%d\n", only.empty() ? "fast" : only.c_str(), g_passed,
                g_failed);
    std::printf(g_failed == 0 ? "ALL_PASS\n" : "FAILED %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
