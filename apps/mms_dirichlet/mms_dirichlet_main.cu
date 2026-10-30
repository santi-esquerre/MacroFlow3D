/**
 * @file mms_dirichlet_main.cu
 * @brief SF-42 N2: manufactured-solution (MMS) order study of the Dirichlet head-solve path of medium (b).
 *        Documented-experiment instrument; NOT a ctest entry except its fast `--self-check` (contract smoke).
 *
 * Geometry (DirichletLongBoxStage.cu:401-417): cell-centred, h = 1/ny, nx = aspect*ny, L1 = nx h,
 * Grid3D(nx,ny,nz=ny,h,h,h); Dirichlet x faces at x=0 (head G1 L1, G1 = 1) and x=L1 (0), periodic y, z.
 * Solve = the lean stage's sequence (DirichletLongBoxStage.cu:504-570; bitwise-identical to DirichletMedium /
 * solve_head by its header): MG hierarchy, K into level 0 + geometric coarsening, production build_rhs_head,
 * init_head_guess, VarCoeffLaplacian (PinMode::Off), MultigridPreconditioner (2/2 smoothing, coarse 50),
 * pcg_solve_external (check every 10, cg_max_iter 2000), then compute_velocity_from_head. No production code
 * is changed; the only addition is the manufactured source term.
 *
 * Manufactured problem (analytic, below):  K = exp(eps Y_a),  h = (L1 - x) + A p,  p = 0 at x = 0, L1,
 *   f = -div(K grad h) = -K [ A Lap p + eps grad Y_a . grad h ].
 * Sign/scaling of the source: the production operator is A_op = div_h(K grad_h) (negative; cell-centre value,
 * already divided by h^2) and the production RHS carries only -2 K H_bc / h^2 on Dirichlet cells, i.e. the
 * discrete system is  A_op x + (Dirichlet term) = (div(K grad h))(x_c)  =>  b = b_Dirichlet + div(K grad h)(x_c)
 * = b_Dirichlet - f(x_c). No extra h^2 factor.
 */

#include "src/core/BCSpec.hpp"
#include "src/core/Grid3D.hpp"
#include "src/multigrid/coefficient_hierarchy.cuh"
#include "src/multigrid/mg_types.hpp"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/operators/varcoeff_laplacian.cuh"
#include "src/numerics/pin_spec.hpp"
#include "src/numerics/solvers/mg_preconditioner.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/physics/common/fields.cuh"
#include "src/physics/flow/rhs_head.cuh"
#include "src/physics/flow/solve_head.cuh"
#include "src/physics/flow/velocity_from_head.cuh"
#include "src/physics/particles/lagrangian_labels/LongBoxStage.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;

namespace mms {

constexpr double kPi = 3.14159265358979323846;
constexpr double kW = 2.0 * kPi; // period 1 in y and z
constexpr double kA = 1.0;       // amplitude of the head perturbation

struct Pt {
    double K, h, hx, hy, hz, g; // g = div(K grad h) (= -f)
    double Yx;
};

/// Analytic point evaluation. L = L1 (box length in x), eps = contrast.
Pt eval(double x, double y, double z, double eps, double L) {
    const double a = kPi / L;
    const double s1 = std::sin(a * x), c1 = std::cos(a * x), s2 = std::sin(2 * a * x), c2 = std::cos(2 * a * x);
    // p = s1 q + s2 r
    const double sy = std::sin(kW * y + 0.9), cy = std::cos(kW * y + 0.9), cz = std::cos(kW * z + 0.1),
                 sz = std::sin(kW * z + 0.1);
    const double q = 0.5 + 0.4 * sy * cz, qy = 0.4 * kW * cy * cz, qz = -0.4 * kW * sy * sz;
    const double qlap = -2.0 * kW * kW * (q - 0.5);
    const double ry_c = std::cos(kW * y - 0.4), ry_s = std::sin(kW * y - 0.4), rz_s = std::sin(kW * z + 0.7),
                 rz_c = std::cos(kW * z + 0.7);
    const double r = 0.3 * ry_c * rz_s, ry = -0.3 * kW * ry_s * rz_s, rz = 0.3 * kW * ry_c * rz_c;
    const double rlap = -2.0 * kW * kW * r;
    const double p_x = a * c1 * q + 2 * a * c2 * r;
    const double p_y = s1 * qy + s2 * ry, p_z = s1 * qz + s2 * rz;
    const double p_lap = -a * a * s1 * q - 4 * a * a * s2 * r + s1 * qlap + s2 * rlap;
    const double p = s1 * q + s2 * r;
    // Y_a
    const double k2 = 2 * a; // 2 pi / L
    const double e1 = k2 * x + 0.4, e2 = k2 * x + kW * y - 0.2;
    const double Y = 0.5 * std::sin(e1) + 0.4 * std::sin(kW * y + 0.3) * std::cos(kW * z - 0.5) +
                     0.3 * std::cos(e2) * std::sin(kW * z);
    const double Yx = 0.5 * k2 * std::cos(e1) - 0.3 * k2 * std::sin(e2) * std::sin(kW * z);
    const double Yy = 0.4 * kW * std::cos(kW * y + 0.3) * std::cos(kW * z - 0.5) - 0.3 * kW * std::sin(e2) * std::sin(kW * z);
    const double Yz = -0.4 * kW * std::sin(kW * y + 0.3) * std::sin(kW * z - 0.5) + 0.3 * kW * std::cos(e2) * std::cos(kW * z);
    Pt o;
    o.K = std::exp(eps * Y);
    o.h = (L - x) + kA * p;
    o.hx = -1.0 + kA * p_x;
    o.hy = kA * p_y;
    o.hz = kA * p_z;
    o.g = o.K * (kA * p_lap + eps * (Yx * o.hx + Yy * o.hy + Yz * o.hz));
    o.Yx = Yx;
    return o;
}

/// FD check of the analytic fields (host only). Returns the worst relative deviations.
struct FdCheck {
    double grad_h, source, bc_x0, bc_xL, min_dKdx_over_K, max_dKdx_over_K;
};
FdCheck fd_check(double eps, double L) {
    FdCheck c{0, 0, 0, 0, 1e300, -1e300};
    const double d = 1e-4;
    // deterministic pseudo-random points (xorshift64)
    unsigned long long s = 88172645463325252ULL;
    auto u01 = [&] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(s >> 11) / 9007199254740992.0;
    };
    for (int t = 0; t < 200; ++t) {
        const double x = 0.05 + (L - 0.1) * u01(), y = u01(), z = u01();
        const Pt P = eval(x, y, z, eps, L);
        const double gx = (eval(x + d, y, z, eps, L).h - eval(x - d, y, z, eps, L).h) / (2 * d);
        const double gy = (eval(x, y + d, z, eps, L).h - eval(x, y - d, z, eps, L).h) / (2 * d);
        const double gz = (eval(x, y, z + d, eps, L).h - eval(x, y, z - d, eps, L).h) / (2 * d);
        const double gn = std::sqrt(P.hx * P.hx + P.hy * P.hy + P.hz * P.hz);
        c.grad_h = std::max(c.grad_h, std::sqrt((gx - P.hx) * (gx - P.hx) + (gy - P.hy) * (gy - P.hy) +
                                                (gz - P.hz) * (gz - P.hz)) / gn);
        // div(K grad h) by central differences of the analytic flux K grad h
        auto F = [&](double X, double Y_, double Z_, int comp) {
            const Pt Q = eval(X, Y_, Z_, eps, L);
            return Q.K * (comp == 0 ? Q.hx : comp == 1 ? Q.hy : Q.hz);
        };
        const double div = (F(x + d, y, z, 0) - F(x - d, y, z, 0) + F(x, y + d, z, 1) - F(x, y - d, z, 1) +
                            F(x, y, z + d, 2) - F(x, y, z - d, 2)) / (2 * d);
        const double scale = P.K * (1.0 + std::abs(P.g / P.K));
        c.source = std::max(c.source, std::abs(div - P.g) / scale);
    }
    for (int t = 0; t < 200; ++t) {
        const double y = u01(), z = u01();
        const Pt P0 = eval(0.0, y, z, eps, L), PL = eval(L, y, z, eps, L);
        c.bc_x0 = std::max(c.bc_x0, std::abs(P0.h - L)); // h(0) must equal G1 L1 = L
        c.bc_xL = std::max(c.bc_xL, std::abs(PL.h));     // h(L1) must equal 0
        c.min_dKdx_over_K = std::min({c.min_dKdx_over_K, eps * P0.Yx, eps * PL.Yx});
        c.max_dKdx_over_K = std::max({c.max_dKdx_over_K, eps * P0.Yx, eps * PL.Yx});
    }
    return c;
}

} // namespace mms

namespace {

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}
void upload(real* d, const real* h, std::size_t n) {
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

struct Norms {
    double l2 = 0, mx = 0;           // absolute error: RMS and max
    double ref_rms = 0, ref_max = 0; // RMS / max |exact| over the same set
    long long count = 0;
};
struct Acc {
    double s2 = 0, e2 = 0, mx = 0, rmax = 0;
    long long n = 0;
    void add(double err, double exact) {
        s2 += exact * exact;
        e2 += err * err;
        mx = std::max(mx, std::abs(err));
        rmax = std::max(rmax, std::abs(exact));
        ++n;
    }
    Norms done() const {
        Norms o;
        if (n > 0) {
            o.l2 = std::sqrt(e2 / n);
            o.ref_rms = std::sqrt(s2 / n);
        }
        o.mx = mx;
        o.ref_max = rmax;
        o.count = n;
        return o;
    }
};

struct CaseResult {
    int nx, ny, nz, mg_levels;
    double eps, pcg_rtol, L1;
    bool converged;
    int iters;
    double res0, resN;
    Norms head, head_far, flux_interior, flux_far, flux_dirichlet, flux_first_layer;
};

/// control_k1: K = 1, h = L1 - x, zero source, zero initial guess (round-off control of RHS/operator/velocity).
CaseResult run_case(CudaContext& ctx, int ny, int aspect, double eps, double pcg_rtol, bool control_k1 = false) {
    const int nx = aspect * ny, nz = ny;
    const std::size_t n = (std::size_t)nx * ny * nz;
    const double h = 1.0 / ny, L1 = nx * h;
    const Grid3D grid(nx, ny, nz, h, h, h);
    const int levels = lagrangian_labels::box_auto_mg_levels(nx, ny, nz);

    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, 1.0 * L1); // G1 = 1: head_drop = G1 L1
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = bc.ymax = BCFace(BCType::Periodic, 0.0);
    bc.zmin = bc.zmax = BCFace(BCType::Periodic, 0.0);

    // host K and manufactured divergence g = div(K grad h) at the cell centres
    std::vector<real> Kh(n), gh(n);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = (std::size_t)i + (std::size_t)nx * (j + (std::size_t)ny * k);
                if (control_k1) {
                    Kh[c] = 1.0;
                    gh[c] = 0.0;
                } else {
                    const mms::Pt P = mms::eval((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h, eps, L1);
                    Kh[c] = P.K;
                    gh[c] = P.g;
                }
            }

    CaseResult R{};
    R.nx = nx;
    R.ny = ny;
    R.nz = nz;
    R.mg_levels = levels;
    R.eps = eps;
    R.pcg_rtol = pcg_rtol;
    R.L1 = L1;

    physics::KField Kf;
    physics::HeadField head;
    {
        physics::HeadSolveConfig hc; // defaults of solve_head: MG 2/2 smoothing, coarse 50, CG check every 10
        multigrid::MGHierarchy hier(grid, levels);
        if (hier.num_levels() != levels)
            throw std::invalid_argument("MG hierarchy has fewer levels than requested");
        DeviceBuffer<real> pcg_r, pcg_p;
        blas::ReductionWorkspace red;
        head.resize(grid);
        pcg_r.resize(n);
        pcg_p.resize(n);
        auto& lvl0 = hier.levels.front();
        upload(lvl0.coefficient.data(), Kh.data(), n);
        for (int l = 0; l + 1 < hier.num_levels(); ++l) {
            auto& fine = hier.levels[l];
            auto& coarse = hier.levels[l + 1];
            multigrid::coarsen_coefficient_geometric(
                ctx, coarse.grid, fine.grid, DeviceSpan<const real>(fine.coefficient.data(), fine.coefficient.size()),
                DeviceSpan<real>(coarse.coefficient.data(), coarse.coefficient.size()));
        }
        physics::build_rhs_head(DeviceSpan<real>(lvl0.r.data(), n), DeviceSpan<const real>(lvl0.coefficient.data(), n),
                                grid, bc, ctx);
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        { // b = b_Dirichlet + div(K grad h)(x_c)   (see file header)
            std::vector<real> b = download(lvl0.r.data(), n);
            for (std::size_t c = 0; c < n; ++c)
                b[c] += gh[c];
            upload(lvl0.r.data(), b.data(), n);
        }
        const DeviceSpan<real> x(head.data.data(), n);
        physics::init_head_guess(x, grid, bc, ctx);
        ctx.synchronize();
        if (control_k1) // vacuity guard: start away from the exact solution
            MACROFLOW3D_CUDA_CHECK(cudaMemset(head.data.data(), 0, n * sizeof(real)));

        const PinSpec pin_spec(pin_enabled(PinMode::Off, bc));
        operators::VarCoeffLaplacian A(grid, DeviceSpan<const real>(lvl0.coefficient.data(), n), bc, pin_spec);
        multigrid::MGConfig mg_cfg;
        mg_cfg.num_levels = hier.num_levels();
        mg_cfg.pre_smooth = hc.mg_pre_smooth;
        mg_cfg.post_smooth = hc.mg_post_smooth;
        mg_cfg.coarse_solve_iters = hc.mg_coarse_iters;
        mg_cfg.verbose = false;
        solvers::MultigridPreconditioner M(hier, bc, mg_cfg, pin_spec);

        solvers::PCGConfig pcg_cfg;
        pcg_cfg.max_iter = 2000; // production cg_max_iter of medium (b)
        pcg_cfg.check_every = hc.cg_check_every;
        pcg_cfg.rtol = pcg_rtol;
        pcg_cfg.verbose = false;
        solvers::PCGExternalVectors vec;
        vec.r = DeviceSpan<real>(pcg_r.data(), n);
        vec.p = DeviceSpan<real>(pcg_p.data(), n);
        vec.z = DeviceSpan<real>(lvl0.b.data(), n);
        vec.Ap = DeviceSpan<real>(lvl0.b.data(), n);
        vec.red = &red;
        const solvers::PCGResult r =
            solvers::pcg_solve_external(ctx, A, M, DeviceSpan<const real>(lvl0.r.data(), n), x, pcg_cfg, vec);
        ctx.synchronize();
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        R.converged = r.converged;
        R.iters = r.iterations;
        R.res0 = r.initial_residual;
        R.resN = r.final_residual;
        Kf.nx = nx;
        Kf.ny = ny;
        Kf.nz = nz;
        Kf.dx = Kf.dy = Kf.dz = h;
        Kf.data = std::move(lvl0.coefficient);
    } // MG hierarchy and PCG buffers freed

    physics::VelocityField vel;
    vel.resize(grid);
    physics::compute_velocity_from_head(vel, head, Kf, grid, bc, ctx);
    ctx.synchronize();
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<real> H = download(head.data.data(), n);
    const std::vector<real> U = download(vel.U.data(), vel.U.size());
    const std::vector<real> V = download(vel.V.data(), vel.V.size());
    const std::vector<real> W = download(vel.W.data(), vel.W.size());

    // ---- head errors at the cell centres ----
    Acc ah, ahf; // ahf: cells at least two cells from the Dirichlet faces (2 <= i <= nx-3, as the far V/W faces)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = (std::size_t)i + (std::size_t)nx * (j + (std::size_t)ny * k);
                const double ex = control_k1 ? L1 - (i + 0.5) * h
                                             : mms::eval((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h, eps, L1).h;
                ah.add(H[c] - ex, ex);
                if (i >= 2 && i <= nx - 3)
                    ahf.add(H[c] - ex, ex);
            }
    R.head = ah.done();
    R.head_far = ahf.done();

    // ---- face-flux errors against the exact point value -K dh/dn at the face centres ----
    // Sets (face index i along x for U; owning-cell index i along x for V/W):
    //   Dirichlet faces     : U i in {0, nx}
    //   first interior layer: U i in {1, nx-1} and the V/W faces of the boundary cells i in {0, nx-1}
    //   interior (i)        : every face except the Dirichlet faces (U i = 1..nx-1, all V, W)
    //   far (ii, max norm)  : faces at least two cells from the Dirichlet faces:
    //                         U 2 <= i <= nx-2, V/W (cell-centre distance (i+1/2)h >= 2h) 2 <= i <= nx-3
    // V, W: j (resp. k) = 0..n-1 only (the periodic face j = ny repeats j = 0).
    Acc ain, afar, adir, afirst;
    auto exact = [&](double x, double y, double z, int comp) {
        if (control_k1)
            return comp == 0 ? 1.0 : 0.0; // K = 1, h = L1 - x: flux = -dh/dx = 1
        const mms::Pt P = mms::eval(x, y, z, eps, L1);
        return -P.K * (comp == 0 ? P.hx : comp == 1 ? P.hy : P.hz);
    };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const double ex = exact(i * h, (j + 0.5) * h, (k + 0.5) * h, 0);
                const double err = U[(std::size_t)i + (std::size_t)(nx + 1) * (j + (std::size_t)ny * k)] - ex;
                if (i == 0 || i == nx) {
                    adir.add(err, ex);
                    continue;
                }
                ain.add(err, ex);
                if (i == 1 || i == nx - 1)
                    afirst.add(err, ex);
                if (i >= 2 && i <= nx - 2)
                    afar.add(err, ex);
            }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double ey = exact((i + 0.5) * h, j * h, (k + 0.5) * h, 1);
                const double errv = V[(std::size_t)i + (std::size_t)nx * (j + (std::size_t)(ny + 1) * k)] - ey;
                const double ez = exact((i + 0.5) * h, (j + 0.5) * h, k * h, 2);
                const double errw = W[(std::size_t)i + (std::size_t)nx * (j + (std::size_t)ny * k)] - ez;
                ain.add(errv, ey);
                ain.add(errw, ez);
                if (i == 0 || i == nx - 1) {
                    afirst.add(errv, ey);
                    afirst.add(errw, ez);
                }
                if (i >= 2 && i <= nx - 3) {
                    afar.add(errv, ey);
                    afar.add(errw, ez);
                }
            }
    R.flux_interior = ain.done();
    R.flux_far = afar.done();
    R.flux_dirichlet = adir.done();
    R.flux_first_layer = afirst.done();
    return R;
}

void write_norms(FILE* f, const char* name, const Norms& N, bool last = false) {
    std::fprintf(f,
                 "  \"%s\": {\"count\": %lld, \"l2_abs\": %.17g, \"max_abs\": %.17g, \"l2_rel\": %.17g, "
                 "\"max_rel\": %.17g, \"exact_rms\": %.17g, \"exact_max\": %.17g}%s\n",
                 name, N.count, N.l2, N.mx, N.ref_rms > 0 ? N.l2 / N.ref_rms : 0.0,
                 N.ref_rms > 0 ? N.mx / N.ref_rms : 0.0, N.ref_rms, N.ref_max, last ? "" : ",");
}

void write_json(const std::string& path, const CaseResult& R, const std::string& commit, const std::string& args) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "{\n  \"schema\": \"sf42-mms-dirichlet-1\",\n  \"commit\": \"%s\",\n  \"args\": \"%s\",\n",
                 commit.c_str(), args.c_str());
    std::fprintf(f, "  \"nx\": %d, \"ny\": %d, \"nz\": %d, \"h\": %.17g, \"L1\": %.17g, \"eps\": %.17g,\n", R.nx, R.ny,
                 R.nz, 1.0 / R.ny, R.L1, R.eps);
    std::fprintf(f, "  \"mg_levels\": %d, \"pcg_rtol\": %.17g, \"pcg_iterations\": %d, \"pcg_converged\": %s,\n",
                 R.mg_levels, R.pcg_rtol, R.iters, R.converged ? "true" : "false");
    std::fprintf(f, "  \"residual_initial\": %.17g, \"residual_final\": %.17g, \"residual_rel\": %.17g,\n", R.res0, R.resN,
                 R.res0 > 0 ? R.resN / R.res0 : 0.0);
    write_norms(f, "head", R.head);
    write_norms(f, "head_far", R.head_far);
    write_norms(f, "flux_interior", R.flux_interior);
    write_norms(f, "flux_far", R.flux_far);
    write_norms(f, "flux_dirichlet", R.flux_dirichlet);
    write_norms(f, "flux_first_layer", R.flux_first_layer, true);
    std::fprintf(f, "}\n");
    std::fclose(f);
}

int self_check(CudaContext& ctx) {
    bool ok = true;
    auto expect = [&](bool c, const char* what, double v) {
        std::printf("%s %-62s %.6g\n", c ? "PASS" : "FAIL", what, v);
        ok = ok && c;
    };
    // 1. analytic fields against finite differences, and the problem requirements
    for (double eps : {0.5, 2.0}) {
        const auto c = mms::fd_check(eps, 4.0);
        expect(c.grad_h < 1e-6, "analytic grad h vs central FD (rel)", c.grad_h);
        expect(c.source < 1e-5, "analytic div(K grad h) vs FD of the flux (rel)", c.source); // FD truncation ~1e-6 at eps=2; a derivation error is O(0.1)
        expect(c.bc_x0 < 1e-13 && c.bc_xL < 1e-13, "h(0) = G1 L1 and h(L1) = 0 (abs)", std::max(c.bc_x0, c.bc_xL));
        expect(std::abs(c.min_dKdx_over_K) > 0.1 && c.min_dKdx_over_K * c.max_dKdx_over_K > 0.0,
               "dlnK/dx != 0, one sign, at both Dirichlet faces (min |.|)",
               std::min(std::abs(c.min_dKdx_over_K), std::abs(c.max_dKdx_over_K)));
    }
    // 2. K = 1, h = L1 - x, zero source, zero initial guess: solve + velocity reproduce the linear head
    {
        const CaseResult R = run_case(ctx, 8, 4, 0.0, 1e-13, true);
        expect(R.converged && R.iters > 0, "K=1 linear head: PCG converged from a zero guess (iters)", R.iters);
        expect(R.head.mx < 1e-10, "K=1 linear head reproduced (max abs error)", R.head.mx);
        expect(std::max(R.flux_interior.mx, R.flux_dirichlet.mx) < 1e-10,
               "K=1 linear fluxes reproduced incl. Dirichlet faces (max abs)",
               std::max(R.flux_interior.mx, R.flux_dirichlet.mx));
    }
    // 3. manufactured case converges at order ~2 in the head (sign / scaling of the source)
    {
        const CaseResult a = run_case(ctx, 8, 4, 1.0, 1e-10), b = run_case(ctx, 16, 4, 1.0, 1e-10);
        const double p = std::log2(a.head.l2 / b.head.l2);
        expect(a.converged && b.converged, "manufactured eps=1: PCG converged (iters at ny=16)", b.iters);
        expect(p > 1.7, "manufactured eps=1: head L2 observed order (ny 8->16)", p);
    }
    std::printf("%s\n", ok ? "SELF-CHECK PASS" : "SELF-CHECK FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int ny = 0, aspect = 4;
    double eps = 1.0, rtol = 1e-10;
    std::string out_dir, commit = "unknown";
    bool do_self = false;
    std::string args;
    for (int a = 1; a < argc; ++a) {
        args += (a > 1 ? " " : "");
        args += argv[a];
    }
    try {
        for (int a = 1; a < argc; ++a) {
            const std::string s = argv[a];
            auto next = [&]() -> const char* {
                if (a + 1 >= argc)
                    throw std::invalid_argument("missing value for " + s);
                return argv[++a];
            };
            if (s == "--ny")
                ny = std::atoi(next());
            else if (s == "--eps")
                eps = std::atof(next());
            else if (s == "--aspect")
                aspect = std::atoi(next());
            else if (s == "--pcg-rtol")
                rtol = std::atof(next());
            else if (s == "--out-dir")
                out_dir = next();
            else if (s == "--commit")
                commit = next();
            else if (s == "--self-check")
                do_self = true;
            else
                throw std::invalid_argument("unknown argument " + s);
        }
        CudaContext ctx(0);
        if (do_self)
            return self_check(ctx);
        if (ny < 8 || aspect < 1 || !(rtol > 0) || !std::isfinite(eps) || out_dir.empty())
            throw std::invalid_argument("usage: mms_dirichlet --ny N(>=8) --eps E --out-dir DIR [--aspect 4] "
                                        "[--pcg-rtol 1e-10] [--commit SHA] | --self-check");
        const CaseResult R = run_case(ctx, ny, aspect, eps, rtol);
        char name[256];
        std::snprintf(name, sizeof name, "mms_ny%04d_eps%g_rtol%g.json", ny, eps, rtol);
        write_json(out_dir + "/" + name, R, commit, args);
        std::printf("%s: iters=%d converged=%d rel_res=%.3e head_l2=%.4e flux_int_l2=%.4e flux_far_max=%.4e\n", name,
                    R.iters, (int)R.converged, R.res0 > 0 ? R.resN / R.res0 : 0.0, R.head.l2, R.flux_interior.l2,
                    R.flux_far.mx);
        return R.converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mms_dirichlet: %s\n", e.what());
        return 1;
    }
}
