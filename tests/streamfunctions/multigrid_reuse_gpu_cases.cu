#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/multigrid/coefficient_hierarchy.cuh"
#include "src/multigrid/cycle/projected_positive_v_cycle.cuh"
#include "src/multigrid/mg_types.hpp"
#include "src/multigrid/smoothers/gsrb_3d.cuh"
#include "src/numerics/blas/blas.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/numerics/solvers/projected_positive_mg_preconditioner.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {

static_assert(std::is_same_v<real, double>, "SF-05 acceptance controls require double precision");
constexpr long double kPi = 3.141592653589793238462643383279502884L;
constexpr double kRoundoff = 4096.0 * std::numeric_limits<real>::epsilon();
constexpr double kResidualLimit = 1.0e-10;
constexpr double kSolverTolerance = 1.0e-12;
constexpr double kGaugeFactor = 4096.0;

[[nodiscard]] std::size_t idx(int n, int x, int y, int z) {
    return x + n * (y + n * z);
}
[[nodiscard]] int wrap(int n, int i) {
    return i < 0 ? i + n : (i == n ? 0 : i);
}

[[nodiscard]] long double mean(const std::vector<real>& v) {
    long double s = 0;
    for (real x : v)
        s += static_cast<long double>(x);
    return s / v.size();
}
[[nodiscard]] double rms(const std::vector<real>& v) {
    long double s = 0;
    for (real x : v) {
        const long double y = x;
        s += y * y;
    }
    return std::sqrt(static_cast<double>(s / v.size()));
}
[[nodiscard]] double dot(const std::vector<real>& a, const std::vector<real>& b) {
    long double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += static_cast<long double>(a[i]) * b[i];
    return static_cast<double>(s);
}
[[nodiscard]] std::vector<real> project(std::vector<real> v) {
    const long double m = mean(v);
    for (real& x : v)
        x = static_cast<real>(static_cast<long double>(x) - m);
    return v;
}
[[nodiscard]] std::vector<real> q_field(int n, bool smooth = true) {
    std::vector<real> q(static_cast<std::size_t>(n) * n * n, real(1));
    if (!smooth)
        return q;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const long double phase =
                    2 * kPi * (static_cast<long double>(x + y + z) + 1.5L) / n;
                q[idx(n, x, y, z)] = static_cast<real>(1.25L + 0.25L * std::cos(phase));
            }
    return q;
}
[[nodiscard]] std::vector<real> manufactured(int n, bool broad = true) {
    std::vector<real> u(static_cast<std::size_t>(n) * n * n);
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const long double X = (x + .5L) / n, Y = (y + .5L) / n, Z = (z + .5L) / n;
                long double value = .45L * sin(2 * kPi * X) - .30L * cos(4 * kPi * Y) +
                                    .20L * sin(2 * kPi * Z) + .10L * cos(2 * kPi * (X + Y - Z));
                // The added fixed modes prevent a deceptively easy single-eigenvalue PCG control.
                if (broad)
                    value += .08L * sin(6 * kPi * X + 4 * kPi * Y) +
                             .06L * cos(4 * kPi * Y - 6 * kPi * Z) + .04L * sin(8 * kPi * (X + Z));
                u[idx(n, x, y, z)] = static_cast<real>(value);
            }
    return project(std::move(u));
}
[[nodiscard]] std::vector<real> apply_cpu(const std::vector<real>& q, const std::vector<real>& u,
                                          int n) {
    std::vector<real> a(u.size());
    const long double h2 = static_cast<long double>(n) * n;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const auto c = idx(n, x, y, z);
                long double sum = 0;
                const long double qc = q[c], uc = u[c];
                for (int d = 0; d < 3; ++d)
                    for (int s : {-1, 1}) {
                        const int xx = d == 0 ? wrap(n, x + s) : x,
                                  yy = d == 1 ? wrap(n, y + s) : y,
                                  zz = d == 2 ? wrap(n, z + s) : z;
                        const auto j = idx(n, xx, yy, zz);
                        const long double qf = 2 * qc * q[j] / (qc + q[j]);
                        sum += qf * (uc - u[j]);
                    }
                a[c] = static_cast<real>(sum * h2);
            }
    return a;
}
[[nodiscard]] std::vector<real> residual_cpu(const std::vector<real>& q, const std::vector<real>& b,
                                             const std::vector<real>& x, int n) {
    auto a = apply_cpu(q, x, n);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = b[i] - a[i];
    return project(std::move(a));
}
[[nodiscard]] std::vector<real> coarsen_geometric_cpu(const std::vector<real>& fine, int fine_n) {
    const int coarse_n = fine_n / 2;
    std::vector<real> coarse(static_cast<std::size_t>(coarse_n) * coarse_n * coarse_n);
    for (int z = 0; z < coarse_n; ++z)
        for (int y = 0; y < coarse_n; ++y)
            for (int x = 0; x < coarse_n; ++x) {
                long double log_sum = 0;
                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            log_sum += std::log(static_cast<long double>(
                                fine[idx(fine_n, 2 * x + dx, 2 * y + dy, 2 * z + dz)]));
                coarse[idx(coarse_n, x, y, z)] = static_cast<real>(std::exp(log_sum / 8));
            }
    return coarse;
}
void upload(CudaContext& c, DeviceBuffer<real>& d, const std::vector<real>& h) {
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(real),
                                           cudaMemcpyHostToDevice, c.cuda_stream()));
}
[[nodiscard]] std::vector<real> download(CudaContext& c, const DeviceBuffer<real>& d) {
    std::vector<real> h(d.size());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.size() * sizeof(real),
                                           cudaMemcpyDeviceToHost, c.cuda_stream()));
    c.synchronize();
    return h;
}
[[nodiscard]] multigrid::MGConfig mg_config(int levels) {
    multigrid::MGConfig c;
    c.num_levels = levels;
    c.pre_smooth = 2;
    c.post_smooth = 2;
    c.coarse_solve_iters = 50;
    return c;
}
[[nodiscard]] std::vector<const real*> coefficient_pointers(const multigrid::MGHierarchy& h) {
    std::vector<const real*> p;
    for (const auto& l : h.levels)
        p.push_back(l.coefficient.data());
    return p;
}
[[nodiscard]] std::vector<std::vector<real>> coefficients(CudaContext& c,
                                                          const multigrid::MGHierarchy& h) {
    std::vector<std::vector<real>> all;
    for (const auto& l : h.levels) {
        std::vector<real> v(l.coefficient.size());
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(v.data(), l.coefficient.data(),
                                               v.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                               c.cuda_stream()));
        all.push_back(std::move(v));
    }
    c.synchronize();
    return all;
}
[[nodiscard]] bool exact(const std::vector<std::vector<real>>& a,
                         const std::vector<std::vector<real>>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() ||
            std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(real)))
            return false;
    return true;
}
[[nodiscard]] double gauge_limit(const std::vector<real>& v) {
    return kGaugeFactor * std::numeric_limits<real>::epsilon() * std::max(rms(v), 1.0);
}

class Identity {
  public:
    void apply(CudaContext& c, DeviceSpan<const real> r, DeviceSpan<real> z) const {
        blas::copy(c, r, z);
    }
};

[[nodiscard]] CaseResult hierarchy_case() {
    double worst = 0;
    bool inverse = true, reuse = true;
    std::ostringstream grids;
    for (int n : {32, 64}) {
        CudaContext c(0);
        Grid3D g(n, n, n, real(1.0 / n), real(1.0 / n), real(1.0 / n));
        multigrid::MGHierarchy h(g, 4);
        auto q = q_field(n);
        std::vector<real> K(q.size());
        for (std::size_t i = 0; i < q.size(); ++i)
            K[i] = real(1) / q[i];
        DeviceBuffer<real> dq(q.size()), dK(K.size());
        upload(c, dq, q);
        upload(c, dK, K);
        multigrid::populate_coefficient_hierarchy(c, h, dq.span());
        multigrid::MGHierarchy k_hierarchy(g, 4);
        multigrid::populate_coefficient_hierarchy(c, k_hierarchy, dK.span());
        const auto before = coefficients(c, h);
        const auto pointers = coefficient_pointers(h);
        const auto K_before = coefficients(c, k_hierarchy);
        std::vector<std::vector<real>> oracle_levels{q};
        int oracle_n = n;
        for (std::size_t level = 1; level < h.levels.size(); ++level) {
            oracle_levels.push_back(coarsen_geometric_cpu(oracle_levels.back(), oracle_n));
            oracle_n /= 2;
        }
        for (std::size_t l = 0; l < h.levels.size(); ++l) {
            const auto& oracle = oracle_levels[l];
            for (std::size_t i = 0; i < oracle.size(); ++i) {
                worst =
                    std::max(worst, std::abs(before[l][i] - oracle[i]) /
                                        std::max(std::abs(static_cast<double>(oracle[i])), 1.0));
                // Independently coarsen K=1/q and compare level coefficients.
                inverse =
                    inverse && std::abs(static_cast<long double>(before[l][i]) * K_before[l][i] -
                                        1.0L) <= kRoundoff;
            }
        }
        auto cfg = mg_config(h.num_levels());
        solvers::ProjectedPositiveMGPreconditioner m(h, cfg);
        auto r = manufactured(n);
        DeviceBuffer<real> dr(r.size()), dz(r.size());
        upload(c, dr, r);
        m.apply(c, dr.span(), dz.span());
        m.apply(c, dr.span(), dz.span());
        reuse = reuse && exact(before, coefficients(c, h)) && pointers == coefficient_pointers(h);
        grids << n << "^3/" << h.num_levels() << " ";
    }
    const bool pass = worst <= kRoundoff && inverse && reuse;
    std::cout << std::setprecision(12)
              << "mg_suite case=mg_hierarchy_q_geometric_coarsening grids=" << grids.str()
              << " max_relative=" << worst << " envelope=" << kRoundoff << " inverse_qK=" << inverse
              << " reuse=" << reuse << '\n';
    return {pass,
            "mg_hierarchy_q_geometric_coarsening",
            "gpu-mg-reuse",
            grids.str(),
            worst,
            kRoundoff,
            "<=4096eps",
            std::to_string(worst),
            "CPU long-double geometric oracle; buffers/pointers invariant"};
}

[[nodiscard]] CaseResult preconditioner_contract_case() {
    constexpr int n = 32;
    CudaContext c(0);
    Grid3D g(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n);
    multigrid::MGHierarchy h(g, 4);
    auto q = q_field(n);
    DeviceBuffer<real> dq(q.size());
    upload(c, dq, q);
    multigrid::populate_coefficient_hierarchy(c, h, dq.span());
    const auto coeff = coefficients(c, h);
    const auto ptr = coefficient_pointers(h);
    auto cfg = mg_config(h.num_levels());
    solvers::ProjectedPositiveMGPreconditioner m(h, cfg);
    auto u = manufactured(n), v = manufactured(n);
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<real>(v[i] + .13 * sin(static_cast<double>(i) * .17));
    v = project(std::move(v));
    const auto u0 = u, v0 = v;
    DeviceBuffer<real> du(u.size()), dv(v.size()), mu(u.size()), mv(u.size());
    upload(c, du, u);
    upload(c, dv, v);
    m.apply(c, du.span(), mu.span());
    m.apply(c, dv.span(), mv.span());
    auto hu = download(c, mu), hv = download(c, mv);
    const double utu = dot(u, hu), vtv = dot(v, hv),
                 delta = std::abs(dot(u, hv) - dot(v, hu)) / std::sqrt(utu * vtv);
    bool levels = true;
    for (std::size_t l = 0; l < h.levels.size(); ++l) {
        auto b = download(c, h.levels[l].b), x = download(c, h.levels[l].x);
        levels = levels && std::abs(static_cast<double>(mean(b))) <= gauge_limit(b) &&
                 std::abs(static_cast<double>(mean(x))) <= gauge_limit(x);
        if (l + 1 < h.levels.size()) {
            auto r = download(c, h.levels[l].r);
            levels = levels && std::abs(static_cast<double>(mean(r))) <= gauge_limit(r);
        }
    }
    bool invalid = true;
    for (auto bad : {[]() {
                         auto x = mg_config(4);
                         x.pre_smooth = 0;
                         return x;
                     }(),
                     []() {
                         auto x = mg_config(4);
                         x.post_smooth = 0;
                         return x;
                     }(),
                     []() {
                         auto x = mg_config(3);
                         return x;
                     }()}) {
        try {
            multigrid::validate_projected_positive_hierarchy(h, bad);
            invalid = false;
        } catch (const std::invalid_argument&) {
        }
    }
    // 24 -> 12 -> 6 -> 3 isolates an odd coarsest level.
    multigrid::MGHierarchy odd(Grid3D(24, 24, 24, 1.0 / 24, 1.0 / 24, 1.0 / 24), 4);
    try {
        multigrid::validate_projected_positive_hierarchy(odd, mg_config(odd.num_levels()));
        invalid = false;
    } catch (const std::invalid_argument&) {
    }
    multigrid::MGHierarchy bad_anisotropic(Grid3D(32, 32, 32, 1.0 / 32, 1.0 / 32, 1.0 / 32), 4);
    bad_anisotropic.levels[0].grid.dy *= real(1.01);
    try {
        multigrid::validate_projected_positive_hierarchy(bad_anisotropic,
                                                         mg_config(bad_anisotropic.num_levels()));
        invalid = false;
    } catch (const std::invalid_argument&) {
    }
    multigrid::MGHierarchy bad_spacing(Grid3D(32, 32, 32, 1.0 / 32, 1.0 / 32, 1.0 / 32), 4);
    bad_spacing.levels[1].grid.dx *= real(1.01);
    bad_spacing.levels[1].grid.dy *= real(1.01);
    bad_spacing.levels[1].grid.dz *= real(1.01);
    try {
        multigrid::validate_projected_positive_hierarchy(bad_spacing,
                                                         mg_config(bad_spacing.num_levels()));
        invalid = false;
    } catch (const std::invalid_argument&) {
    }
    const auto du_after = download(c, du), dv_after = download(c, dv);
    const bool inputs_immutable = du_after == u0 && dv_after == v0;
    const bool reuse = exact(coeff, coefficients(c, h)) && ptr == coefficient_pointers(h);
    const bool pass = delta <= 1e-10 && utu > kRoundoff && vtv > kRoundoff && inputs_immutable &&
                      std::abs(static_cast<double>(mean(hu))) <= gauge_limit(hu) &&
                      std::abs(static_cast<double>(mean(hv))) <= gauge_limit(hv) && levels &&
                      invalid && reuse;
    std::cout << std::setprecision(12)
              << "mg_suite case=mg_projected_preconditioner_contract grid=32^3 delta_sym=" << delta
              << " uMu=" << utu << " vMv=" << vtv << " input_immutable=" << inputs_immutable
              << " output_gauge="
              << std::max(std::abs(static_cast<double>(mean(hu))),
                          std::abs(static_cast<double>(mean(hv))))
              << " levels=" << levels << " invalid_early=" << invalid << " reuse=" << reuse << '\n';
    return {
        pass,
        "mg_projected_preconditioner_contract",
        "gpu-mg-reuse",
        "32^3",
        delta,
        std::max(std::abs(static_cast<double>(mean(hu))), std::abs(static_cast<double>(mean(hv)))),
        "sym<=1e-10",
        std::to_string(delta),
        "positive forms, gauges, invalid hierarchy and reuse"};
}

struct SolveMetrics {
    int id_iter{}, mg_iter{};
    double id_res{}, mg_res{}, solution{};
    bool pass{};
};

[[nodiscard]] SolveMetrics solve_control(int n, bool smooth) {
    CudaContext c(0);
    Grid3D g(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n);
    auto q = q_field(n, smooth), u = manufactured(n), b = project(apply_cpu(q, u, n));
    DeviceBuffer<real> dq(q.size()), db(b.size()), xi(b.size()), xm(b.size());
    upload(c, dq, q);
    upload(c, db, b);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(xi.data(), 0, xi.size() * sizeof(real), c.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(xm.data(), 0, xm.size() * sizeof(real), c.cuda_stream()));
    operators::LesterPositiveDiffusionOperator A(g, dq.span());
    constraints::MeanZeroProjector projector;
    solvers::ProjectedPCGWorkspace wi, wm;
    wi.prepare(b.size());
    wm.prepare(b.size());
    // Identity is a reference solve, not subject to the SF-05 MG cap.
    solvers::ProjectedPCGConfig identity_cfg{3000, 1, kSolverTolerance};
    solvers::ProjectedPCGConfig mg_cfg{100, 1, kSolverTolerance};
    Identity identity;
    const auto ri = solvers::projected_pcg_solve(c, A, identity, db.span(), xi.span(), identity_cfg,
                                                 projector, wi);
    multigrid::MGHierarchy h(g, 4);
    multigrid::populate_coefficient_hierarchy(c, h, dq.span());
    solvers::ProjectedPositiveMGPreconditioner mg(h, mg_config(h.num_levels()));
    const auto rm =
        solvers::projected_pcg_solve(c, A, mg, db.span(), xm.span(), mg_cfg, projector, wm);
    auto hi = download(c, xi), hm = download(c, xm);
    const double b_norm = rms(b);
    const double id_rel = rms(residual_cpu(q, b, hi, n)) / b_norm;
    const double mg_rel = rms(residual_cpu(q, b, hm, n)) / b_norm;
    std::vector<real> d(hm.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        d[i] = hm[i] - hi[i];
    d = project(std::move(d));
    SolveMetrics z{ri.iterations,
                   rm.iterations,
                   id_rel,
                   mg_rel,
                   rms(d),
                   ri.converged && rm.converged && id_rel <= kResidualLimit &&
                       mg_rel <= kResidualLimit && rm.iterations <= 100 &&
                       rms(d) <= kSolverTolerance &&
                       rm.iterations < ri.iterations};
    return z;
}

[[nodiscard]] CaseResult pcg_case(bool smooth) {
    auto a = solve_control(32, smooth), b = solve_control(64, smooth);
    const double growth = static_cast<double>(b.mg_iter) / a.mg_iter;
    const bool pass = a.pass && b.pass && growth <= 1.5;
    const char* name = smooth ? "mg_projected_pcg_smooth_suite" : "mg_projected_pcg_constant_suite";
    std::cout << std::setprecision(12) << "mg_suite case=" << name
              << " q=" << (smooth ? "smooth" : "constant") << " N32_identity_iters=" << a.id_iter
              << " N32_mg_iters=" << a.mg_iter << " N32_id_relres=" << a.id_res
              << " N32_mg_relres=" << a.mg_res << " N64_identity_iters=" << b.id_iter
              << " N64_mg_iters=" << b.mg_iter << " N64_id_relres=" << b.id_res
              << " N64_mg_relres=" << b.mg_res << " growth=" << growth
              << " solution_rms=" << std::max(a.solution, b.solution)
              << " solution_rms_limit=" << kSolverTolerance << '\n';
    return {pass,
            name,
            "gpu-mg-reuse",
            "32^3->64^3",
            a.mg_res,
            b.mg_res,
            "relres<=1e-10; iters<=100; growth<=1.5; solution_rms<=1e-12",
            std::to_string(growth),
            "true CPU residual, reference agreement, and Identity reduction"};
}

[[nodiscard]] CaseResult flow_case() {
    // Direct solve_head CG/PCG comparison exposed a pre-existing legacy
    // disagreement (recorded in the SF-05 report), so this narrow control
    // guards the unmodified flow smoother contract without making an
    // acceptance claim about that separate legacy solver path.
    constexpr int n = 16;
    CudaContext c(0);
    Grid3D g(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n);
    auto K = q_field(n), b = manufactured(n), initial = manufactured(n, false);
    DeviceBuffer<real> dK(K.size()), db(b.size()), default_x(initial.size()),
        ordered_x(initial.size());
    upload(c, dK, K);
    upload(c, db, b);
    upload(c, default_x, initial);
    upload(c, ordered_x, initial);
    BCSpec bc;
    const BCFace periodic{BCType::Periodic, real(0)};
    bc.xmin = bc.xmax = bc.ymin = bc.ymax = bc.zmin = bc.zmax = periodic;
    multigrid::gsrb_smooth_3d(c, g, default_x.span(), db.span(), dK.span(), 3, bc, PinSpec{false});
    multigrid::gsrb_smooth_3d_ordered(c, g, ordered_x.span(), db.span(), dK.span(), 3, bc,
                                      multigrid::GSRBColorOrder::RedBlack, PinSpec{false});
    auto actual = download(c, default_x), expected = download(c, ordered_x);
    std::vector<real> difference(actual.size());
    for (std::size_t i = 0; i < difference.size(); ++i)
        difference[i] = actual[i] - expected[i];
    const double error = rms(difference);
    const bool pass = error == 0.0;
    std::cout << std::setprecision(12) << "mg_suite case=mg_legacy_flow_control grid=16^3 "
              << "default_order=red_black ordered_red_black_rms=" << error << '\n';
    return {pass,
            "mg_legacy_flow_control",
            "legacy-flow-regression",
            "16^3",
            error,
            0.0,
            "exact default=red_black",
            std::to_string(error),
            "default flow GSRB order/result preserved"};
}
} // namespace
CaseRegistry multigrid_reuse_case_registry() {
    return {{"mg_hierarchy_q_geometric_coarsening", hierarchy_case},
            {"mg_projected_preconditioner_contract", preconditioner_contract_case},
            {"mg_projected_pcg_constant_suite", []() { return pcg_case(false); }},
            {"mg_projected_pcg_smooth_suite", []() { return pcg_case(true); }},
            {"mg_legacy_flow_control", flow_case}};
}
} // namespace macroflow3d::streamfunctions::test
