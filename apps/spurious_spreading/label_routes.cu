/**
 * @file label_routes.cu
 * @brief SF-32 N2a: label routes of the `spurious_spreading` instrument
 *        (see label_routes.hpp for the contract).
 *
 * The "stack" route calls the FROZEN periodic streamfunction stack through
 * its public API only, exactly as `apps/closure_gate/ev_ladder_main.cu`
 * (SF-30 N3) does for one stage at lambda = 1: analytic Y = eps f
 * (closure_fields.hpp) or Y = 0 (homogeneous); SF-19
 * `solve_affine_periodic_flow` with qbar = e1 on K = exp(Y) as the Darcy
 * velocity; problem view with the log-conductivity representation, triply
 * periodic BCSpec and AffineGauge::benchmark(1); `PicardInitialState::
 * zero_source`, `CoefficientState::rebuild`; one `solve_streamfunctions`
 * call. Nothing under src/physics/streamfunctions/ is modified. The
 * report-to-JSON helpers below (status/exit/pcg labels, j_pcg, j_mg,
 * j_linear, j_flow_config, j_solver_config) are copied verbatim from
 * ev_ladder_main.cu (a .cu cannot be included).
 */

#include "apps/closure_gate/closure_fields.hpp"
#include "apps/spurious_spreading/analytic_pair_g.hpp"
#include "apps/spurious_spreading/label_routes.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/streamfunctions/Diagnostics.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/runtime/cuda_check.cuh"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace spurious_spreading {

using namespace macroflow3d;
namespace sf = macroflow3d::streamfunctions;
namespace st = macroflow3d::physics::particles::streamline_tracker;

// ===========================================================================
// CLI helpers
// ===========================================================================

double parse_double_arg(const std::string& name, const std::string& v) {
    std::size_t pos = 0;
    double d = 0.0;
    try {
        d = std::stod(v, &pos);
    } catch (const std::exception&) {
        throw UsageError("invalid value for " + name + ": '" + v + "'");
    }
    if (pos != v.size() || !std::isfinite(d)) {
        throw UsageError("invalid value for " + name + ": '" + v + "'");
    }
    return d;
}

long long parse_int_arg(const std::string& name, const std::string& v) {
    std::size_t pos = 0;
    long long i = 0;
    try {
        i = std::stoll(v, &pos);
    } catch (const std::exception&) {
        throw UsageError("invalid integer for " + name + ": '" + v + "'");
    }
    if (pos != v.size()) {
        throw UsageError("invalid integer for " + name + ": '" + v + "'");
    }
    return i;
}

unsigned long long parse_u64_arg(const std::string& name, const std::string& v) {
    std::size_t pos = 0;
    unsigned long long u = 0;
    if (v.empty() || v[0] == '-') {
        throw UsageError("invalid unsigned integer for " + name + ": '" + v + "'");
    }
    try {
        u = std::stoull(v, &pos);
    } catch (const std::exception&) {
        throw UsageError("invalid unsigned integer for " + name + ": '" + v + "'");
    }
    if (pos != v.size()) {
        throw UsageError("invalid unsigned integer for " + name + ": '" + v + "'");
    }
    return u;
}

bool parse_bool01_arg(const std::string& name, const std::string& v) {
    if (v == "1")
        return true;
    if (v == "0")
        return false;
    throw UsageError(name + " expects 0 or 1, got '" + v + "'");
}

bool valid_label_grid_n(long long n) {
    if (n < 4 || n > 1024)
        return false;
    return (n & (n - 1)) == 0;
}

// ===========================================================================
// Files
// ===========================================================================

LabelFiles label_files(const std::string& prefix) {
    return LabelFiles{prefix + ".json", prefix + "_u1.bin", prefix + "_u2.bin"};
}

namespace {

std::string basename_of(const std::string& path) {
    const std::size_t s = path.find_last_of('/');
    return s == std::string::npos ? path : path.substr(s + 1);
}

void require_little_endian() {
    const uint32_t probe = 0x01020304u;
    unsigned char b[4];
    std::memcpy(b, &probe, 4);
    if (b[0] != 0x04) {
        throw std::runtime_error("labels: the .bin format is little-endian; this host is not");
    }
}

void write_raw(const std::string& path, const std::vector<double>& v) {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
        throw std::runtime_error("labels: cannot open '" + path + "' for writing");
    os.write(reinterpret_cast<const char*>(v.data()),
             static_cast<std::streamsize>(v.size() * sizeof(double)));
    if (!os)
        throw std::runtime_error("labels: failed writing '" + path + "'");
}

std::vector<double> read_raw(const std::string& path, std::size_t expected) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is)
        throw std::runtime_error("labels: cannot open '" + path + "'");
    const std::streamoff bytes = is.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) != expected * sizeof(double)) {
        throw std::runtime_error("labels: '" + path + "' has " + std::to_string(bytes) +
                                 " bytes, expected " + std::to_string(expected * sizeof(double)));
    }
    is.seekg(0);
    std::vector<double> v(expected);
    is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    if (!is)
        throw std::runtime_error("labels: failed reading '" + path + "'");
    return v;
}

void read_vec3(const JVal& meta, const char* key, real out[3]) {
    const JVal* a = meta.find(key);
    if (a == nullptr || !a->is_array() || a->size() != 3) {
        throw std::runtime_error(std::string("labels: metadata key '") + key +
                                 "' missing or not a 3-array");
    }
    for (int d = 0; d < 3; ++d) {
        if (!a->at(d).is_number())
            throw std::runtime_error(std::string("labels: metadata '") + key + "' not numeric");
        out[d] = static_cast<real>(a->at(d).as_double());
        if (!std::isfinite(out[d]))
            throw std::runtime_error(std::string("labels: metadata '") + key + "' not finite");
    }
}

} // namespace

void mkdir_p(const std::string& path) {
    std::string cur;
    std::size_t pos = 0;
    while (pos <= path.size()) {
        const std::size_t next = path.find('/', pos);
        const std::string part =
            path.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        cur += part;
        if (!part.empty() && part != "." && part != "..") {
            if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("cannot create directory '" + cur +
                                         "': " + std::strerror(errno));
            }
        }
        if (next == std::string::npos)
            break;
        cur += '/';
        pos = next + 1;
    }
    struct stat sb;
    if (::stat(path.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
        throw std::runtime_error("'" + path + "' is not a directory");
}

JVal label_files_json(const std::string& prefix) {
    const LabelFiles f = label_files(prefix);
    return jobj({{"u1", basename_of(f.u1)},
                 {"u2", basename_of(f.u2)},
                 {"dtype", "float64 little-endian"},
                 {"layout", "i + n*(j + n*k), cell centres (i+1/2)h"}});
}

void write_labels(const std::string& prefix, int n, const std::vector<double>& u1,
                  const std::vector<double>& u2, const JVal& meta) {
    require_little_endian();
    const std::size_t cells = static_cast<std::size_t>(n) * n * n;
    if (u1.size() != cells || u2.size() != cells) {
        throw std::runtime_error("labels: write_labels: array size != n^3");
    }
    const std::size_t slash = prefix.find_last_of('/');
    if (slash != std::string::npos && slash > 0)
        mkdir_p(prefix.substr(0, slash));
    const LabelFiles f = label_files(prefix);
    write_raw(f.u1, u1);
    write_raw(f.u2, u2);
    // Round-trip check: what is on disk must be bitwise what was meant to be written.
    const std::vector<double> r1 = read_raw(f.u1, cells);
    const std::vector<double> r2 = read_raw(f.u2, cells);
    if (std::memcmp(r1.data(), u1.data(), cells * sizeof(double)) != 0 ||
        std::memcmp(r2.data(), u2.data(), cells * sizeof(double)) != 0) {
        throw std::runtime_error("labels: bitwise round trip of the .bin files failed");
    }
    std::ofstream os(f.json, std::ios::trunc);
    if (!os)
        throw std::runtime_error("labels: cannot open '" + f.json + "' for writing");
    os << meta.dump(2) << '\n';
    if (!os)
        throw std::runtime_error("labels: failed writing '" + f.json + "'");
}

LoadedLabels read_labels(const std::string& prefix) {
    require_little_endian();
    const LabelFiles f = label_files(prefix);
    std::ifstream is(f.json);
    if (!is)
        throw std::runtime_error("labels: cannot open '" + f.json + "'");
    std::stringstream ss;
    ss << is.rdbuf();
    LoadedLabels L;
    L.meta = parse_json(ss.str());
    if (!L.meta.is_object())
        throw std::runtime_error("labels: '" + f.json + "' is not a JSON object");
    const JVal* jn = L.meta.find("n");
    const JVal* jh = L.meta.find("h");
    if (jn == nullptr || jn->kind() != JVal::Kind::signed_int)
        throw std::runtime_error("labels: metadata key 'n' missing or not an integer");
    if (jh == nullptr || !jh->is_number())
        throw std::runtime_error("labels: metadata key 'h' missing or not a number");
    const long long n = jn->as_int();
    if (!valid_label_grid_n(n))
        throw std::runtime_error("labels: metadata 'n' must be a power of two in [4, 1024]");
    L.n = static_cast<int>(n);
    L.h = jh->as_double();
    if (L.h != 1.0 / static_cast<double>(n))
        throw std::runtime_error("labels: metadata 'h' != 1/n");
    read_vec3(L.meta, "gbar1", L.gbar1);
    read_vec3(L.meta, "gbar2", L.gbar2);
    const std::size_t cells = static_cast<std::size_t>(n) * n * n;
    L.u1 = read_raw(f.u1, cells);
    L.u2 = read_raw(f.u2, cells);
    return L;
}

// ===========================================================================
// Splines and min |c| on the grid
// ===========================================================================

namespace {

constexpr int kBlock = 256;

__global__ void min_abs_c_cells_kernel(st::SplineLabelPair labels, int n, real h, real* out) {
    const long long cells = static_cast<long long>(n) * n * n;
    const long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= cells)
        return;
    const int i = static_cast<int>(c % n);
    const int j = static_cast<int>((c / n) % n);
    const int k = static_cast<int>(c / (static_cast<long long>(n) * n));
    const real xi[3] = {(static_cast<real>(i) + static_cast<real>(0.5)) * h,
                        (static_cast<real>(j) + static_cast<real>(0.5)) * h,
                        (static_cast<real>(k) + static_cast<real>(0.5)) * h};
    const int32_t w[3] = {0, 0, 0};
    st::LabelSample s;
    labels(xi, w, s);
    real cv[3];
    st::cross3(s.g1, s.g2, cv);
    out[c] = st::norm3(cv);
}

} // namespace

void SplineLabels::build(const CudaContext& ctx, int n, const std::vector<double>& u1,
                         const std::vector<double>& u2, const real gbar1[3], const real gbar2[3]) {
    if (!valid_label_grid_n(n))
        throw std::invalid_argument("SplineLabels: n must be a power of two in [4, 1024]");
    const std::size_t cells = static_cast<std::size_t>(n) * n * n;
    if (u1.size() != cells || u2.size() != cells)
        throw std::invalid_argument("SplineLabels: array size != n^3");
    n_ = n;
    h_ = 1.0 / static_cast<double>(n);
    const Grid3D grid(n, n, n, h_, h_, h_);
    DeviceBuffer<real> d1(cells), d2(cells);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(d1.data(), u1.data(), cells * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(d2.data(), u2.data(), cells * sizeof(real), cudaMemcpyHostToDevice));
    interpolation::prefilter_periodic_tricubic_bspline(
        ctx, grid, DeviceSpan<const real>(d1.data(), cells), ws1_);
    interpolation::prefilter_periodic_tricubic_bspline(
        ctx, grid, DeviceSpan<const real>(d2.data(), cells), ws2_);
    ctx.synchronize();
    pair_ = st::make_spline_label_pair(ws1_.view(), ws2_.view(), gbar1, gbar2);
}

double SplineLabels::min_abs_c_grid(const CudaContext& ctx) const {
    if (n_ == 0)
        throw std::logic_error("SplineLabels::min_abs_c_grid before build");
    const std::size_t cells = static_cast<std::size_t>(n_) * n_ * n_;
    DeviceBuffer<real> d(cells);
    const unsigned blocks = static_cast<unsigned>((cells + kBlock - 1) / kBlock);
    min_abs_c_cells_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(
        pair_, n_, static_cast<real>(h_), d.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
    std::vector<real> hv(cells);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(hv.data(), d.data(), cells * sizeof(real), cudaMemcpyDeviceToHost));
    double m = std::numeric_limits<double>::infinity();
    bool any_nan = false;
    for (std::size_t c = 0; c < cells; ++c) {
        const double v = static_cast<double>(hv[c]);
        if (std::isnan(v))
            any_nan = true;
        else if (v < m)
            m = v;
    }
    // A non-finite |c| anywhere makes the pair unusable: report NaN (fails the > 0.5 guard).
    return any_nan ? std::numeric_limits<double>::quiet_NaN() : m;
}

// ===========================================================================
// solve-labels (frozen stack; ev_ladder recipe)
// ===========================================================================

namespace {

using json = JVal;

const char* status_label(sf::StreamfunctionSolveStatus s) {
    switch (s) {
    case sf::StreamfunctionSolveStatus::not_run:
        return "not_run";
    case sf::StreamfunctionSolveStatus::converged:
        return "converged";
    case sf::StreamfunctionSolveStatus::not_converged:
        return "not_converged";
    case sf::StreamfunctionSolveStatus::invalid_problem:
        return "invalid_problem";
    }
    return "unknown";
}

const char* exit_label(sf::PicardExitReason r) {
    switch (r) {
    case sf::PicardExitReason::none:
        return "none";
    case sf::PicardExitReason::converged:
        return "converged";
    case sf::PicardExitReason::budget_exhausted:
        return "budget_exhausted";
    case sf::PicardExitReason::linear_block_failure:
        return "linear_block_failure";
    case sf::PicardExitReason::stagnated:
        return "stagnated";
    case sf::PicardExitReason::omega_floor_rejected:
        return "omega_floor_rejected";
    case sf::PicardExitReason::newton_exhausted:
        return "newton_exhausted";
    case sf::PicardExitReason::newton_budget_exhausted:
        return "newton_budget_exhausted";
    }
    return "unknown";
}

const char* pcg_label(solvers::ProjectedPCGStatus s) {
    switch (s) {
    case solvers::ProjectedPCGStatus::converged:
        return "converged";
    case solvers::ProjectedPCGStatus::max_iterations:
        return "max_iterations";
    case solvers::ProjectedPCGStatus::invalid_configuration:
        return "invalid_configuration";
    case solvers::ProjectedPCGStatus::size_mismatch:
        return "size_mismatch";
    case solvers::ProjectedPCGStatus::aliasing:
        return "aliasing";
    case solvers::ProjectedPCGStatus::breakdown_pAp:
        return "breakdown_pAp";
    case solvers::ProjectedPCGStatus::breakdown_rz:
        return "breakdown_rz";
    case solvers::ProjectedPCGStatus::nonfinite_value:
        return "nonfinite_value";
    }
    return "unknown";
}
json j_pcg(const solvers::ProjectedPCGResult& r) {
    json j;
    j["status"] = pcg_label(r.status);
    j["converged"] = r.converged;
    j["iterations"] = r.iterations;
    j["raw_rhs_mean"] = static_cast<double>(r.raw_rhs_mean);
    j["raw_rhs_l2_norm"] = static_cast<double>(r.raw_rhs_l2_norm);
    j["raw_rhs_compatibility_defect"] = static_cast<double>(r.raw_rhs_compatibility_defect);
    j["initial_projected_residual"] = static_cast<double>(r.initial_projected_residual);
    j["final_projected_residual"] = static_cast<double>(r.final_projected_residual);
    j["relative_projected_residual"] = static_cast<double>(r.relative_projected_residual);
    j["final_field_mean"] = static_cast<double>(r.final_field_mean);
    return j;
}

json j_mg(const multigrid::MGConfig& m) {
    json j;
    j["num_levels"] = m.num_levels;
    j["pre_smooth"] = m.pre_smooth;
    j["post_smooth"] = m.post_smooth;
    j["coarse_solve_iters"] = m.coarse_solve_iters;
    j["check_convergence_every"] = m.check_convergence_every;
    j["omega"] = static_cast<double>(m.omega);
    return j;
}

json j_linear(const solvers::ProjectedPCGConfig& l) {
    json j;
    j["rtol"] = static_cast<double>(l.rtol);
    j["max_iter"] = l.max_iter;
    j["check_every"] = l.check_every;
    return j;
}

json j_flow_config(const physics::AffinePeriodicFlowConfig& c) {
    json j;
    j["qbar"] = json::array({static_cast<double>(c.qbar[0]), static_cast<double>(c.qbar[1]),
                             static_cast<double>(c.qbar[2])});
    j["linear"] = j_linear(c.linear);
    j["mg"] = j_mg(c.mg);
    return j;
}

json j_flow_report(const physics::AffinePeriodicFlowReport& f) {
    json j;
    json keff = json::array();
    for (int i = 0; i < 3; ++i) {
        keff.push_back(
            json::array({static_cast<double>(f.K_eff[i][0]), static_cast<double>(f.K_eff[i][1]),
                         static_cast<double>(f.K_eff[i][2])}));
    }
    j["K_eff"] = keff;
    j["symmetry_defect_rel"] = static_cast<double>(f.symmetry_defect_rel);
    j["eigenvalues_symmetric_part"] =
        json::array({static_cast<double>(f.eigenvalues_symmetric_part[0]),
                     static_cast<double>(f.eigenvalues_symmetric_part[1]),
                     static_cast<double>(f.eigenvalues_symmetric_part[2])});
    j["G"] = json::array(
        {static_cast<double>(f.G[0]), static_cast<double>(f.G[1]), static_cast<double>(f.G[2])});
    j["achieved_mean_flux"] = json::array({static_cast<double>(f.achieved_mean_flux[0]),
                                           static_cast<double>(f.achieved_mean_flux[1]),
                                           static_cast<double>(f.achieved_mean_flux[2])});
    j["div_max_abs"] = static_cast<double>(f.div_max_abs);
    j["div_rms"] = static_cast<double>(f.div_rms);
    j["corrector_results"] =
        json::array({j_pcg(f.corrector_results[0]), j_pcg(f.corrector_results[1]),
                     j_pcg(f.corrector_results[2])});
    j["memory_total_bytes"] = f.memory.total_bytes;
    return j;
}

json j_solver_config(const sf::StreamfunctionSolverConfig& c) {
    json j;
    j["picard"] = jobj({{"max_iter", c.picard.max_iter},
                        {"tolerance", static_cast<double>(c.picard.tolerance)},
                        {"omega", static_cast<double>(c.picard.omega)}});
    const auto& a = c.adaptive;
    j["adaptive"] =
        jobj({{"enabled", a.enabled},
              {"omega_min", static_cast<double>(a.omega_min)},
              {"backtrack_factor", static_cast<double>(a.backtrack_factor)},
              {"growth_factor", static_cast<double>(a.growth_factor)},
              {"omega_max", static_cast<double>(a.omega_max)},
              {"easy_streak", a.easy_streak},
              {"armijo_c", static_cast<double>(a.armijo_c)},
              {"stagnation_window", a.stagnation_window},
              {"stagnation_min_reduction", static_cast<double>(a.stagnation_min_reduction)},
              {"max_unexplained_fraction", static_cast<double>(a.max_unexplained_fraction)},
              {"unexplained_growth_factor", static_cast<double>(a.unexplained_growth_factor)},
              {"unexplained_growth_offset", static_cast<double>(a.unexplained_growth_offset)},
              {"percentile_collapse_factor", static_cast<double>(a.percentile_collapse_factor)},
              {"floor_guard", jobj({{"enabled", a.floor_guard.enabled},
                                    {"window", a.floor_guard.window},
                                    {"drop_factor", static_cast<double>(a.floor_guard.drop_factor)},
                                    {"max_resets", a.floor_guard.max_resets}})}});
    const auto& an = c.anderson;
    j["anderson"] = jobj({{"enabled", an.enabled},
                          {"depth", an.depth},
                          {"start_iteration", an.start_iteration},
                          {"condition_limit", static_cast<double>(an.condition_limit)},
                          {"restart_on_stagnation", an.restart_on_stagnation},
                          {"max_restarts", an.max_restarts}});
    const auto& nw = c.newton;
    j["newton"] =
        jobj({{"enabled", nw.enabled},
              {"activation_r_F", static_cast<double>(nw.activation_r_F)},
              {"stagnation_activation_r_F", static_cast<double>(nw.stagnation_activation_r_F)},
              {"forcing_coefficient", static_cast<double>(nw.forcing_coefficient)},
              {"forcing_min", static_cast<double>(nw.forcing_min)},
              {"forcing_max", static_cast<double>(nw.forcing_max)},
              {"armijo_c", static_cast<double>(nw.armijo_c)},
              {"alpha_min", static_cast<double>(nw.alpha_min)},
              {"backtrack_factor", static_cast<double>(nw.backtrack_factor)},
              {"max_newton_iterations", nw.max_newton_iterations},
              {"rescue_picard_steps", nw.rescue_picard_steps},
              {"gmres", jobj({{"restart", nw.gmres.restart},
                              {"max_iterations", nw.gmres.max_iterations},
                              {"rel_tol", static_cast<double>(nw.gmres.rel_tol)}})},
              {"delta", jobj({{"delta_min", static_cast<double>(nw.delta.delta_min)},
                              {"delta_max", static_cast<double>(nw.delta.delta_max)}})},
              {"rescue_resets_omega", nw.rescue_resets_omega}});
    j["eta"] = static_cast<double>(c.eta);
    j["epsilon"] = static_cast<double>(c.epsilon);
    j["linear"] = j_linear(c.linear);
    j["mg"] = j_mg(c.mg);
    j["histogram"] = jobj({{"c_min_rel", static_cast<double>(c.histogram.c_min_rel)},
                           {"c_max_rel", static_cast<double>(c.histogram.c_max_rel)}});
    json thr = json::array();
    for (int t = 0; t < c.diagnostics.num_degeneracy_thresholds; ++t) {
        thr.push_back(static_cast<double>(c.diagnostics.degeneracy_thresholds[t]));
    }
    j["diagnostics"] =
        jobj({{"angle_exclusion_rel", static_cast<double>(c.diagnostics.angle_exclusion_rel)},
              {"low_speed_rel", static_cast<double>(c.diagnostics.low_speed_rel)},
              {"num_degeneracy_thresholds", c.diagnostics.num_degeneracy_thresholds},
              {"degeneracy_thresholds", thr}});
    json sthr = json::array();
    for (int t = 0; t < c.num_degeneracy_thresholds; ++t) {
        sthr.push_back(static_cast<double>(c.degeneracy_thresholds[t]));
    }
    j["source_num_degeneracy_thresholds"] = c.num_degeneracy_thresholds;
    j["source_degeneracy_thresholds"] = sthr;
    j["initial_state"] =
        c.initial_state == sf::PicardInitialState::zero_source ? "zero_source" : "warm_start";
    j["coefficient_state"] =
        c.coefficient_state == sf::CoefficientState::rebuild ? "rebuild" : "reuse";
    return j;
}

// MG depth rule for `--mg-levels auto` (copied from ev_ladder_main.cu): halve
// while the extent stays even and the next extent is >= 4 and even.
int auto_mg_levels(int n) {
    int levels = 1;
    int m = n;
    while (m % 2 == 0 && (m / 2) >= 4 && ((m / 2) % 2) == 0) {
        m /= 2;
        ++levels;
    }
    return levels;
}

struct SolveOptions {
    std::string field;
    int n = 0;
    double eps = 0.25;
    double epsilon = 1e-6;
    double tolerance = 1e-8;
    int max_iter = 1000;
    bool anderson = true;
    bool newton = false;
    double pcg_rtol = 1e-10;
    bool mg_levels_auto = true;
    int mg_levels = 0;
    bool no_timing = false;
    std::string out;
};

SolveOptions parse_solve_options(int argc, char** argv) {
    SolveOptions o;
    bool have_field = false, have_n = false, have_out = false;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (key == "--no-timing") {
            o.no_timing = true;
            continue;
        }
        if (a + 1 >= argc)
            throw UsageError("missing value for " + key);
        const std::string val = argv[++a];
        if (key == "--field") {
            o.field = val;
            have_field = true;
        } else if (key == "--n") {
            const long long n = parse_int_arg(key, val);
            if (!valid_label_grid_n(n))
                throw UsageError("--n must be a power of two in [4, 1024]");
            o.n = static_cast<int>(n);
            have_n = true;
        } else if (key == "--eps") {
            o.eps = parse_double_arg(key, val);
        } else if (key == "--epsilon") {
            o.epsilon = parse_double_arg(key, val);
        } else if (key == "--tolerance") {
            o.tolerance = parse_double_arg(key, val);
        } else if (key == "--max-iter") {
            const long long m = parse_int_arg(key, val);
            if (m < 0 || m > 1000000)
                throw UsageError("--max-iter must be in [0, 1e6]");
            o.max_iter = static_cast<int>(m);
        } else if (key == "--anderson") {
            o.anderson = parse_bool01_arg(key, val);
        } else if (key == "--newton") {
            o.newton = parse_bool01_arg(key, val);
        } else if (key == "--pcg-rtol") {
            o.pcg_rtol = parse_double_arg(key, val);
        } else if (key == "--mg-levels") {
            if (val == "auto") {
                o.mg_levels_auto = true;
            } else {
                const long long l = parse_int_arg(key, val);
                if (l < 1 || l > 32)
                    throw UsageError("--mg-levels must be auto or in [1, 32]");
                o.mg_levels_auto = false;
                o.mg_levels = static_cast<int>(l);
            }
        } else if (key == "--out") {
            o.out = val;
            have_out = true;
        } else {
            throw UsageError("unknown option " + key);
        }
    }
    if (!have_field)
        throw UsageError("solve-labels: --field is required");
    if (!have_n)
        throw UsageError("solve-labels: --n is required");
    if (!have_out || o.out.empty())
        throw UsageError("solve-labels: --out <prefix> is required");
    const bool known = o.field == "homogeneous" || o.field == "lester2021" ||
                       o.field == "lester_brk" || o.field == "control2d" || o.field == "two_mode" ||
                       o.field == "generic3d";
    if (!known)
        throw UsageError("solve-labels: unknown --field '" + o.field + "'");
    return o;
}

JVal gbar_json(const real g[3]) {
    return JVal::array(
        {static_cast<double>(g[0]), static_cast<double>(g[1]), static_cast<double>(g[2])});
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int run_solve_labels(int argc, char** argv) {
    const SolveOptions o = parse_solve_options(argc, argv);
    const int N = o.n;
    const real h = real{1} / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    const std::size_t n = grid.num_cells();
    const int mg_levels = o.mg_levels_auto ? auto_mg_levels(N) : o.mg_levels;
    const real gbar1[3] = {0.0, 1.0, 0.0};
    const real gbar2[3] = {0.0, 0.0, 1.0};

    CudaContext ctx(0);
    const auto t_total0 = std::chrono::steady_clock::now();

    // ---- 1. Y on the host ----
    std::vector<double> y_host;
    if (o.field == "homogeneous") {
        y_host.assign(n, 0.0);
    } else {
        const closure_gate::AnalyticField af = closure_gate::analytic_field_from_name(o.field);
        closure_gate::fill_analytic_log_conductivity(grid, af, o.eps, y_host);
    }
    double y_mean = 0.0, y_min = y_host[0], y_max = y_host[0];
    for (double v : y_host) {
        y_mean += v;
        y_min = std::min(y_min, v);
        y_max = std::max(y_max, v);
    }
    y_mean /= static_cast<double>(n);
    double y_var = 0.0;
    for (double v : y_host)
        y_var += (v - y_mean) * (v - y_mean);
    y_var /= static_cast<double>(n);

    // ---- configuration (ev_ladder, one stage at lambda = 1) ----
    sf::StreamfunctionSolverConfig cfg{};
    cfg.picard.max_iter = o.max_iter;
    cfg.picard.tolerance = static_cast<real>(o.tolerance);
    cfg.adaptive.enabled = true;
    cfg.anderson.enabled = o.anderson;
    cfg.anderson.depth = 5;
    cfg.anderson.start_iteration = 5;
    cfg.anderson.condition_limit = real{1e12};
    cfg.newton.enabled = o.newton;
    cfg.eta = real{1};
    cfg.epsilon = static_cast<real>(o.epsilon);
    cfg.mg.num_levels = mg_levels;
    cfg.initial_state = sf::PicardInitialState::zero_source;
    cfg.coefficient_state = sf::CoefficientState::rebuild;

    physics::AffinePeriodicFlowConfig flow_cfg{};
    flow_cfg.qbar[0] = real{1};
    flow_cfg.qbar[1] = real{0};
    flow_cfg.qbar[2] = real{0};
    flow_cfg.linear.rtol = static_cast<real>(o.pcg_rtol);
    flow_cfg.mg.num_levels = mg_levels;

    std::printf("==== spurious_spreading solve-labels (SF-32 N2a; frozen periodic stack) ====\n");
    std::printf("field = %s  eps = %.17g  N = %d  h = %.17g  mg_levels = %d (%s)\n",
                o.field.c_str(), o.eps, N, static_cast<double>(h), mg_levels,
                o.mg_levels_auto ? "auto" : "explicit");
    std::printf("  Y: mean = %.17g  var = %.17g  min = %.17g  max = %.17g\n", y_mean, y_var, y_min,
                y_max);
    std::printf("config: eta = 1  epsilon = %g  tolerance = %g  max_iter = %d  anderson = %d  "
                "newton = %d  darcy pcg rtol = %g\n",
                o.epsilon, o.tolerance, o.max_iter, o.anderson ? 1 : 0, o.newton ? 1 : 0,
                o.pcg_rtol);

    const std::size_t nu = static_cast<std::size_t>(N + 1) * N * N;
    const std::size_t nv = static_cast<std::size_t>(N) * (N + 1) * N;
    const std::size_t nw = static_cast<std::size_t>(N) * N * (N + 1);
    DeviceBuffer<real> y_dev(n), k_dev(n);
    DeviceBuffer<real> flow_u(nu), flow_v(nv), flow_w(nw);
    {
        std::vector<double> k_host(n);
        for (std::size_t c = 0; c < n; ++c)
            k_host[c] = std::exp(y_host[c]);
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(y_dev.data(), y_host.data(), n * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(k_dev.data(), k_host.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    }

    // ---- 2. Darcy reference (SF-19, qbar = e1) ----
    physics::AffinePeriodicFlowWorkspace flow_ws;
    const auto t0 = std::chrono::steady_clock::now();
    const physics::AffinePeriodicFlowReport flow = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(k_dev.span()), flow_cfg,
        physics::AffinePeriodicVelocityView{flow_u.span(), flow_v.span(), flow_w.span()}, flow_ws);
    ctx.synchronize();
    const double wall_darcy = seconds_since(t0);
    const bool darcy_ok = flow.corrector_results[0].converged &&
                          flow.corrector_results[1].converged &&
                          flow.corrector_results[2].converged;
    std::printf("darcy: achieved mean flux = (%.17g, %.17g, %.17g)  div_max_abs = %.3e  wall = "
                "%.3f s\n",
                static_cast<double>(flow.achieved_mean_flux[0]),
                static_cast<double>(flow.achieved_mean_flux[1]),
                static_cast<double>(flow.achieved_mean_flux[2]),
                static_cast<double>(flow.div_max_abs), wall_darcy);
    for (int d = 0; d < 3; ++d) {
        const auto& c = flow.corrector_results[d];
        std::printf("  corrector %c: %s iters=%d rel_res=%.3e\n", "xyz"[d], pcg_label(c.status),
                    c.iterations, static_cast<double>(c.relative_projected_residual));
    }
    if (!darcy_ok) {
        std::fprintf(stderr, "solve-labels: a Darcy corrector PCG did not converge; no labels "
                             "written\n");
        return 4;
    }

    // ---- 3. problem view and one solve: whatever it leaves is the pair ----
    sf::StreamfunctionFields fields;
    sf::StreamfunctionWorkspace workspace;
    fields.prepare(grid);
    workspace.prepare(grid, cfg);

    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    const sf::AffineGauge gauge = sf::AffineGauge::benchmark(real{1});

    sf::StreamfunctionProblemView view;
    view.grid = grid;
    view.conductivity = DeviceSpan<const real>(y_dev.span());
    view.conductivity_representation = sf::ConductivityRepresentation::log_conductivity_y;
    view.darcy_velocity = sf::CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                          DeviceSpan<const real>(flow_v.span()),
                                                          DeviceSpan<const real>(flow_w.span())};
    view.bc = bc;
    view.gauge = gauge;

    const auto t2 = std::chrono::steady_clock::now();
    const sf::StreamfunctionSolveReport rep =
        sf::solve_streamfunctions(ctx, view, cfg, fields, workspace);
    ctx.synchronize();
    const double wall_solve = seconds_since(t2);

    double pct[3];
    const double kPct[3] = {0.001, 0.01, 0.05};
    for (int p = 0; p < 3; ++p) {
        pct[p] = static_cast<double>(
            sf::residual_histogram_percentile(rep.residual, static_cast<real>(kPct[p])));
    }
    const double r_f_init = rep.picard_history.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : static_cast<double>(rep.picard_history.front().r_F);
    const double r_f_final = static_cast<double>(rep.residual.r_F);
    const bool tol_met = std::isfinite(r_f_final) && rep.residual.r_F <= cfg.picard.tolerance;
    const auto& dg = rep.diagnostics;

    // ---- 4. download the accepted state and spline it ----
    std::vector<double> u1(n), u2(n);
    {
        const sf::StreamfunctionFields& cf = fields;
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(u1.data(), cf.u1_span().data(), n * sizeof(real), cudaMemcpyDeviceToHost));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(u2.data(), cf.u2_span().data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    }
    std::size_t nonfinite = 0;
    for (std::size_t c = 0; c < n; ++c) {
        if (!std::isfinite(u1[c]) || !std::isfinite(u2[c]))
            ++nonfinite;
    }
    double min_c_grid = std::numeric_limits<double>::quiet_NaN();
    if (nonfinite == 0) {
        SplineLabels sl;
        sl.build(ctx, N, u1, u2, gbar1, gbar2);
        min_c_grid = sl.min_abs_c_grid(ctx);
    }
    const double wall_total = seconds_since(t_total0);

    std::printf("solve: status = %s  exit = %s  iterations = %d  wall = %.3f s\n",
                status_label(rep.status), exit_label(rep.exit_reason), rep.picard_iterations,
                wall_solve);
    std::printf("  r_F initial = %.17g\n  r_F final   = %.17g  tolerance = %g  tolerance_met = "
                "%s\n",
                r_f_init, r_f_final, o.tolerance, tol_met ? "true" : "false");
    std::printf("  e_v = %.17g  e_psi1 = %.6e  e_psi2 = %.6e  e_div = %.6e\n",
                static_cast<double>(dg.e_v), static_cast<double>(dg.invariance_e_psi1),
                static_cast<double>(dg.invariance_e_psi2), static_cast<double>(dg.e_div));
    std::printf("  |c| (report) min = %.6e  0.1%% = %.6e  1%% = %.6e  5%% = %.6e\n",
                static_cast<double>(dg.c_min), pct[0], pct[1], pct[2]);
    std::printf("  min_abs_c_grid (spline, cell centres) = %.17g%s\n", min_c_grid,
                nonfinite ? "  (NOT computed: non-finite fluctuation values)" : "");
    if (nonfinite)
        std::printf("  WARNING: %zu cells with non-finite u1/u2\n", nonfinite);
    if (!(min_c_grid > 0.5))
        std::printf("  WARNING: min_abs_c_grid <= 0.5: return-map will refuse this pair\n");

    JVal meta;
    meta["tool"] = "spurious_spreading solve-labels";
    meta["increment"] = "SF-32 N2a";
    meta["route"] = "stack";
    meta["field"] = o.field;
    meta["eps"] = o.eps;
    meta["n"] = N;
    meta["h"] = static_cast<double>(h);
    meta["vbar"] = 1.0;
    meta["gbar1"] = gbar_json(gbar1);
    meta["gbar2"] = gbar_json(gbar2);
    meta["exit_reason"] = exit_label(rep.exit_reason);
    meta["status"] = status_label(rep.status);
    meta["iterations"] = rep.picard_iterations;
    meta["r_F"] = r_f_final;
    meta["r_F_initial"] = r_f_init;
    meta["r_F_final"] = r_f_final;
    meta["tolerance"] = o.tolerance;
    meta["tolerance_met"] = tol_met;
    meta["e_v"] = static_cast<double>(dg.e_v);
    meta["e_psi1"] = static_cast<double>(dg.invariance_e_psi1);
    meta["e_psi2"] = static_cast<double>(dg.invariance_e_psi2);
    meta["e_div"] = static_cast<double>(dg.e_div);
    meta["abs_c"] = jobj({{"min", static_cast<double>(dg.c_min)},
                          {"p0.001", pct[0]},
                          {"p0.01", pct[1]},
                          {"p0.05", pct[2]}});
    meta["abs_c_source"] =
        "min: StreamfunctionSolveReport::diagnostics.c_min; percentiles: "
        "residual_histogram_percentile(report.residual, p) (histogram upper edges)";
    meta["min_abs_c_grid"] = min_c_grid;
    meta["min_abs_c_grid_source"] =
        "SF-28 spline pair, |grad psi1 x grad psi2| at the n^3 cell centres";
    meta["nonfinite_cells"] = nonfinite;
    meta["Y"] = jobj({{"mean", y_mean}, {"variance", y_var}, {"min", y_min}, {"max", y_max}});
    meta["darcy_pcg_converged"] = darcy_ok;
    meta["darcy"] = j_flow_report(flow);
    meta["wall_seconds_darcy"] = o.no_timing ? 0.0 : wall_darcy;
    meta["wall_seconds_solve"] = o.no_timing ? 0.0 : wall_solve;
    meta["wall_seconds"] = o.no_timing ? 0.0 : wall_total;
    meta["files"] = label_files_json(o.out);
    meta["config"] = jobj({{"field", o.field},
                           {"eps", o.eps},
                           {"n", o.n},
                           {"max_iter", o.max_iter},
                           {"tolerance", o.tolerance},
                           {"epsilon", o.epsilon},
                           {"anderson", o.anderson},
                           {"newton", o.newton},
                           {"pcg_rtol", o.pcg_rtol},
                           {"mg_levels", mg_levels},
                           {"mg_levels_mode", o.mg_levels_auto ? "auto" : "explicit"},
                           {"no_timing", o.no_timing},
                           {"out", o.out},
                           {"lambda", 1.0},
                           {"initial_state", "zero_source"},
                           {"coefficient_state", "rebuild"},
                           {"gauge", "AffineGauge::benchmark(1)"},
                           {"conductivity_representation", "log_conductivity_y"},
                           {"solver_config", j_solver_config(cfg)},
                           {"darcy_config", j_flow_config(flow_cfg)}});
    write_labels(o.out, N, u1, u2, meta);
    std::printf("labels written: %s{.json,_u1.bin,_u2.bin}  (total wall %.3f s)\n", o.out.c_str(),
                wall_total);
    return 0;
}

// ===========================================================================
// analytic-labels
// ===========================================================================

namespace {

struct AnalyticOptions {
    std::string pair;
    bool have_amplitude = false;
    double amplitude = 0.0;
    bool have_amplitude_b = false;
    double amplitude_b = 0.08;
    int n = 0;
    bool no_timing = false;
    std::string out;
};

AnalyticOptions parse_analytic_options(int argc, char** argv) {
    AnalyticOptions o;
    bool have_pair = false, have_n = false, have_out = false;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (key == "--no-timing") {
            o.no_timing = true;
            continue;
        }
        if (a + 1 >= argc)
            throw UsageError("missing value for " + key);
        const std::string val = argv[++a];
        if (key == "--pair") {
            o.pair = val;
            have_pair = true;
        } else if (key == "--amplitude") {
            o.amplitude = parse_double_arg(key, val);
            o.have_amplitude = true;
        } else if (key == "--amplitude-b") {
            o.amplitude_b = parse_double_arg(key, val);
            o.have_amplitude_b = true;
        } else if (key == "--n") {
            const long long n = parse_int_arg(key, val);
            if (!valid_label_grid_n(n))
                throw UsageError("--n must be a power of two in [4, 1024]");
            o.n = static_cast<int>(n);
            have_n = true;
        } else if (key == "--out") {
            o.out = val;
            have_out = true;
        } else {
            throw UsageError("unknown option " + key);
        }
    }
    if (!have_pair)
        throw UsageError("analytic-labels: --pair is required");
    if (!have_n)
        throw UsageError("analytic-labels: --n is required");
    if (!have_out || o.out.empty())
        throw UsageError("analytic-labels: --out <prefix> is required");
    AnalyticPair p;
    try {
        p = analytic_pair_from_name(o.pair);
    } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
    }
    if (!o.have_amplitude) {
        // Defaults: SF-32 control amplitude for G, SF-31 constants for A and B.
        o.amplitude = (p == AnalyticPair::G) ? 0.05 : (p == AnalyticPair::U ? 0.0 : 0.1);
    }
    if (p == AnalyticPair::U && o.amplitude != 0.0)
        throw UsageError("analytic-labels: pair U has no amplitude (use --amplitude 0 or omit it)");
    if (p != AnalyticPair::B && o.have_amplitude_b)
        throw UsageError("analytic-labels: --amplitude-b applies to pair B only");
    return o;
}

const char* pair_formula(AnalyticPair p) {
    switch (p) {
    case AnalyticPair::U:
        return "s1 = 0, s2 = 0";
    case AnalyticPair::A:
        return "s1 = a sin(2 pi x1), s2 = 0";
    case AnalyticPair::B:
        return "s1 = a sin(2 pi x1), s2 = b sin(2 pi x2)";
    case AnalyticPair::G:
        return "s1 = e [sin(2 pi x1) cos(2 pi x3) + 0.5 sin(2 pi (x2 + x3))], "
               "s2 = e [cos(2 pi x1) sin(2 pi x2) + 0.5 cos(2 pi (x1 - x3))]";
    }
    return "?";
}

} // namespace

int run_analytic_labels(int argc, char** argv) {
    const AnalyticOptions o = parse_analytic_options(argc, argv);
    const AnalyticPair p = analytic_pair_from_name(o.pair);
    const AnalyticPairParams prm{p, o.amplitude, o.amplitude_b};
    const int N = o.n;
    const double h = 1.0 / static_cast<double>(N);
    const std::size_t cells = static_cast<std::size_t>(N) * N * N;
    const real gbar1[3] = {0.0, 1.0, 0.0};
    const real gbar2[3] = {0.0, 0.0, 1.0};

    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> u1(cells), u2(cells);
    double min_c_exact = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * h;
        for (int j = 0; j < N; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * h;
            for (int i = 0; i < N; ++i) {
                const double x = (static_cast<double>(i) + 0.5) * h;
                double s1, s2, g1[3], g2[3];
                analytic_pair_fluct(prm, x, y, z, s1, g1, s2, g2);
                const std::size_t c =
                    static_cast<std::size_t>(i) +
                    static_cast<std::size_t>(N) *
                        (static_cast<std::size_t>(j) + static_cast<std::size_t>(N) * k);
                u1[c] = s1;
                u2[c] = s2;
                min_c_exact = std::min(min_c_exact, analytic_pair_abs_c(prm, x, y, z));
            }
        }
    }
    SplineLabels sl;
    sl.build(ctx, N, u1, u2, gbar1, gbar2);
    const double min_c_grid = sl.min_abs_c_grid(ctx);
    const double wall = seconds_since(t0);

    std::printf("==== spurious_spreading analytic-labels (SF-32 N2a) ====\n");
    std::printf("pair = %s  amplitude = %.17g%s  N = %d  h = %.17g\n", o.pair.c_str(), o.amplitude,
                p == AnalyticPair::B
                    ? (std::string("  amplitude_b = ") + std::to_string(o.amplitude_b)).c_str()
                    : "",
                N, h);
    std::printf("  min_abs_c_grid (spline, cell centres) = %.17g\n", min_c_grid);
    std::printf("  min_abs_c_exact (closed form, cell centres) = %.17g\n", min_c_exact);
    if (!(min_c_grid > 0.5))
        std::printf("  WARNING: min_abs_c_grid <= 0.5: return-map will refuse this pair\n");

    JVal meta;
    meta["tool"] = "spurious_spreading analytic-labels";
    meta["increment"] = "SF-32 N2a";
    meta["route"] = "analytic";
    meta["pair"] = o.pair;
    meta["formula"] = pair_formula(p);
    meta["amplitude"] = o.amplitude;
    meta["amplitude_b"] = (p == AnalyticPair::B) ? JVal(o.amplitude_b) : JVal(nullptr);
    meta["n"] = N;
    meta["h"] = h;
    meta["vbar"] = 1.0;
    meta["gbar1"] = gbar_json(gbar1);
    meta["gbar2"] = gbar_json(gbar2);
    meta["exit_reason"] = JVal(nullptr);
    meta["iterations"] = JVal(nullptr);
    meta["r_F"] = JVal(nullptr);
    meta["e_v"] = JVal(nullptr);
    meta["e_psi1"] = JVal(nullptr);
    meta["e_psi2"] = JVal(nullptr);
    meta["min_abs_c_grid"] = min_c_grid;
    meta["min_abs_c_grid_source"] =
        "SF-28 spline pair, |grad psi1 x grad psi2| at the n^3 cell centres";
    meta["min_abs_c_exact_cells"] = min_c_exact;
    meta["wall_seconds"] = o.no_timing ? 0.0 : wall;
    meta["files"] = label_files_json(o.out);
    meta["config"] =
        jobj({{"pair", o.pair},
              {"amplitude", o.amplitude},
              {"amplitude_b", (p == AnalyticPair::B) ? JVal(o.amplitude_b) : JVal(nullptr)},
              {"n", o.n},
              {"no_timing", o.no_timing},
              {"out", o.out}});
    write_labels(o.out, N, u1, u2, meta);
    std::printf("labels written: %s{.json,_u1.bin,_u2.bin}\n", o.out.c_str());
    return 0;
}

} // namespace spurious_spreading
