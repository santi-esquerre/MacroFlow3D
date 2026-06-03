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
#include "src/physics/particles/pspta/invariants/PsptaInvariantField.cuh"
#include "src/physics/particles/pspta/invariants/TransportOperator3D.cuh"
#include "src/physics/stochastic/stochastic.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/PetscSlepcInit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;
namespace blas = macroflow3d::blas;

namespace {

struct CaseSpec {
    std::string name;
    Grid3D grid;
    VelocityField velocity;
    bool expected_yz_subspace = false;

    explicit CaseSpec(const std::string& n, const Grid3D& g) : name(n), grid(g), velocity(g) {}
};

struct HostVelocity {
    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
};

struct RawFieldData {
    std::vector<double> psi1;
    std::vector<double> psi2;
    std::vector<double> Apsi1;
    std::vector<double> Apsi2;
    std::vector<double> g1x;
    std::vector<double> g1y;
    std::vector<double> g1z;
    std::vector<double> g2x;
    std::vector<double> g2y;
    std::vector<double> g2z;
};

struct QualityMetrics {
    double rms_vdotgrad1 = 0.0;
    double max_vdotgrad1 = 0.0;
    double rms_vdotgrad2 = 0.0;
    double max_vdotgrad2 = 0.0;

    double rms_ri1 = 0.0;
    double max_ri1 = 0.0;
    double rms_ri2 = 0.0;
    double max_ri2 = 0.0;

    double rms_mismatch = 0.0;
    double max_mismatch = 0.0;
    double rel_rms_mismatch = 0.0;

    double mean_abs_cos = 0.0;
    double max_abs_cos = 0.0;
    double degeneracy_fraction = 0.0;

    double masked_fraction = 0.0;
    double mean_speed = 0.0;
    double max_speed = 0.0;
    double low_vel_threshold = 0.0;
    double low_vel_fraction = 0.0;

    double rms_vdotgrad1_low_vel = 0.0;
    double rms_vdotgrad1_high_vel = 0.0;
    double rms_vdotgrad2_low_vel = 0.0;
    double rms_vdotgrad2_high_vel = 0.0;
    double rel_rms_mismatch_low_vel = 0.0;
    double rel_rms_mismatch_high_vel = 0.0;
    double mean_abs_cos_low_vel = 0.0;
    double mean_abs_cos_high_vel = 0.0;

    double mean_abs_cos_degenerate = 0.0;
    double mean_abs_cos_nondegenerate = 0.0;
    double rel_rms_mismatch_degenerate = 0.0;
    double rel_rms_mismatch_nondegenerate = 0.0;
};

struct RotatedBasisMetrics {
    double angle_deg = 0.0;
    double lambda1 = 0.0;
    double lambda2 = 0.0;
    double residual1 = 0.0;
    double residual2 = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    double orthogonality = 0.0;
    bool gauge_ready = false;
    QualityMetrics quality;
    double combined_score = 0.0;
};

struct SolveSummary {
    std::string case_name;
    double mu = 0.0;
    EigensolverResult result;
    std::vector<DeviceBuffer<real>> eigenvectors;
    ModalQualityReport modal_quality;
    InvariantConstructionInfo construction_info;
    std::vector<double> expected_captures;
};

// Gauge readiness: after in-subspace rotation (det=1 preserves the cross product),
// fit a single scalar alpha such that alpha * (grad psi1 x grad psi2) best-matches v
// in the L2 sense. The residual floor sqrt(1 - cos^2(v, c)) is a geometric lower bound
// that depends only on the subspace, not on intra-subspace rotation or scale.
struct GaugeReadyMetrics {
    double angle_deg = 0.0;
    double mean_psi1 = 0.0;
    double mean_psi2 = 0.0;
    double orientation_sign = 1.0;
    double symmetric_scale = 1.0;
    double v_norm = 0.0;
    double cross_norm = 0.0;
    double cross_norm_after_gauge = 0.0;
    double v_dot_cross = 0.0;
    double cos_v_cross = 0.0;
    double alpha_opt = 0.0;
    double rel_residual_before_gauge = 0.0;
    double rel_residual_after_gauge = 0.0;
    double residual_floor_rel = 0.0;
    double rms_vdotgrad1 = 0.0;
    double rms_vdotgrad2 = 0.0;
    double rms_ri1 = 0.0;
    double rms_ri2 = 0.0;
    double mean_abs_cos = 0.0;
    double degeneracy_fraction = 0.0;
    double post_gauge_grad1_norm = 0.0;
    double post_gauge_grad2_norm = 0.0;
};

// Per-mode decomposition of the Rayleigh quotient of A = D^T D + mu L into the
// transport (D^T D) and regularization (mu L) contributions.
struct ModalEnergyRow {
    int mode_index = 0;
    double eigenvalue_solver = 0.0;
    double psi_norm_sq = 0.0;
    double e_transport = 0.0;
    double e_regularization = 0.0;
    double e_total = 0.0;
    double rayleigh_recomputed = 0.0;
    double f_transport = 0.0;
    double f_regularization = 0.0;
    double residual_Ax_lambda_x_rel = 0.0;
};

// Localization of reconstruction residual by spatial/geometric region.
struct LocalizationStats {
    std::string region;
    double fraction = 0.0;
    double rms_vdotgrad1 = 0.0;
    double rms_vdotgrad2 = 0.0;
    double rel_residual_after_gauge = 0.0;
    double mean_abs_cos = 0.0;
    long long cell_count = 0;
};

struct GaugeReadyEvaluation {
    GaugeReadyMetrics metrics;
    std::vector<double> psi1;
    std::vector<double> psi2;
    std::vector<double> g1x;
    std::vector<double> g1y;
    std::vector<double> g1z;
    std::vector<double> g2x;
    std::vector<double> g2y;
    std::vector<double> g2z;
};

static void fill_uniform_velocity(VelocityField& vel, real vx, real vy, real vz) {
    std::vector<real> hU(vel.size_U(), vx);
    std::vector<real> hV(vel.size_V(), vy);
    std::vector<real> hW(vel.size_W(), vz);
    cudaMemcpy(vel.U.data(), hU.data(), hU.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.V.data(), hV.data(), hV.size() * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(vel.W.data(), hW.data(), hW.size() * sizeof(real), cudaMemcpyHostToDevice);
}

static void fill_layered_x_velocity(VelocityField& vel) {
    std::vector<real> u(vel.size_U(), 0.0f);
    std::vector<real> v(vel.size_V(), 0.0f);
    std::vector<real> w(vel.size_W(), 0.0f);

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
                }
            }
        }
    }
    return h;
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

static double subspace_capture(CudaContext& ctx, const DeviceBuffer<real>& psi_buf,
                               const std::vector<DeviceBuffer<real>>& basis) {
    if (basis.empty())
        return -1.0;

    blas::ReductionWorkspace ws;
    DeviceSpan<const real> psi(psi_buf.data(), psi_buf.size());
    const double norm_psi = blas::nrm2_host(ctx, psi, ws);
    if (norm_psi <= 1e-30)
        return 0.0;

    double capture = 0.0;
    for (const auto& basis_vec : basis) {
        DeviceSpan<const real> bj(basis_vec.data(), basis_vec.size());
        const double norm_b = blas::nrm2_host(ctx, bj, ws);
        if (norm_b <= 1e-30)
            continue;
        const double overlap = blas::dot_host(ctx, psi, bj, ws) / (norm_psi * norm_b);
        capture += overlap * overlap;
    }
    return capture;
}

static double eigenspace_similarity(CudaContext& ctx, const std::vector<DeviceBuffer<real>>& a,
                                    const std::vector<DeviceBuffer<real>>& b) {
    if (a.size() < 2 || b.size() < 2)
        return -1.0;

    blas::ReductionWorkspace ws;
    double sum_sq = 0.0;
    for (int i = 0; i < 2; ++i) {
        const DeviceSpan<const real> ai(a[i].data(), a[i].size());
        const double ni = blas::nrm2_host(ctx, ai, ws);
        for (int j = 0; j < 2; ++j) {
            const DeviceSpan<const real> bj(b[j].data(), b[j].size());
            const double nj = blas::nrm2_host(ctx, bj, ws);
            if (ni <= 1e-30 || nj <= 1e-30)
                continue;
            const double cij = blas::dot_host(ctx, ai, bj, ws) / (ni * nj);
            sum_sq += cij * cij;
        }
    }
    return 0.5 * sum_sq;
}

static std::vector<double> copy_device_to_host(const DeviceBuffer<real>& buf) {
    std::vector<double> out(buf.size());
    std::vector<real> tmp(buf.size());
    cudaMemcpy(tmp.data(), buf.data(), tmp.size() * sizeof(real), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < tmp.size(); ++i)
        out[i] = static_cast<double>(tmp[i]);
    return out;
}

static HostVelocity copy_cell_center_velocity(const VelocityField& vel) {
    HostVelocity hv;
    hv.vx.resize(static_cast<size_t>(vel.nx) * vel.ny * vel.nz);
    hv.vy.resize(hv.vx.size());
    hv.vz.resize(hv.vx.size());

    std::vector<real> U(vel.size_U()), V(vel.size_V()), W(vel.size_W());
    cudaMemcpy(U.data(), vel.U.data(), U.size() * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(V.data(), vel.V.data(), V.size() * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(W.data(), vel.W.data(), W.size() * sizeof(real), cudaMemcpyDeviceToHost);

    const auto idxU = [&vel](int i, int j, int k) {
        return static_cast<size_t>(i) + static_cast<size_t>(vel.nx + 1) * j +
               static_cast<size_t>(vel.nx + 1) * vel.ny * k;
    };
    const auto idxV = [&vel](int i, int j, int k) {
        return static_cast<size_t>(i) + static_cast<size_t>(vel.nx) * j +
               static_cast<size_t>(vel.nx) * (vel.ny + 1) * k;
    };
    const auto idxW = [&vel](int i, int j, int k) {
        return static_cast<size_t>(i) + static_cast<size_t>(vel.nx) * j +
               static_cast<size_t>(vel.nx) * vel.ny * k;
    };

    for (int k = 0; k < vel.nz; ++k) {
        for (int j = 0; j < vel.ny; ++j) {
            for (int i = 0; i < vel.nx; ++i) {
                const size_t c = static_cast<size_t>(i + vel.nx * (j + vel.ny * k));
                hv.vx[c] = 0.5 * (static_cast<double>(U[idxU(i, j, k)]) +
                                  static_cast<double>(U[idxU(i + 1, j, k)]));
                hv.vy[c] = 0.5 * (static_cast<double>(V[idxV(i, j, k)]) +
                                  static_cast<double>(V[idxV(i, j + 1, k)]));
                hv.vz[c] = 0.5 * (static_cast<double>(W[idxW(i, j, k)]) +
                                  static_cast<double>(W[idxW(i, j, k + 1)]));
            }
        }
    }

    return hv;
}

static void compute_gradients_periodic_yz(const Grid3D& grid, const std::vector<double>& psi,
                                          std::vector<double>& gx, std::vector<double>& gy,
                                          std::vector<double>& gz) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    gx.resize(psi.size());
    gy.resize(psi.size());
    gz.resize(psi.size());

    auto idx = [nx, ny](int i, int j, int k) { return static_cast<size_t>(i + nx * (j + ny * k)); };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t c = idx(i, j, k);
                if (i == 0) {
                    gx[c] = (psi[idx(i + 1, j, k)] - psi[c]) / grid.dx;
                } else if (i == nx - 1) {
                    gx[c] = (psi[c] - psi[idx(i - 1, j, k)]) / grid.dx;
                } else {
                    gx[c] = (psi[idx(i + 1, j, k)] - psi[idx(i - 1, j, k)]) / (2.0 * grid.dx);
                }

                const int jm = (j - 1 + ny) % ny;
                const int jp = (j + 1) % ny;
                const int km = (k - 1 + nz) % nz;
                const int kp = (k + 1) % nz;
                gy[c] = (psi[idx(i, jp, k)] - psi[idx(i, jm, k)]) / (2.0 * grid.dy);
                gz[c] = (psi[idx(i, j, kp)] - psi[idx(i, j, km)]) / (2.0 * grid.dz);
            }
        }
    }
}

static void apply_operator_host(CombinedOperatorA& A, const DeviceBuffer<real>& in,
                                std::vector<double>& out, CudaContext& ctx) {
    DeviceBuffer<real> d_out(in.size());
    A.apply_A(DeviceSpan<const real>(in.data(), in.size()), d_out.span(), ctx.cuda_stream());
    cudaStreamSynchronize(ctx.cuda_stream());
    out = copy_device_to_host(d_out);
}

static DeviceBuffer<real> copy_host_to_device(const std::vector<double>& host) {
    DeviceBuffer<real> out(host.size());
    std::vector<real> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i)
        tmp[i] = static_cast<real>(host[i]);
    cudaMemcpy(out.data(), tmp.data(), tmp.size() * sizeof(real), cudaMemcpyHostToDevice);
    return out;
}

static CaseSpec make_uniform_case() {
    Grid3D grid(16, 16, 16, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0);
    CaseSpec spec("uniform_x", grid);
    fill_uniform_velocity(spec.velocity, 1.0, 0.0, 0.0);
    spec.expected_yz_subspace = true;
    return spec;
}

static CaseSpec make_layered_case() {
    Grid3D grid(16, 16, 16, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0);
    CaseSpec spec("layered_x", grid);
    fill_layered_x_velocity(spec.velocity);
    spec.expected_yz_subspace = true;
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
    if (!solve_result.converged)
        throw std::runtime_error("small Darcy control case did not converge");

    compute_velocity_from_head(spec.velocity, head, K, grid, bc, ctx);
    spec.expected_yz_subspace = false;
    return spec;
}

static SolveSummary solve_case_with_mu(const CaseSpec& case_spec, double mu, CudaContext& ctx) {
    SolveSummary out;
    out.case_name = case_spec.name;
    out.mu = mu;

    TransportOperatorConfig D_cfg;
    D_cfg.x_bc = TransportXBoundary::OneSided;
    TransportOperator3D D(&case_spec.velocity, case_spec.grid, D_cfg);
    LaplacianOperator3D L(case_spec.grid, LaplacianOperator3D::XBoundary::Neumann);
    CombinedOperatorA A(&D, &L, mu);

    auto backend = create_eigensolver_backend("slepc");
    if (!backend)
        throw std::runtime_error("slepc backend unavailable");

    EigensolverConfig cfg;
    cfg.n_eigenvectors = 3;
    cfg.tolerance = 1.0e-8;
    cfg.max_iterations = 500;
    cfg.verbose = false;

    out.result = backend->solve(A, cfg, ctx, out.eigenvectors);

    PsptaInvariantField inv;
    inv.resize(case_spec.grid);
    inv.ingest_eigenvectors(out.eigenvectors[0], out.eigenvectors[1], out.result, mu,
                            backend->name(), ctx, ctx.cuda_stream());
    out.modal_quality = inv.modal_quality();
    out.construction_info = inv.construction_info();

    if (case_spec.expected_yz_subspace) {
        auto basis = make_expected_yz_basis(case_spec.grid);
        for (int i = 0; i < 2; ++i)
            out.expected_captures.push_back(subspace_capture(ctx, out.eigenvectors[i], basis));
    }

    return out;
}

static RawFieldData prepare_raw_field_data(const CaseSpec& case_spec, CombinedOperatorA& A,
                                           const std::vector<DeviceBuffer<real>>& evs,
                                           CudaContext& ctx) {
    RawFieldData data;
    data.psi1 = copy_device_to_host(evs[0]);
    data.psi2 = copy_device_to_host(evs[1]);
    apply_operator_host(A, evs[0], data.Apsi1, ctx);
    apply_operator_host(A, evs[1], data.Apsi2, ctx);
    compute_gradients_periodic_yz(case_spec.grid, data.psi1, data.g1x, data.g1y, data.g1z);
    compute_gradients_periodic_yz(case_spec.grid, data.psi2, data.g2x, data.g2y, data.g2z);
    return data;
}

static double percentile(std::vector<double> values, double q) {
    if (values.empty())
        return 0.0;
    const size_t k = static_cast<size_t>(q * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + k, values.end());
    return values[k];
}

static RotatedBasisMetrics evaluate_rotation(const Grid3D& grid, const HostVelocity& vel,
                                             const RawFieldData& raw, double angle_deg) {
    RotatedBasisMetrics out;
    out.angle_deg = angle_deg;

    const double theta = angle_deg * M_PI / 180.0;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const size_t n = raw.psi1.size();

    std::vector<double> speeds;
    speeds.reserve(n);
    for (size_t idx = 0; idx < n; ++idx) {
        const double vmag = std::sqrt(vel.vx[idx] * vel.vx[idx] + vel.vy[idx] * vel.vy[idx] +
                                      vel.vz[idx] * vel.vz[idx]);
        speeds.push_back(vmag);
        out.quality.mean_speed += vmag;
        out.quality.max_speed = std::max(out.quality.max_speed, vmag);
    }
    out.quality.mean_speed /= std::max<double>(n, 1.0);
    out.quality.low_vel_threshold = percentile(speeds, 0.10);

    double ssq_r1 = 0.0, ssq_r2 = 0.0, ssq_ri1 = 0.0, ssq_ri2 = 0.0;
    double ssq_mismatch = 0.0, ssq_rel_mismatch = 0.0;
    double sum_cos = 0.0;
    long long masked = 0;
    long long low_vel_count = 0;
    long long deg_count = 0;

    double low_ssq_r1 = 0.0, high_ssq_r1 = 0.0;
    double low_ssq_r2 = 0.0, high_ssq_r2 = 0.0;
    double low_ssq_rel = 0.0, high_ssq_rel = 0.0;
    double low_sum_cos = 0.0, high_sum_cos = 0.0;
    long long high_vel_count = 0;
    double deg_sum_cos = 0.0, nondeg_sum_cos = 0.0;
    double deg_ssq_rel = 0.0, nondeg_ssq_rel = 0.0;
    long long nondeg_count = 0;

    double dot12 = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    double ssq_resid1 = 0.0;
    double ssq_resid2 = 0.0;
    double dot_phi1_Aphi1 = 0.0;
    double dot_phi2_Aphi2 = 0.0;

    for (size_t idx = 0; idx < n; ++idx) {
        const double psi1 = c * raw.psi1[idx] - s * raw.psi2[idx];
        const double psi2 = s * raw.psi1[idx] + c * raw.psi2[idx];
        const double Apsi1 = c * raw.Apsi1[idx] - s * raw.Apsi2[idx];
        const double Apsi2 = s * raw.Apsi1[idx] + c * raw.Apsi2[idx];

        const double g1x = c * raw.g1x[idx] - s * raw.g2x[idx];
        const double g1y = c * raw.g1y[idx] - s * raw.g2y[idx];
        const double g1z = c * raw.g1z[idx] - s * raw.g2z[idx];
        const double g2x = s * raw.g1x[idx] + c * raw.g2x[idx];
        const double g2y = s * raw.g1y[idx] + c * raw.g2y[idx];
        const double g2z = s * raw.g1z[idx] + c * raw.g2z[idx];

        const double vx = vel.vx[idx];
        const double vy = vel.vy[idx];
        const double vz = vel.vz[idx];
        const double vmag = speeds[idx];

        const double d1 = vx * g1x + vy * g1y + vz * g1z;
        const double d2 = vx * g2x + vy * g2y + vz * g2z;

        const double g1mag = std::sqrt(g1x * g1x + g1y * g1y + g1z * g1z);
        const double g2mag = std::sqrt(g2x * g2x + g2y * g2y + g2z * g2z);
        const double denom1 = vmag * g1mag;
        const double denom2 = vmag * g2mag;
        const double ri1 = std::fabs(d1) / (denom1 + 1e-12);
        const double ri2 = std::fabs(d2) / (denom2 + 1e-12);

        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        const double mismatch =
            std::sqrt((vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz));
        const double rel_mismatch = mismatch / (vmag + 1e-12);
        const double abs_cos =
            std::fabs(g1x * g2x + g1y * g2y + g1z * g2z) / (g1mag * g2mag + 1e-12);

        ssq_r1 += d1 * d1;
        ssq_r2 += d2 * d2;
        ssq_ri1 += ri1 * ri1;
        ssq_ri2 += ri2 * ri2;
        ssq_mismatch += mismatch * mismatch;
        ssq_rel_mismatch += rel_mismatch * rel_mismatch;
        sum_cos += abs_cos;

        out.quality.max_vdotgrad1 = std::max(out.quality.max_vdotgrad1, std::fabs(d1));
        out.quality.max_vdotgrad2 = std::max(out.quality.max_vdotgrad2, std::fabs(d2));
        out.quality.max_ri1 = std::max(out.quality.max_ri1, ri1);
        out.quality.max_ri2 = std::max(out.quality.max_ri2, ri2);
        out.quality.max_mismatch = std::max(out.quality.max_mismatch, mismatch);
        out.quality.max_abs_cos = std::max(out.quality.max_abs_cos, abs_cos);

        const bool is_masked = (vmag < 1e-12) || (g1mag < 1e-12) || (g2mag < 1e-12);
        if (is_masked)
            ++masked;

        const bool low_vel = (vmag <= out.quality.low_vel_threshold);
        if (low_vel) {
            ++low_vel_count;
            low_ssq_r1 += d1 * d1;
            low_ssq_r2 += d2 * d2;
            low_ssq_rel += rel_mismatch * rel_mismatch;
            low_sum_cos += abs_cos;
        } else {
            ++high_vel_count;
            high_ssq_r1 += d1 * d1;
            high_ssq_r2 += d2 * d2;
            high_ssq_rel += rel_mismatch * rel_mismatch;
            high_sum_cos += abs_cos;
        }

        const bool deg = abs_cos > 0.9;
        if (deg) {
            ++deg_count;
            deg_sum_cos += abs_cos;
            deg_ssq_rel += rel_mismatch * rel_mismatch;
        } else {
            ++nondeg_count;
            nondeg_sum_cos += abs_cos;
            nondeg_ssq_rel += rel_mismatch * rel_mismatch;
        }

        dot12 += psi1 * psi2;
        norm1_sq += psi1 * psi1;
        norm2_sq += psi2 * psi2;
        dot_phi1_Aphi1 += psi1 * Apsi1;
        dot_phi2_Aphi2 += psi2 * Apsi2;
    }

    out.quality.rms_vdotgrad1 = std::sqrt(ssq_r1 / std::max<double>(n, 1.0));
    out.quality.rms_vdotgrad2 = std::sqrt(ssq_r2 / std::max<double>(n, 1.0));
    out.quality.rms_ri1 = std::sqrt(ssq_ri1 / std::max<double>(n, 1.0));
    out.quality.rms_ri2 = std::sqrt(ssq_ri2 / std::max<double>(n, 1.0));
    out.quality.rms_mismatch = std::sqrt(ssq_mismatch / std::max<double>(n, 1.0));
    out.quality.rel_rms_mismatch = std::sqrt(ssq_rel_mismatch / std::max<double>(n, 1.0));
    out.quality.mean_abs_cos = sum_cos / std::max<double>(n, 1.0);
    out.quality.degeneracy_fraction = static_cast<double>(deg_count) / std::max<double>(n, 1.0);
    out.quality.masked_fraction = static_cast<double>(masked) / std::max<double>(n, 1.0);
    out.quality.low_vel_fraction = static_cast<double>(low_vel_count) / std::max<double>(n, 1.0);

    out.quality.rms_vdotgrad1_low_vel =
        std::sqrt(low_ssq_r1 / std::max<double>(low_vel_count, 1.0));
    out.quality.rms_vdotgrad1_high_vel =
        std::sqrt(high_ssq_r1 / std::max<double>(high_vel_count, 1.0));
    out.quality.rms_vdotgrad2_low_vel =
        std::sqrt(low_ssq_r2 / std::max<double>(low_vel_count, 1.0));
    out.quality.rms_vdotgrad2_high_vel =
        std::sqrt(high_ssq_r2 / std::max<double>(high_vel_count, 1.0));
    out.quality.rel_rms_mismatch_low_vel =
        std::sqrt(low_ssq_rel / std::max<double>(low_vel_count, 1.0));
    out.quality.rel_rms_mismatch_high_vel =
        std::sqrt(high_ssq_rel / std::max<double>(high_vel_count, 1.0));
    out.quality.mean_abs_cos_low_vel = low_sum_cos / std::max<double>(low_vel_count, 1.0);
    out.quality.mean_abs_cos_high_vel = high_sum_cos / std::max<double>(high_vel_count, 1.0);
    out.quality.mean_abs_cos_degenerate = deg_sum_cos / std::max<double>(deg_count, 1.0);
    out.quality.mean_abs_cos_nondegenerate = nondeg_sum_cos / std::max<double>(nondeg_count, 1.0);
    out.quality.rel_rms_mismatch_degenerate =
        std::sqrt(deg_ssq_rel / std::max<double>(deg_count, 1.0));
    out.quality.rel_rms_mismatch_nondegenerate =
        std::sqrt(nondeg_ssq_rel / std::max<double>(nondeg_count, 1.0));

    out.norm1 = std::sqrt(norm1_sq);
    out.norm2 = std::sqrt(norm2_sq);
    out.orthogonality = std::fabs(dot12) / std::max(out.norm1 * out.norm2, 1e-30);
    out.lambda1 = dot_phi1_Aphi1 / std::max(norm1_sq, 1e-30);
    out.lambda2 = dot_phi2_Aphi2 / std::max(norm2_sq, 1e-30);

    for (size_t idx = 0; idx < n; ++idx) {
        const double psi1 = c * raw.psi1[idx] - s * raw.psi2[idx];
        const double psi2 = s * raw.psi1[idx] + c * raw.psi2[idx];
        const double Apsi1 = c * raw.Apsi1[idx] - s * raw.Apsi2[idx];
        const double Apsi2 = s * raw.Apsi1[idx] + c * raw.Apsi2[idx];
        const double r1 = Apsi1 - out.lambda1 * psi1;
        const double r2 = Apsi2 - out.lambda2 * psi2;
        ssq_resid1 += r1 * r1;
        ssq_resid2 += r2 * r2;
    }

    out.residual1 = std::sqrt(ssq_resid1) / std::max(out.norm1, 1e-30);
    out.residual2 = std::sqrt(ssq_resid2) / std::max(out.norm2, 1e-30);
    out.gauge_ready = (out.norm1 > 0.1 && out.norm2 > 0.1 && out.orthogonality < 0.1 &&
                       out.residual1 < 1e-3 && out.residual2 < 1e-3);
    out.combined_score = out.quality.rms_ri1 + out.quality.rms_ri2 + out.quality.rel_rms_mismatch +
                         0.1 * out.quality.mean_abs_cos + 0.1 * out.quality.degeneracy_fraction;
    return out;
}

static GaugeReadyEvaluation evaluate_gauge_ready(const Grid3D& grid, const HostVelocity& vel,
                                                 const RawFieldData& raw, double angle_deg) {
    GaugeReadyEvaluation eval;
    GaugeReadyMetrics& out = eval.metrics;
    out.angle_deg = angle_deg;

    const double theta = angle_deg * M_PI / 180.0;
    const double c_th = std::cos(theta);
    const double s_th = std::sin(theta);
    const size_t n = raw.psi1.size();

    eval.psi1.resize(n);
    eval.psi2.resize(n);

    double mean1 = 0.0;
    double mean2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        eval.psi1[i] = c_th * raw.psi1[i] - s_th * raw.psi2[i];
        eval.psi2[i] = s_th * raw.psi1[i] + c_th * raw.psi2[i];
        mean1 += eval.psi1[i];
        mean2 += eval.psi2[i];
    }
    mean1 /= std::max<double>(n, 1.0);
    mean2 /= std::max<double>(n, 1.0);
    out.mean_psi1 = mean1;
    out.mean_psi2 = mean2;
    for (size_t i = 0; i < n; ++i) {
        eval.psi1[i] -= mean1;
        eval.psi2[i] -= mean2;
    }

    compute_gradients_periodic_yz(grid, eval.psi1, eval.g1x, eval.g1y, eval.g1z);
    compute_gradients_periodic_yz(grid, eval.psi2, eval.g2x, eval.g2y, eval.g2z);

    double vnorm_sq = 0.0;
    double cross_norm_sq = 0.0;
    double vdotc = 0.0;
    double ssq_res_before = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double vx = vel.vx[i];
        const double vy = vel.vy[i];
        const double vz = vel.vz[i];
        const double cx = eval.g1y[i] * eval.g2z[i] - eval.g1z[i] * eval.g2y[i];
        const double cy = eval.g1z[i] * eval.g2x[i] - eval.g1x[i] * eval.g2z[i];
        const double cz = eval.g1x[i] * eval.g2y[i] - eval.g1y[i] * eval.g2x[i];
        vnorm_sq += vx * vx + vy * vy + vz * vz;
        cross_norm_sq += cx * cx + cy * cy + cz * cz;
        vdotc += vx * cx + vy * cy + vz * cz;
        ssq_res_before += (vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz);
    }

    out.v_norm = std::sqrt(vnorm_sq);
    out.cross_norm = std::sqrt(cross_norm_sq);
    out.v_dot_cross = vdotc;
    out.cos_v_cross = vdotc / std::max(out.v_norm * out.cross_norm, 1e-30);
    out.alpha_opt = (cross_norm_sq > 1e-30) ? vdotc / cross_norm_sq : 0.0;
    out.orientation_sign = (out.alpha_opt >= 0.0) ? 1.0 : -1.0;
    out.symmetric_scale = std::sqrt(std::fabs(out.alpha_opt));
    out.rel_residual_before_gauge = std::sqrt(ssq_res_before) / std::max(out.v_norm, 1e-30);

    for (size_t i = 0; i < n; ++i) {
        eval.psi1[i] *= out.symmetric_scale;
        eval.psi2[i] *= out.orientation_sign * out.symmetric_scale;
        eval.g1x[i] *= out.symmetric_scale;
        eval.g1y[i] *= out.symmetric_scale;
        eval.g1z[i] *= out.symmetric_scale;
        eval.g2x[i] *= out.orientation_sign * out.symmetric_scale;
        eval.g2y[i] *= out.orientation_sign * out.symmetric_scale;
        eval.g2z[i] *= out.orientation_sign * out.symmetric_scale;
    }

    double ssq_res_after = 0.0;
    double ssq_r1 = 0.0;
    double ssq_r2 = 0.0;
    double ssq_ri1 = 0.0;
    double ssq_ri2 = 0.0;
    double sum_abs_cos = 0.0;
    double g1_norm_sq = 0.0;
    double g2_norm_sq = 0.0;
    double cross_norm_after_sq = 0.0;
    long long deg_count = 0;

    for (size_t i = 0; i < n; ++i) {
        const double vx = vel.vx[i];
        const double vy = vel.vy[i];
        const double vz = vel.vz[i];
        const double vmag = std::sqrt(vx * vx + vy * vy + vz * vz);

        const double g1x = eval.g1x[i];
        const double g1y = eval.g1y[i];
        const double g1z = eval.g1z[i];
        const double g2x = eval.g2x[i];
        const double g2y = eval.g2y[i];
        const double g2z = eval.g2z[i];

        const double d1 = vx * g1x + vy * g1y + vz * g1z;
        const double d2 = vx * g2x + vy * g2y + vz * g2z;
        const double g1sq = g1x * g1x + g1y * g1y + g1z * g1z;
        const double g2sq = g2x * g2x + g2y * g2y + g2z * g2z;
        const double g1mag = std::sqrt(g1sq);
        const double g2mag = std::sqrt(g2sq);
        const double abs_cos =
            std::fabs(g1x * g2x + g1y * g2y + g1z * g2z) / (g1mag * g2mag + 1e-12);

        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        const double rx = vx - cx;
        const double ry = vy - cy;
        const double rz = vz - cz;

        ssq_res_after += rx * rx + ry * ry + rz * rz;
        ssq_r1 += d1 * d1;
        ssq_r2 += d2 * d2;
        ssq_ri1 += std::pow(std::fabs(d1) / (vmag * g1mag + 1e-12), 2);
        ssq_ri2 += std::pow(std::fabs(d2) / (vmag * g2mag + 1e-12), 2);
        sum_abs_cos += abs_cos;
        g1_norm_sq += g1sq;
        g2_norm_sq += g2sq;
        cross_norm_after_sq += cx * cx + cy * cy + cz * cz;
        if (abs_cos > 0.9)
            ++deg_count;
    }

    out.cross_norm_after_gauge = std::sqrt(cross_norm_after_sq);
    out.rel_residual_after_gauge = std::sqrt(ssq_res_after) / std::max(out.v_norm, 1e-30);
    out.residual_floor_rel = std::sqrt(std::max(1.0 - out.cos_v_cross * out.cos_v_cross, 0.0));
    out.rms_vdotgrad1 = std::sqrt(ssq_r1 / std::max<double>(n, 1.0));
    out.rms_vdotgrad2 = std::sqrt(ssq_r2 / std::max<double>(n, 1.0));
    out.rms_ri1 = std::sqrt(ssq_ri1 / std::max<double>(n, 1.0));
    out.rms_ri2 = std::sqrt(ssq_ri2 / std::max<double>(n, 1.0));
    out.mean_abs_cos = sum_abs_cos / std::max<double>(n, 1.0);
    out.degeneracy_fraction = static_cast<double>(deg_count) / std::max<double>(n, 1.0);
    out.post_gauge_grad1_norm = std::sqrt(g1_norm_sq);
    out.post_gauge_grad2_norm = std::sqrt(g2_norm_sq);
    return eval;
}

static double gauge_ready_score(const GaugeReadyMetrics& m) {
    return m.rms_ri1 + m.rms_ri2 + m.rel_residual_after_gauge + 0.1 * m.mean_abs_cos +
           0.1 * m.degeneracy_fraction;
}

static std::vector<ModalEnergyRow>
compute_modal_energy(const TransportOperator3D& D, const LaplacianOperator3D& L, double mu,
                     const std::vector<DeviceBuffer<real>>& vectors,
                     const std::vector<double>& eigenvalues, CudaContext& ctx) {
    std::vector<ModalEnergyRow> rows;
    if (vectors.empty())
        return rows;

    blas::ReductionWorkspace red;
    DeviceBuffer<real> d_dtd(vectors.front().size());
    DeviceBuffer<real> d_l(vectors.front().size());
    DeviceBuffer<real> d_work(vectors.front().size());

    rows.reserve(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
        const DeviceSpan<const real> psi(vectors[i].data(), vectors[i].size());
        D.apply_DTD(psi, d_dtd.span(), d_work.span(), ctx.cuda_stream());
        L.apply_L(psi, d_l.span(), ctx.cuda_stream());
        cudaStreamSynchronize(ctx.cuda_stream());

        ModalEnergyRow row;
        row.mode_index = static_cast<int>(i);
        row.eigenvalue_solver =
            (i < eigenvalues.size()) ? eigenvalues[i] : std::numeric_limits<double>::quiet_NaN();
        row.psi_norm_sq = blas::dot_host(ctx, psi, psi, red);
        row.e_transport =
            blas::dot_host(ctx, psi, DeviceSpan<const real>(d_dtd.data(), d_dtd.size()), red);
        row.e_regularization =
            mu * blas::dot_host(ctx, psi, DeviceSpan<const real>(d_l.data(), d_l.size()), red);
        row.e_total = row.e_transport + row.e_regularization;
        row.rayleigh_recomputed = row.e_total / std::max(row.psi_norm_sq, 1e-30);
        row.f_transport = row.e_transport / std::max(std::fabs(row.e_total), 1e-30);
        row.f_regularization = row.e_regularization / std::max(std::fabs(row.e_total), 1e-30);

        std::vector<double> h_psi = copy_device_to_host(vectors[i]);
        std::vector<double> h_dtd = copy_device_to_host(d_dtd);
        std::vector<double> h_l = copy_device_to_host(d_l);
        double ssq_resid = 0.0;
        double ssq_psi = 0.0;
        for (size_t c = 0; c < h_psi.size(); ++c) {
            const double ap = h_dtd[c] + mu * h_l[c];
            const double r = ap - row.rayleigh_recomputed * h_psi[c];
            ssq_resid += r * r;
            ssq_psi += h_psi[c] * h_psi[c];
        }
        row.residual_Ax_lambda_x_rel = std::sqrt(ssq_resid) / std::max(std::sqrt(ssq_psi), 1e-30);
        rows.push_back(row);
    }
    return rows;
}

struct LocalAcc {
    long long cell_count = 0;
    double ssq_r1 = 0.0;
    double ssq_r2 = 0.0;
    double ssq_residual = 0.0;
    double ssq_v = 0.0;
    double sum_abs_cos = 0.0;
};

static LocalizationStats finalize_localization(const LocalAcc& acc, long long total_cells,
                                               const std::string& region) {
    LocalizationStats out;
    out.region = region;
    out.cell_count = acc.cell_count;
    out.fraction = static_cast<double>(acc.cell_count) / std::max<double>(total_cells, 1.0);
    const double denom = std::max<double>(acc.cell_count, 1.0);
    out.rms_vdotgrad1 = std::sqrt(acc.ssq_r1 / denom);
    out.rms_vdotgrad2 = std::sqrt(acc.ssq_r2 / denom);
    out.rel_residual_after_gauge = std::sqrt(acc.ssq_residual / std::max(acc.ssq_v, 1e-30));
    out.mean_abs_cos = acc.sum_abs_cos / denom;
    return out;
}

static std::vector<LocalizationStats> compute_localization_v2(const Grid3D& grid,
                                                              const HostVelocity& vel,
                                                              const GaugeReadyEvaluation& gauge) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const size_t n = gauge.psi1.size();
    auto idx = [nx, ny](int i, int j, int k) { return static_cast<size_t>(i + nx * (j + ny * k)); };

    std::vector<double> cross_mag(n);
    std::vector<double> abs_cos(n);
    for (size_t c = 0; c < n; ++c) {
        const double g1x = gauge.g1x[c];
        const double g1y = gauge.g1y[c];
        const double g1z = gauge.g1z[c];
        const double g2x = gauge.g2x[c];
        const double g2y = gauge.g2y[c];
        const double g2z = gauge.g2z[c];
        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        cross_mag[c] = std::sqrt(cx * cx + cy * cy + cz * cz);
        const double g1mag = std::sqrt(g1x * g1x + g1y * g1y + g1z * g1z);
        const double g2mag = std::sqrt(g2x * g2x + g2y * g2y + g2z * g2z);
        abs_cos[c] = std::fabs(g1x * g2x + g1y * g2y + g1z * g2z) / (g1mag * g2mag + 1e-12);
    }

    const double cross_q20 = percentile(cross_mag, 0.20);
    const double cross_q80 = percentile(cross_mag, 0.80);

    LocalAcc x_boundary_halo;
    LocalAcc x_interior;
    LocalAcc low_cross_q20;
    LocalAcc high_cross_q80;
    LocalAcc degenerate;
    LocalAcc nondegenerate;
    std::vector<LocalAcc> slice_acc(static_cast<size_t>(nx));

    auto accumulate = [&](LocalAcc& acc, size_t c, double d1, double d2, double residual_sq,
                          double v_sq, double abs_cos_val) {
        ++acc.cell_count;
        acc.ssq_r1 += d1 * d1;
        acc.ssq_r2 += d2 * d2;
        acc.ssq_residual += residual_sq;
        acc.ssq_v += v_sq;
        acc.sum_abs_cos += abs_cos_val;
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t c = idx(i, j, k);
                const double vx = vel.vx[c];
                const double vy = vel.vy[c];
                const double vz = vel.vz[c];
                const double g1x = gauge.g1x[c];
                const double g1y = gauge.g1y[c];
                const double g1z = gauge.g1z[c];
                const double g2x = gauge.g2x[c];
                const double g2y = gauge.g2y[c];
                const double g2z = gauge.g2z[c];
                const double d1 = vx * g1x + vy * g1y + vz * g1z;
                const double d2 = vx * g2x + vy * g2y + vz * g2z;
                const double cx = g1y * g2z - g1z * g2y;
                const double cy = g1z * g2x - g1x * g2z;
                const double cz = g1x * g2y - g1y * g2x;
                const double residual_sq =
                    (vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz);
                const double v_sq = vx * vx + vy * vy + vz * vz;

                accumulate(slice_acc[static_cast<size_t>(i)], c, d1, d2, residual_sq, v_sq,
                           abs_cos[c]);

                const bool near_x_boundary = (i <= 1) || (i >= nx - 2);
                accumulate(near_x_boundary ? x_boundary_halo : x_interior, c, d1, d2, residual_sq,
                           v_sq, abs_cos[c]);

                if (cross_mag[c] <= cross_q20)
                    accumulate(low_cross_q20, c, d1, d2, residual_sq, v_sq, abs_cos[c]);
                if (cross_mag[c] >= cross_q80)
                    accumulate(high_cross_q80, c, d1, d2, residual_sq, v_sq, abs_cos[c]);

                if (abs_cos[c] > 0.9) {
                    accumulate(degenerate, c, d1, d2, residual_sq, v_sq, abs_cos[c]);
                } else {
                    accumulate(nondegenerate, c, d1, d2, residual_sq, v_sq, abs_cos[c]);
                }
            }
        }
    }

    const long long total_cells = static_cast<long long>(n);
    std::vector<LocalizationStats> rows;
    rows.push_back(finalize_localization(x_interior, total_cells, "x_interior"));
    rows.push_back(finalize_localization(x_boundary_halo, total_cells, "x_boundary_halo"));
    rows.push_back(finalize_localization(low_cross_q20, total_cells, "cross_mag_q20_low"));
    rows.push_back(finalize_localization(high_cross_q80, total_cells, "cross_mag_q80_high"));
    rows.push_back(finalize_localization(degenerate, total_cells, "degenerate_abs_cos_gt_0.9"));
    rows.push_back(
        finalize_localization(nondegenerate, total_cells, "nondegenerate_abs_cos_le_0.9"));
    for (int i = 0; i < nx; ++i) {
        rows.push_back(finalize_localization(slice_acc[static_cast<size_t>(i)], total_cells,
                                             "x_slice_" + std::to_string(i)));
    }
    return rows;
}

static void write_summary_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,eig0,eig1,eig2,gap01,gap12,subspace_similarity,"
          "modal_ortho,residual1,residual2,gauge_ready,expected_capture_0,expected_capture_1,"
          "rms_vdotgrad1,max_vdotgrad1,rms_vdotgrad2,max_vdotgrad2,"
          "rms_ri1,max_ri1,rms_ri2,max_ri2,"
          "rms_mismatch,max_mismatch,rel_rms_mismatch,"
          "mean_abs_cos,max_abs_cos,degeneracy_fraction,masked_fraction,mean_speed,max_speed,"
          "low_vel_fraction,low_vel_threshold,combined_score\n";
}

static void write_rotation_header(std::ofstream& os) {
    os << "case,mu,angle_deg,lambda1,lambda2,residual1,residual2,norm1,norm2,orthogonality,"
          "gauge_ready,rms_ri1,rms_ri2,rel_rms_mismatch,mean_abs_cos,degeneracy_fraction,"
          "combined_score,alpha_opt,rel_residual_before_gauge,rel_residual_after_gauge,"
          "residual_floor_rel\n";
}

static void write_local_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,region,fraction,rms_vdotgrad1,rms_vdotgrad2,"
          "rel_rms_mismatch,mean_abs_cos\n";
}

static void write_gauge_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,mean_psi1,mean_psi2,orientation_sign,symmetric_scale,"
          "alpha_opt,v_norm,cross_norm,cross_norm_after_gauge,v_dot_cross,cos_v_cross,"
          "rel_residual_before_gauge,rel_residual_after_gauge,residual_floor_rel,"
          "rms_vdotgrad1,rms_vdotgrad2,rms_ri1,rms_ri2,mean_abs_cos,degeneracy_fraction\n";
}

static void write_energy_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,alpha_opt,mode_index,eigenvalue_solver,psi_norm_sq,"
          "e_transport,e_regularization,e_total,rayleigh_recomputed,f_transport,"
          "f_regularization,residual_Ax_lambda_x_rel\n";
}

static void write_local_v2_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,alpha_opt,region,fraction,cell_count,rms_vdotgrad1,"
          "rms_vdotgrad2,rel_residual_after_gauge,mean_abs_cos\n";
}

} // namespace

int main() {
    runtime::PetscSlepcInit::ensure();
    CudaContext ctx(0);

    std::filesystem::create_directories("artifacts/gate3");
    std::ofstream summary("artifacts/gate3/invariant_quality_summary.csv");
    std::ofstream rotation("artifacts/gate3/invariant_quality_rotation_scan.csv");
    std::ofstream local("artifacts/gate3/invariant_quality_localization.csv");
    std::ofstream gauge("artifacts/gate3/invariant_quality_gauge.csv");
    std::ofstream energy("artifacts/gate3/invariant_quality_energy.csv");
    std::ofstream local_v2("artifacts/gate3/invariant_quality_localization_v2.csv");
    write_summary_header(summary);
    write_rotation_header(rotation);
    write_local_header(local);
    write_gauge_header(gauge);
    write_energy_header(energy);
    write_local_v2_header(local_v2);

    std::vector<CaseSpec> cases;
    cases.push_back(make_uniform_case());
    cases.push_back(make_layered_case());
    cases.push_back(make_small_darcy_case(ctx));

    const std::vector<double> mu_values = {1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3};

    std::printf("=== Gate 3 Invariant Quality Analysis ===\n");
    std::printf("Artifacts:\n");
    std::printf("  artifacts/gate3/invariant_quality_summary.csv\n");
    std::printf("  artifacts/gate3/invariant_quality_rotation_scan.csv\n");
    std::printf("  artifacts/gate3/invariant_quality_localization.csv\n\n");
    std::printf("  artifacts/gate3/invariant_quality_gauge.csv\n");
    std::printf("  artifacts/gate3/invariant_quality_energy.csv\n");
    std::printf("  artifacts/gate3/invariant_quality_localization_v2.csv\n\n");

    for (const auto& case_spec : cases) {
        std::printf("=== Case: %s ===\n", case_spec.name.c_str());

        std::vector<DeviceBuffer<real>> baseline_subspace;
        bool baseline_set = false;

        for (double mu : mu_values) {
            std::printf("  mu = %.1e\n", mu);
            SolveSummary solve = solve_case_with_mu(case_spec, mu, ctx);

            TransportOperatorConfig D_cfg;
            D_cfg.x_bc = TransportXBoundary::OneSided;
            TransportOperator3D D(&case_spec.velocity, case_spec.grid, D_cfg);
            LaplacianOperator3D L(case_spec.grid, LaplacianOperator3D::XBoundary::Neumann);
            CombinedOperatorA A(&D, &L, mu);

            RawFieldData raw = prepare_raw_field_data(case_spec, A, solve.eigenvectors, ctx);
            const HostVelocity hv = copy_cell_center_velocity(case_spec.velocity);

            if (!baseline_set) {
                baseline_subspace.clear();
                baseline_subspace.emplace_back(solve.eigenvectors[0].size());
                baseline_subspace.emplace_back(solve.eigenvectors[1].size());
                cudaMemcpy(baseline_subspace[0].data(), solve.eigenvectors[0].data(),
                           solve.eigenvectors[0].size() * sizeof(real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(baseline_subspace[1].data(), solve.eigenvectors[1].data(),
                           solve.eigenvectors[1].size() * sizeof(real), cudaMemcpyDeviceToDevice);
                baseline_set = true;
            }

            const double subspace_similarity =
                baseline_set ? eigenspace_similarity(ctx, baseline_subspace, solve.eigenvectors)
                             : 1.0;

            RotatedBasisMetrics original = evaluate_rotation(case_spec.grid, hv, raw, 0.0);
            RotatedBasisMetrics best = original;
            GaugeReadyEvaluation original_gauge =
                evaluate_gauge_ready(case_spec.grid, hv, raw, 0.0);
            GaugeReadyEvaluation best_gauge = original_gauge;
            double best_gauge_score = gauge_ready_score(original_gauge.metrics);

            for (int angle = 0; angle < 180; ++angle) {
                RotatedBasisMetrics trial =
                    evaluate_rotation(case_spec.grid, hv, raw, static_cast<double>(angle));
                GaugeReadyEvaluation gauge_trial =
                    evaluate_gauge_ready(case_spec.grid, hv, raw, static_cast<double>(angle));
                rotation << case_spec.name << "," << mu << "," << trial.angle_deg << ","
                         << trial.lambda1 << "," << trial.lambda2 << "," << trial.residual1 << ","
                         << trial.residual2 << "," << trial.norm1 << "," << trial.norm2 << ","
                         << trial.orthogonality << "," << (trial.gauge_ready ? 1 : 0) << ","
                         << trial.quality.rms_ri1 << "," << trial.quality.rms_ri2 << ","
                         << trial.quality.rel_rms_mismatch << "," << trial.quality.mean_abs_cos
                         << "," << trial.quality.degeneracy_fraction << "," << trial.combined_score
                         << "," << gauge_trial.metrics.alpha_opt << ","
                         << gauge_trial.metrics.rel_residual_before_gauge << ","
                         << gauge_trial.metrics.rel_residual_after_gauge << ","
                         << gauge_trial.metrics.residual_floor_rel << "\n";
                const double trial_gauge_score = gauge_ready_score(gauge_trial.metrics);
                if (trial_gauge_score < best_gauge_score) {
                    best = trial;
                    best_gauge = std::move(gauge_trial);
                    best_gauge_score = trial_gauge_score;
                }
            }

            const auto write_summary_row = [&](const char* basis_kind,
                                               const RotatedBasisMetrics& m) {
                const double eig0 =
                    solve.result.eigenvalues.size() > 0 ? solve.result.eigenvalues[0] : 0.0;
                const double eig1 =
                    solve.result.eigenvalues.size() > 1 ? solve.result.eigenvalues[1] : 0.0;
                const double eig2 =
                    solve.result.eigenvalues.size() > 2 ? solve.result.eigenvalues[2] : 0.0;
                const double gap01 = eig1 - eig0;
                const double gap12 = eig2 - eig1;
                const double cap0 =
                    solve.expected_captures.size() > 0 ? solve.expected_captures[0] : -1.0;
                const double cap1 =
                    solve.expected_captures.size() > 1 ? solve.expected_captures[1] : -1.0;

                summary << case_spec.name << "," << mu << "," << basis_kind << "," << m.angle_deg
                        << "," << eig0 << "," << eig1 << "," << eig2 << "," << gap01 << "," << gap12
                        << "," << subspace_similarity << "," << solve.modal_quality.orthogonality
                        << "," << m.residual1 << "," << m.residual2 << ","
                        << (m.gauge_ready ? 1 : 0) << "," << cap0 << "," << cap1 << ","
                        << m.quality.rms_vdotgrad1 << "," << m.quality.max_vdotgrad1 << ","
                        << m.quality.rms_vdotgrad2 << "," << m.quality.max_vdotgrad2 << ","
                        << m.quality.rms_ri1 << "," << m.quality.max_ri1 << "," << m.quality.rms_ri2
                        << "," << m.quality.max_ri2 << "," << m.quality.rms_mismatch << ","
                        << m.quality.max_mismatch << "," << m.quality.rel_rms_mismatch << ","
                        << m.quality.mean_abs_cos << "," << m.quality.max_abs_cos << ","
                        << m.quality.degeneracy_fraction << "," << m.quality.masked_fraction << ","
                        << m.quality.mean_speed << "," << m.quality.max_speed << ","
                        << m.quality.low_vel_fraction << "," << m.quality.low_vel_threshold << ","
                        << m.combined_score << "\n";

                local << case_spec.name << "," << mu << "," << basis_kind << "," << m.angle_deg
                      << ",low_velocity," << m.quality.low_vel_fraction << ","
                      << m.quality.rms_vdotgrad1_low_vel << "," << m.quality.rms_vdotgrad2_low_vel
                      << "," << m.quality.rel_rms_mismatch_low_vel << ","
                      << m.quality.mean_abs_cos_low_vel << "\n";
                local << case_spec.name << "," << mu << "," << basis_kind << "," << m.angle_deg
                      << ",high_velocity," << (1.0 - m.quality.low_vel_fraction) << ","
                      << m.quality.rms_vdotgrad1_high_vel << "," << m.quality.rms_vdotgrad2_high_vel
                      << "," << m.quality.rel_rms_mismatch_high_vel << ","
                      << m.quality.mean_abs_cos_high_vel << "\n";
                local << case_spec.name << "," << mu << "," << basis_kind << "," << m.angle_deg
                      << ",degenerate," << m.quality.degeneracy_fraction << ",0,0,"
                      << m.quality.rel_rms_mismatch_degenerate << ","
                      << m.quality.mean_abs_cos_degenerate << "\n";
                local << case_spec.name << "," << mu << "," << basis_kind << "," << m.angle_deg
                      << ",nondegenerate," << (1.0 - m.quality.degeneracy_fraction) << ",0,0,"
                      << m.quality.rel_rms_mismatch_nondegenerate << ","
                      << m.quality.mean_abs_cos_nondegenerate << "\n";
            };

            write_summary_row("original", original);
            write_summary_row("best_rotation", best);

            const auto write_gauge_row = [&](const char* basis_kind,
                                             const GaugeReadyEvaluation& g_eval) {
                const auto& g = g_eval.metrics;
                gauge << case_spec.name << "," << mu << "," << basis_kind << "," << g.angle_deg
                      << "," << g.mean_psi1 << "," << g.mean_psi2 << "," << g.orientation_sign
                      << "," << g.symmetric_scale << "," << g.alpha_opt << "," << g.v_norm << ","
                      << g.cross_norm << "," << g.cross_norm_after_gauge << "," << g.v_dot_cross
                      << "," << g.cos_v_cross << "," << g.rel_residual_before_gauge << ","
                      << g.rel_residual_after_gauge << "," << g.residual_floor_rel << ","
                      << g.rms_vdotgrad1 << "," << g.rms_vdotgrad2 << "," << g.rms_ri1 << ","
                      << g.rms_ri2 << "," << g.mean_abs_cos << "," << g.degeneracy_fraction << "\n";
            };

            write_gauge_row("original", original_gauge);
            write_gauge_row("best_rotation", best_gauge);

            const auto write_energy_rows = [&](const char* basis_kind, double angle_deg,
                                               double alpha_opt,
                                               const std::vector<ModalEnergyRow>& rows) {
                for (const auto& row : rows) {
                    energy << case_spec.name << "," << mu << "," << basis_kind << "," << angle_deg
                           << "," << alpha_opt << "," << row.mode_index << ","
                           << row.eigenvalue_solver << "," << row.psi_norm_sq << ","
                           << row.e_transport << "," << row.e_regularization << "," << row.e_total
                           << "," << row.rayleigh_recomputed << "," << row.f_transport << ","
                           << row.f_regularization << "," << row.residual_Ax_lambda_x_rel << "\n";
                }
            };

            const std::vector<ModalEnergyRow> eigenmode_energy =
                compute_modal_energy(D, L, mu, solve.eigenvectors, solve.result.eigenvalues, ctx);
            write_energy_rows("solver_modes", 0.0, 1.0, eigenmode_energy);

            std::vector<DeviceBuffer<real>> gauge_vectors;
            gauge_vectors.push_back(copy_host_to_device(best_gauge.psi1));
            gauge_vectors.push_back(copy_host_to_device(best_gauge.psi2));
            const std::vector<ModalEnergyRow> gauge_energy =
                compute_modal_energy(D, L, mu, gauge_vectors, {}, ctx);
            write_energy_rows("gauge_ready_best_rotation", best_gauge.metrics.angle_deg,
                              best_gauge.metrics.alpha_opt, gauge_energy);

            const auto best_localization = compute_localization_v2(case_spec.grid, hv, best_gauge);
            for (const auto& row : best_localization) {
                local_v2 << case_spec.name << "," << mu << ",gauge_ready_best_rotation,"
                         << best_gauge.metrics.angle_deg << "," << best_gauge.metrics.alpha_opt
                         << "," << row.region << "," << row.fraction << "," << row.cell_count << ","
                         << row.rms_vdotgrad1 << "," << row.rms_vdotgrad2 << ","
                         << row.rel_residual_after_gauge << "," << row.mean_abs_cos << "\n";
            }

            const double improvement = (gauge_ready_score(original_gauge.metrics) > 1e-30)
                                           ? (gauge_ready_score(original_gauge.metrics) -
                                              gauge_ready_score(best_gauge.metrics)) /
                                                 gauge_ready_score(original_gauge.metrics)
                                           : 0.0;

            auto find_region = [&](const char* name) {
                return std::find_if(
                    best_localization.begin(), best_localization.end(),
                    [&](const LocalizationStats& row) { return row.region == name; });
            };
            const auto boundary_it = find_region("x_boundary_halo");
            const auto interior_it = find_region("x_interior");

            std::printf("    eig=[%.4e, %.4e, %.4e] gap12=%.3e subspace(mu_ref)=%.3f\n",
                        solve.result.eigenvalues[0], solve.result.eigenvalues[1],
                        solve.result.eigenvalues.size() > 2 ? solve.result.eigenvalues[2] : 0.0,
                        (solve.result.eigenvalues.size() > 2
                             ? solve.result.eigenvalues[2] - solve.result.eigenvalues[1]
                             : 0.0),
                        subspace_similarity);
            std::printf("    original: ri=[%.3e, %.3e] rel_mismatch=%.3e mean|cos|=%.3e deg=%.3f "
                        "score=%.3e\n",
                        original.quality.rms_ri1, original.quality.rms_ri2,
                        original.quality.rel_rms_mismatch, original.quality.mean_abs_cos,
                        original.quality.degeneracy_fraction, original.combined_score);
            std::printf("    best rot: angle=%.1f ri=[%.3e, %.3e] rel_mismatch=%.3e mean|cos|=%.3e "
                        "deg=%.3f score=%.3e improve=%.1f%%\n",
                        best.angle_deg, best.quality.rms_ri1, best.quality.rms_ri2,
                        best.quality.rel_rms_mismatch, best.quality.mean_abs_cos,
                        best.quality.degeneracy_fraction, best.combined_score, 100.0 * improvement);
            std::printf(
                "    gauge(original): alpha=%.3e rel_before=%.3e rel_after=%.3e floor=%.3e\n",
                original_gauge.metrics.alpha_opt, original_gauge.metrics.rel_residual_before_gauge,
                original_gauge.metrics.rel_residual_after_gauge,
                original_gauge.metrics.residual_floor_rel);
            std::printf("    gauge(best): angle=%.1f alpha=%.3e rel_before=%.3e rel_after=%.3e "
                        "floor=%.3e\n",
                        best_gauge.metrics.angle_deg, best_gauge.metrics.alpha_opt,
                        best_gauge.metrics.rel_residual_before_gauge,
                        best_gauge.metrics.rel_residual_after_gauge,
                        best_gauge.metrics.residual_floor_rel);
            if (gauge_energy.size() >= 2) {
                std::printf("    energy(gauge): mode0 fD=%.3f fL=%.3f | mode1 fD=%.3f fL=%.3f\n",
                            gauge_energy[0].f_transport, gauge_energy[0].f_regularization,
                            gauge_energy[1].f_transport, gauge_energy[1].f_regularization);
            }
            if (boundary_it != best_localization.end() && interior_it != best_localization.end()) {
                std::printf(
                    "    local(gauge): x_boundary rel_after=%.3e | x_interior rel_after=%.3e\n",
                    boundary_it->rel_residual_after_gauge, interior_it->rel_residual_after_gauge);
            }
        }
        std::printf("\n");
    }

    std::printf("Done.\n");
    return 0;
}

#else

#include <cstdio>
int main() {
    std::printf("analyze_invariant_quality: PETSc not enabled. Skipping.\n");
    return 0;
}

#endif
