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
          "combined_score\n";
}

static void write_local_header(std::ofstream& os) {
    os << "case,mu,basis_kind,angle_deg,region,fraction,rms_vdotgrad1,rms_vdotgrad2,"
          "rel_rms_mismatch,mean_abs_cos\n";
}

} // namespace

int main() {
    runtime::PetscSlepcInit::ensure();
    CudaContext ctx(0);

    std::filesystem::create_directories("artifacts/gate3");
    std::ofstream summary("artifacts/gate3/invariant_quality_summary.csv");
    std::ofstream rotation("artifacts/gate3/invariant_quality_rotation_scan.csv");
    std::ofstream local("artifacts/gate3/invariant_quality_localization.csv");
    write_summary_header(summary);
    write_rotation_header(rotation);
    write_local_header(local);

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

            for (int angle = 0; angle < 180; ++angle) {
                RotatedBasisMetrics trial =
                    evaluate_rotation(case_spec.grid, hv, raw, static_cast<double>(angle));
                rotation << case_spec.name << "," << mu << "," << trial.angle_deg << ","
                         << trial.lambda1 << "," << trial.lambda2 << "," << trial.residual1 << ","
                         << trial.residual2 << "," << trial.norm1 << "," << trial.norm2 << ","
                         << trial.orthogonality << "," << (trial.gauge_ready ? 1 : 0) << ","
                         << trial.quality.rms_ri1 << "," << trial.quality.rms_ri2 << ","
                         << trial.quality.rel_rms_mismatch << "," << trial.quality.mean_abs_cos
                         << "," << trial.quality.degeneracy_fraction << "," << trial.combined_score
                         << "\n";
                if (trial.combined_score < best.combined_score)
                    best = trial;
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

            const double improvement =
                (original.combined_score > 1e-30)
                    ? (original.combined_score - best.combined_score) / original.combined_score
                    : 0.0;

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
