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
#include "src/physics/particles/par2_adapter/par2_views.hpp"
#include "src/physics/particles/pspta/invariants/EigensolverBackend.cuh"
#include "src/physics/particles/pspta/invariants/InvariantPairSearch.hpp"
#include "src/physics/particles/pspta/invariants/PsptaInvariantField.cuh"
#include "src/physics/particles/pspta/invariants/RefinementAC.cuh"
#include "src/physics/particles/pspta/invariants/SLEPcBackend.cuh"
#include "src/physics/particles/pspta/invariants/TransportOperator3D.cuh"
#include "src/physics/particles/pspta/PsptaEngine.hpp"
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
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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

struct ModeFieldData {
    std::vector<double> psi;
    std::vector<double> Apsi;
    std::vector<double> gx;
    std::vector<double> gy;
    std::vector<double> gz;
};

struct RawFieldData {
    int mode_i = -1;
    int mode_j = -1;
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
    double mean_abs_v_cross_cos = 0.0;
    double max_abs_v_cross_cos = 0.0;

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
    int mode_i = -1;
    int mode_j = -1;
    double angle_deg = 0.0;
    double lambda1 = 0.0;
    double lambda2 = 0.0;
    double residual1 = 0.0;
    double residual2 = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    double orthogonality = 0.0;
    bool gauge_ready = false;
    double crossfit_alpha = 1.0;
    double gradient_rms1 = 0.0;
    double gradient_rms2 = 0.0;
    double min_gradient_rms = 0.0;
    double field_range1 = 0.0;
    double field_range2 = 0.0;
    double min_field_range = 0.0;
    double expected_pair_subspace_capture = -1.0;
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
    double expected_full_subspace_capture = -1.0;
    double matrix_free_min_rayleigh = 0.0;
    double exact_action_relerr_mean = 0.0;
    double exact_action_relerr_max = 0.0;
    double probed_action_relerr_mean = 0.0;
    double probed_action_relerr_max = 0.0;
    double exact_rayleigh_relerr_mean = 0.0;
    double exact_rayleigh_relerr_max = 0.0;
    double probed_rayleigh_relerr_mean = 0.0;
    double probed_rayleigh_relerr_max = 0.0;
    double exact_symmetry_defect = 0.0;
    double probed_symmetry_defect = 0.0;
    double exact_min_rayleigh = 0.0;
    double probed_min_rayleigh = 0.0;
    int cluster_count_rel_1p3 = 0;
    int cluster_count_rel_2p0 = 0;
    double prefix2_similarity = -1.0;
    double prefix4_similarity = -1.0;
};

struct ParticleBuffers {
    DeviceBuffer<real> x;
    DeviceBuffer<real> y;
    DeviceBuffer<real> z;
    DeviceBuffer<uint8_t> status;
    DeviceBuffer<int32_t> wrapX;
    DeviceBuffer<int32_t> wrapY;
    DeviceBuffer<int32_t> wrapZ;

    void resize(int n) {
        x.resize(static_cast<size_t>(n));
        y.resize(static_cast<size_t>(n));
        z.resize(static_cast<size_t>(n));
        status.resize(static_cast<size_t>(n));
        wrapX.resize(static_cast<size_t>(n));
        wrapY.resize(static_cast<size_t>(n));
        wrapZ.resize(static_cast<size_t>(n));
    }

    particles::ParticlesSoA<real> view(int n) {
        particles::ParticlesSoA<real> pv;
        pv.x = x.data();
        pv.y = y.data();
        pv.z = z.data();
        pv.n = n;
        pv.status = status.data();
        pv.wrapX = wrapX.data();
        pv.wrapY = wrapY.data();
        pv.wrapZ = wrapZ.data();
        return pv;
    }
};

struct TransportProbeMetrics {
    std::string variant_name;
    std::string engine_semantics;
    int mode_i = -1;
    int mode_j = -1;
    bool host_candidate_admissible = false;
    std::string host_rejection_reason;
    double host_candidate_score = 0.0;
    double host_min_gradient_rms = 0.0;
    double host_min_field_range = 0.0;
    double host_rms_r1 = 0.0;
    double host_rms_r2 = 0.0;
    double host_rel_rms_mismatch = 0.0;
    double host_mean_abs_alignment = 0.0;
    double host_degeneracy_fraction = 0.0;
    bool inlet_gauge_applied = false;
    double rotation_deg = 0.0;
    double scale1 = 1.0;
    double shift1 = 0.0;
    double scale2 = 1.0;
    double shift2 = 0.0;
    bool wrapped_to_periods = false;
    InvariantQualityReport quality;
    PsptaEngine::InvariantPreservationStats preservation_prepare;
    PsptaEngine::InvariantPreservationStats preservation_final;
    PsptaEngine::TransportStats transport;
    int n_particles = 0;
    int n_steps = 0;
    double dt = 0.0;
};

struct StrategyATransformSpec {
    std::string name;
    int mode_i = -1;
    int mode_j = -1;
    double rotation_deg = 0.0;
    double scale1 = 1.0;
    double shift1 = 0.0;
    double scale2 = 1.0;
    double shift2 = 0.0;
    bool wrap_to_periods = false;
};

struct PairTransportPlan {
    RawFieldData raw;
    RotatedBasisMetrics best;
};

struct Subspace4GaugeCandidate {
    std::array<int, 4> mode_ids{{-1, -1, -1, -1}};
    std::array<double, 4> coeff1{{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> coeff2{{0.0, 0.0, 0.0, 0.0}};
    RotatedBasisMetrics host;
};

struct Subspace4TransportPlan {
    RawFieldData raw;
    Subspace4GaugeCandidate candidate;
    CandidateDecision host_decision;
    PairSearchCandidate consumed_summary;
    CandidateDecision consumed_decision;
    InvariantQualityReport consumed_quality;
};

struct StrategyCProbeMetrics {
    std::string init_name;
    int init_rank = -1;
    RefinementACReport refinement;
    TransportProbeMetrics transport;
};

static RotatedBasisMetrics evaluate_rotation(const Grid3D& grid, const HostVelocity& vel,
                                             const RawFieldData& raw, double angle_deg);

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

static double subspace_capture_of_basis(CudaContext& ctx,
                                        const std::vector<DeviceBuffer<real>>& subspace,
                                        const std::vector<DeviceBuffer<real>>& basis) {
    if (subspace.empty() || basis.empty())
        return -1.0;

    blas::ReductionWorkspace ws;
    double capture_sum = 0.0;
    int n_valid_basis = 0;

    for (const auto& basis_vec : basis) {
        const DeviceSpan<const real> bj(basis_vec.data(), basis_vec.size());
        const double norm_b = blas::nrm2_host(ctx, bj, ws);
        if (norm_b <= 1e-30)
            continue;

        double basis_capture = 0.0;
        for (const auto& mode_vec : subspace) {
            const DeviceSpan<const real> ei(mode_vec.data(), mode_vec.size());
            const double norm_e = blas::nrm2_host(ctx, ei, ws);
            if (norm_e <= 1e-30)
                continue;
            const double overlap = blas::dot_host(ctx, ei, bj, ws) / (norm_e * norm_b);
            basis_capture += overlap * overlap;
        }
        capture_sum += std::min(1.0, basis_capture);
        ++n_valid_basis;
    }

    return (n_valid_basis > 0) ? (capture_sum / static_cast<double>(n_valid_basis)) : -1.0;
}

static double eigenspace_similarity_prefix(CudaContext& ctx,
                                           const std::vector<DeviceBuffer<real>>& a,
                                           const std::vector<DeviceBuffer<real>>& b, int n_prefix) {
    const int k = std::min<int>({n_prefix, static_cast<int>(a.size()), static_cast<int>(b.size())});
    if (k <= 0)
        return -1.0;

    blas::ReductionWorkspace ws;
    double sum_sq = 0.0;
    for (int i = 0; i < k; ++i) {
        const DeviceSpan<const real> ai(a[i].data(), a[i].size());
        const double ni = blas::nrm2_host(ctx, ai, ws);
        for (int j = 0; j < k; ++j) {
            const DeviceSpan<const real> bj(b[j].data(), b[j].size());
            const double nj = blas::nrm2_host(ctx, bj, ws);
            if (ni <= 1e-30 || nj <= 1e-30)
                continue;
            const double cij = blas::dot_host(ctx, ai, bj, ws) / (ni * nj);
            sum_sq += cij * cij;
        }
    }
    return sum_sq / static_cast<double>(k);
}

static void fill_vec_from_host(Vec vec, const std::vector<real>& host);
static std::vector<double> copy_vec_to_host(Vec vec, size_t n);

static double sample_min_rayleigh(CombinedOperatorA& A, const Grid3D& grid, CudaContext& ctx,
                                  unsigned seed = 20260420, int n_trials = 5) {
    const size_t n = grid.num_cells();
    ShellContext shell_ctx;
    Mat A_shell = SLEPcBackend::create_shell_operator(A, ctx, shell_ctx);

    Vec x = nullptr;
    Vec Ax = nullptr;
    VecCreateSeqCUDA(PETSC_COMM_SELF, static_cast<PetscInt>(n), &x);
    VecDuplicate(x, &Ax);

    std::vector<real> h_x(n);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    double min_rayleigh = std::numeric_limits<double>::infinity();

    for (int trial = 0; trial < n_trials; ++trial) {
        for (size_t i = 0; i < n; ++i)
            h_x[i] = static_cast<real>(dist(rng));

        fill_vec_from_host(x, h_x);
        MatMult(A_shell, x, Ax);

        PetscScalar xx = 0.0;
        PetscScalar xAx = 0.0;
        VecDot(x, x, &xx);
        VecDot(x, Ax, &xAx);
        const double rayleigh =
            static_cast<double>(xAx) / std::max(static_cast<double>(xx), 1.0e-30);
        min_rayleigh = std::min(min_rayleigh, rayleigh);
    }

    VecDestroy(&x);
    VecDestroy(&Ax);
    MatDestroy(&A_shell);
    return std::isfinite(min_rayleigh) ? min_rayleigh : 0.0;
}

static void fill_vec_from_host(Vec vec, const std::vector<real>& host) {
    PetscScalar* d_ptr = nullptr;
    VecCUDAGetArrayWrite(vec, &d_ptr);
    cudaMemcpy(d_ptr, host.data(), host.size() * sizeof(real), cudaMemcpyHostToDevice);
    VecCUDARestoreArrayWrite(vec, &d_ptr);
}

static std::vector<double> copy_vec_to_host(Vec vec, size_t n) {
    const PetscScalar* d_ptr = nullptr;
    VecCUDAGetArrayRead(vec, &d_ptr);
    std::vector<real> host_real(n);
    cudaMemcpy(host_real.data(), d_ptr, n * sizeof(real), cudaMemcpyDeviceToHost);
    VecCUDARestoreArrayRead(vec, &d_ptr);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<double>(host_real[i]);
    return out;
}

static double l2_norm(const std::vector<double>& values) {
    double ssq = 0.0;
    for (double value : values)
        ssq += value * value;
    return std::sqrt(ssq);
}

static void accumulate_symmetry_defect(Mat A_mat, Vec x, Vec y, Vec Ax, Vec Ay,
                                       double& max_defect) {
    MatMult(A_mat, x, Ax);
    MatMult(A_mat, y, Ay);

    PetscScalar xAy = 0.0;
    PetscScalar yAx = 0.0;
    VecDot(x, Ay, &xAy);
    VecDot(y, Ax, &yAx);

    const double denom = std::max(
        {std::fabs(static_cast<double>(xAy)), std::fabs(static_cast<double>(yAx)), 1.0e-30});
    const double defect = std::fabs(static_cast<double>(xAy - yAx)) / denom;
    max_defect = std::max(max_defect, defect);
}

static void audit_operator_fidelity(CombinedOperatorA& A, const Grid3D& grid, CudaContext& ctx,
                                    SolveSummary& out, unsigned seed = 20260420, int n_trials = 5) {
    const size_t n = grid.num_cells();
    ShellContext shell_ctx;
    Mat A_shell = SLEPcBackend::create_shell_operator(A, ctx, shell_ctx);
    Mat A_exact = SLEPcBackend::assemble_explicit_operator(A, ctx);
    Mat A_probed = SLEPcBackend::assemble_probed_operator(A, ctx, 0.0);

    Vec x = nullptr;
    Vec y = nullptr;
    Vec Ax_shell = nullptr;
    Vec Ax_exact = nullptr;
    Vec Ax_probed = nullptr;
    Vec diff = nullptr;
    VecCreateSeqCUDA(PETSC_COMM_SELF, static_cast<PetscInt>(n), &x);
    VecDuplicate(x, &y);
    VecDuplicate(x, &Ax_shell);
    VecDuplicate(x, &Ax_exact);
    VecDuplicate(x, &Ax_probed);
    VecDuplicate(x, &diff);

    std::vector<real> h_x(n);
    std::vector<real> h_y(n);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    double sum_exact_action = 0.0;
    double sum_probed_action = 0.0;
    double max_exact_action = 0.0;
    double max_probed_action = 0.0;
    double sum_exact_rayleigh = 0.0;
    double sum_probed_rayleigh = 0.0;
    double max_exact_rayleigh = 0.0;
    double max_probed_rayleigh = 0.0;
    double min_exact_rayleigh = std::numeric_limits<double>::infinity();
    double min_probed_rayleigh = std::numeric_limits<double>::infinity();
    double exact_symmetry = 0.0;
    double probed_symmetry = 0.0;

    for (int trial = 0; trial < n_trials; ++trial) {
        for (size_t i = 0; i < n; ++i) {
            h_x[i] = static_cast<real>(dist(rng));
            h_y[i] = static_cast<real>(dist(rng));
        }

        fill_vec_from_host(x, h_x);
        fill_vec_from_host(y, h_y);
        MatMult(A_shell, x, Ax_shell);
        MatMult(A_exact, x, Ax_exact);
        MatMult(A_probed, x, Ax_probed);

        PetscReal mf_norm_raw = 0.0;
        VecNorm(Ax_shell, NORM_2, &mf_norm_raw);
        const double mf_norm = std::max(static_cast<double>(mf_norm_raw), 1.0e-30);

        VecCopy(Ax_exact, diff);
        VecAXPY(diff, -1.0, Ax_shell);
        PetscReal diff_exact_norm_raw = 0.0;
        VecNorm(diff, NORM_2, &diff_exact_norm_raw);
        const double exact_action = static_cast<double>(diff_exact_norm_raw) / mf_norm;

        VecCopy(Ax_probed, diff);
        VecAXPY(diff, -1.0, Ax_shell);
        PetscReal diff_probed_norm_raw = 0.0;
        VecNorm(diff, NORM_2, &diff_probed_norm_raw);
        const double probed_action = static_cast<double>(diff_probed_norm_raw) / mf_norm;
        sum_exact_action += exact_action;
        sum_probed_action += probed_action;
        max_exact_action = std::max(max_exact_action, exact_action);
        max_probed_action = std::max(max_probed_action, probed_action);

        PetscScalar xx = 0.0;
        PetscScalar xAx_shell = 0.0;
        PetscScalar xAx_exact = 0.0;
        PetscScalar xAx_probed = 0.0;
        VecDot(x, x, &xx);
        VecDot(x, Ax_shell, &xAx_shell);
        VecDot(x, Ax_exact, &xAx_exact);
        VecDot(x, Ax_probed, &xAx_probed);
        const double rayleigh_mf =
            static_cast<double>(xAx_shell) / std::max(static_cast<double>(xx), 1.0e-30);
        const double rayleigh_exact =
            static_cast<double>(xAx_exact) / std::max(static_cast<double>(xx), 1.0e-30);
        const double rayleigh_probed =
            static_cast<double>(xAx_probed) / std::max(static_cast<double>(xx), 1.0e-30);
        const double rayleigh_denom = std::max(std::fabs(rayleigh_mf), 1.0e-30);
        const double exact_rayleigh_err = std::fabs(rayleigh_exact - rayleigh_mf) / rayleigh_denom;
        const double probed_rayleigh_err =
            std::fabs(rayleigh_probed - rayleigh_mf) / rayleigh_denom;

        sum_exact_rayleigh += exact_rayleigh_err;
        sum_probed_rayleigh += probed_rayleigh_err;
        max_exact_rayleigh = std::max(max_exact_rayleigh, exact_rayleigh_err);
        max_probed_rayleigh = std::max(max_probed_rayleigh, probed_rayleigh_err);
        min_exact_rayleigh = std::min(min_exact_rayleigh, rayleigh_exact);
        min_probed_rayleigh = std::min(min_probed_rayleigh, rayleigh_probed);

        accumulate_symmetry_defect(A_exact, x, y, Ax_exact, Ax_probed, exact_symmetry);
        accumulate_symmetry_defect(A_probed, x, y, Ax_exact, Ax_probed, probed_symmetry);
    }

    out.exact_action_relerr_mean = sum_exact_action / static_cast<double>(n_trials);
    out.exact_action_relerr_max = max_exact_action;
    out.probed_action_relerr_mean = sum_probed_action / static_cast<double>(n_trials);
    out.probed_action_relerr_max = max_probed_action;
    out.exact_rayleigh_relerr_mean = sum_exact_rayleigh / static_cast<double>(n_trials);
    out.exact_rayleigh_relerr_max = max_exact_rayleigh;
    out.probed_rayleigh_relerr_mean = sum_probed_rayleigh / static_cast<double>(n_trials);
    out.probed_rayleigh_relerr_max = max_probed_rayleigh;
    out.exact_symmetry_defect = exact_symmetry;
    out.probed_symmetry_defect = probed_symmetry;
    out.exact_min_rayleigh = std::isfinite(min_exact_rayleigh) ? min_exact_rayleigh : 0.0;
    out.probed_min_rayleigh = std::isfinite(min_probed_rayleigh) ? min_probed_rayleigh : 0.0;

    VecDestroy(&x);
    VecDestroy(&y);
    VecDestroy(&Ax_shell);
    VecDestroy(&Ax_exact);
    VecDestroy(&Ax_probed);
    VecDestroy(&diff);
    MatDestroy(&A_shell);
    MatDestroy(&A_exact);
    MatDestroy(&A_probed);
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
    out.matrix_free_min_rayleigh = sample_min_rayleigh(A, case_spec.grid, ctx);
    audit_operator_fidelity(A, case_spec.grid, ctx, out);

    auto backend = create_eigensolver_backend("slepc");
    if (!backend)
        throw std::runtime_error("slepc backend unavailable");

    EigensolverConfig cfg;
    cfg.n_eigenvectors = (case_spec.name == "darcy_small") ? 8 : 6;
    cfg.tolerance = 1.0e-8;
    cfg.max_iterations = 500;
    cfg.verbose = false;

    out.result = backend->solve(A, cfg, ctx, out.eigenvectors);
    if (!out.result.eigenvalues.empty()) {
        const double lambda0 = std::max(out.result.eigenvalues.front(), 0.0);
        const double tol = 1.0e-12;
        for (double lambda : out.result.eigenvalues) {
            if (lambda <= 1.3 * lambda0 + tol)
                ++out.cluster_count_rel_1p3;
            if (lambda <= 2.0 * lambda0 + tol)
                ++out.cluster_count_rel_2p0;
        }
    }

    PsptaInvariantField inv;
    inv.resize(case_spec.grid);
    inv.ingest_eigenvectors(out.eigenvectors[0], out.eigenvectors[1], out.result, mu,
                            backend->name(), ctx, ctx.cuda_stream());
    out.modal_quality = inv.modal_quality();
    out.construction_info = inv.construction_info();

    if (case_spec.expected_yz_subspace) {
        auto basis = make_expected_yz_basis(case_spec.grid);
        const int n_captured =
            std::min<int>(out.result.n_converged, static_cast<int>(out.eigenvectors.size()));
        for (int i = 0; i < n_captured; ++i)
            out.expected_captures.push_back(subspace_capture(ctx, out.eigenvectors[i], basis));
        std::vector<DeviceBuffer<real>> captured_subspace;
        captured_subspace.reserve(static_cast<size_t>(n_captured));
        for (int i = 0; i < n_captured; ++i) {
            captured_subspace.emplace_back(out.eigenvectors[i].size());
            cudaMemcpy(captured_subspace.back().data(), out.eigenvectors[i].data(),
                       out.eigenvectors[i].size() * sizeof(real), cudaMemcpyDeviceToDevice);
        }
        out.expected_full_subspace_capture =
            subspace_capture_of_basis(ctx, captured_subspace, basis);
    }

    return out;
}

static double max_speed(const HostVelocity& hv) {
    double vmax = 0.0;
    for (size_t idx = 0; idx < hv.vx.size(); ++idx) {
        const double speed =
            std::sqrt(hv.vx[idx] * hv.vx[idx] + hv.vy[idx] * hv.vy[idx] + hv.vz[idx] * hv.vz[idx]);
        vmax = std::max(vmax, speed);
    }
    return vmax;
}

static void rotate_pair_host(const RawFieldData& raw, double angle_deg, std::vector<double>& psi1,
                             std::vector<double>& psi2) {
    const double theta = angle_deg * M_PI / 180.0;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    psi1.resize(raw.psi1.size());
    psi2.resize(raw.psi2.size());
    for (size_t idx = 0; idx < raw.psi1.size(); ++idx) {
        psi1[idx] = c * raw.psi1[idx] - s * raw.psi2[idx];
        psi2[idx] = s * raw.psi1[idx] + c * raw.psi2[idx];
    }
}

static std::pair<double, double> minmax_pair(const std::vector<double>& values) {
    if (values.empty())
        return {0.0, 0.0};
    auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
    return {*lo_it, *hi_it};
}

static double wrap_period_host(double value, double period) {
    if (period <= 0.0)
        return value;
    value = std::fmod(value, period);
    if (value < 0.0)
        value += period;
    return value;
}

static void upload_host_pair(const std::vector<double>& psi1, const std::vector<double>& psi2,
                             PsptaInvariantField& inv, cudaStream_t stream) {
    std::vector<float> h1(psi1.size());
    std::vector<float> h2(psi2.size());
    for (size_t idx = 0; idx < psi1.size(); ++idx) {
        h1[idx] = static_cast<float>(psi1[idx]);
        h2[idx] = static_cast<float>(psi2[idx]);
    }
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi1_buffer().data(), h1.data(),
                                           h1.size() * sizeof(float), cudaMemcpyHostToDevice,
                                           stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi2_buffer().data(), h2.data(),
                                           h2.size() * sizeof(float), cudaMemcpyHostToDevice,
                                           stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
}

static PsptaInvariantField build_strategy_a_field(const CaseSpec& case_spec,
                                                  const SolveSummary& solve,
                                                  const RawFieldData& raw,
                                                  const StrategyATransformSpec& spec,
                                                  CudaContext& ctx) {
    PsptaInvariantField inv;
    inv.resize(case_spec.grid);
    std::vector<double> psi1_host;
    std::vector<double> psi2_host;
    rotate_pair_host(raw, spec.rotation_deg, psi1_host, psi2_host);

    for (size_t idx = 0; idx < psi1_host.size(); ++idx) {
        psi1_host[idx] = spec.scale1 * psi1_host[idx] + spec.shift1;
        psi2_host[idx] = spec.scale2 * psi2_host[idx] + spec.shift2;
        if (spec.wrap_to_periods) {
            psi1_host[idx] = wrap_period_host(psi1_host[idx], case_spec.grid.Ly());
            psi2_host[idx] = wrap_period_host(psi2_host[idx], case_spec.grid.Lz());
        }
    }

    upload_host_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());

    InvariantConstructionInfo info = solve.construction_info;
    info.inlet_gauge_applied = false;
    info.gauge_method = spec.wrap_to_periods ? "affine_period_fit" : "none";
    if (!info.notes.empty())
        info.notes += "; ";
    info.notes +=
        "transport_variant=" + spec.name + ", mode_i=" + std::to_string(spec.mode_i) +
        ", mode_j=" + std::to_string(spec.mode_j) +
        ", rotation_deg=" + std::to_string(spec.rotation_deg) +
        ", scale1=" + std::to_string(spec.scale1) + ", shift1=" + std::to_string(spec.shift1) +
        ", scale2=" + std::to_string(spec.scale2) + ", shift2=" + std::to_string(spec.shift2) +
        ", wrap_to_periods=" + std::string(spec.wrap_to_periods ? "true" : "false");
    inv.set_construction_info(info);

    return inv;
}

static TransportProbeMetrics run_transport_probe(const CaseSpec& case_spec, const HostVelocity& hv,
                                                 PsptaInvariantField& inv,
                                                 const StrategyATransformSpec& spec,
                                                 CudaContext& ctx) {
    TransportProbeMetrics out;
    out.variant_name = spec.name;
    out.engine_semantics = "legacy_LyLz_self_period";
    out.mode_i = spec.mode_i;
    out.mode_j = spec.mode_j;
    out.inlet_gauge_applied = inv.construction_info().inlet_gauge_applied;
    out.rotation_deg = spec.rotation_deg;
    out.scale1 = spec.scale1;
    out.shift1 = spec.shift1;
    out.scale2 = spec.scale2;
    out.shift2 = spec.shift2;
    out.wrapped_to_periods = spec.wrap_to_periods;
    out.quality = inv.compute_quality(case_spec.velocity, ctx.cuda_stream());

    const double vmax = max_speed(hv);
    out.n_particles = 2048;
    out.n_steps = 8;
    out.dt = (vmax > 0.0) ? (0.25 * static_cast<double>(case_spec.grid.dx) / vmax) : 0.0;

    ParticleBuffers buffers;
    buffers.resize(out.n_particles);
    auto pv = buffers.view(out.n_particles);

    PsptaEngine engine(case_spec.grid, ctx.cuda_stream(), 0x5A17AULL);
    engine.bind_velocity(&case_spec.velocity);
    engine.bind_invariants(&inv);
    engine.bind_particles(pv);

    const real x0 = static_cast<real>(0.25 * static_cast<double>(case_spec.grid.Lx()));
    engine.inject_box(x0, static_cast<real>(0.0), static_cast<real>(0.0), x0, case_spec.grid.Ly(),
                      case_spec.grid.Lz(), 0, out.n_particles);
    engine.ensure_tracking();
    engine.prepare();
    out.preservation_prepare = engine.compute_invariant_preservation();

    for (int step = 0; step < out.n_steps; ++step)
        engine.step(static_cast<real>(out.dt));

    out.transport = engine.compute_transport_stats();
    out.preservation_final = engine.compute_invariant_preservation();
    return out;
}

static StrategyCProbeMetrics run_strategy_c_probe(const CaseSpec& case_spec, const HostVelocity& hv,
                                                  PsptaInvariantField& inv,
                                                  const std::string& init_name, int init_rank,
                                                  const std::vector<ModeFieldData>& mode_data,
                                                  const std::array<int, 4>& subspace4_mode_ids,
                                                  CudaContext& ctx) {
    StrategyCProbeMetrics out;
    out.init_name = init_name;
    out.init_rank = init_rank;

    RefinementACConfig cfg;
    cfg.enabled = true;
    cfg.strategy = RefinementACStrategy::SubspaceQuadraticGaussNewtonEngineProxy;
    cfg.max_iterations = 6;
    cfg.omega = 1.0;
    cfg.omega_min = 1.0e-4;
    cfg.max_backtracks = 8;
    cfg.stop_rel_quality = 1.0e-4;
    cfg.stop_abs_quality = 0.0;
    cfg.poisson_tol = 1.0e-8;
    cfg.poisson_max_iter = 400;
    cfg.invariance_weight = 1.0;
    cfg.local_tikhonov = 1.0e-6;
    cfg.max_invariance_growth = 0.25;
    cfg.max_degeneracy_growth = 0.10;
    cfg.gn_lambda_initial = 1.0e-2;
    cfg.gn_lambda_up = 5.0;
    cfg.gn_lambda_down = 0.5;
    cfg.gn_fd_relative_step = 1.0e-3;
    cfg.gn_fd_absolute_step = 1.0e-4;
    cfg.gn_trust_radius_initial = 0.5;
    cfg.gn_trust_radius_min = 1.0e-3;
    cfg.gn_trust_radius_max = 2.0;
    cfg.projection_proxy_yz_weight = 0.25;
    cfg.projection_proxy_cond_weight = 0.50;
    cfg.projection_proxy_cond_floor = 0.15;
    cfg.projection_proxy_acceptance_weight = 0.25;
    cfg.engine_proxy_sample_count = 64;
    cfg.engine_proxy_sample_steps = 2;
    cfg.engine_proxy_fail_weight = 1.0;
    cfg.engine_proxy_iter_weight = 0.10;
    cfg.engine_proxy_residual_weight = 0.10;
    cfg.engine_proxy_low_recip_weight = 0.10;
    cfg.engine_proxy_acceptance_weight = 0.25;
    cfg.engine_proxy_selector_mode = EngineProxySelectorMode::FailFractionLexicographic;
    cfg.verbose = false;

    RefinementAC refinement(case_spec.grid, &case_spec.velocity, cfg);
    GaugeFixerConfig gf_cfg;
    gf_cfg.method = GaugeMethod::None;
    refinement.set_gauge_fixer(std::make_unique<GaugeFixer>(gf_cfg));
    std::vector<std::vector<float>> basis_host;
    basis_host.reserve(subspace4_mode_ids.size());
    for (int mode : subspace4_mode_ids) {
        std::vector<float> field(mode_data.at(static_cast<size_t>(mode)).psi.size(), 0.0f);
        for (size_t idx = 0; idx < field.size(); ++idx)
            field[idx] = static_cast<float>(mode_data.at(static_cast<size_t>(mode)).psi[idx]);
        basis_host.push_back(std::move(field));
    }
    refinement.set_subspace_basis_host(std::move(basis_host));
    out.refinement = refinement.refine(inv, ctx);

    StrategyATransformSpec spec{
        "strategy_c_from_" + init_name, -1, -1, 0.0, 1.0, 0.0, 1.0, 0.0, false};
    out.transport = run_transport_probe(case_spec, hv, inv, spec, ctx);
    return out;
}

static std::vector<ModeFieldData>
prepare_mode_field_data(const CaseSpec& case_spec, CombinedOperatorA& A,
                        const std::vector<DeviceBuffer<real>>& evs, CudaContext& ctx) {
    std::vector<ModeFieldData> out(evs.size());
    for (size_t mode = 0; mode < evs.size(); ++mode) {
        out[mode].psi = copy_device_to_host(evs[mode]);
        apply_operator_host(A, evs[mode], out[mode].Apsi, ctx);
        compute_gradients_periodic_yz(case_spec.grid, out[mode].psi, out[mode].gx, out[mode].gy,
                                      out[mode].gz);
    }
    return out;
}

static RawFieldData prepare_raw_pair_data(const std::vector<ModeFieldData>& modes, int mode_i,
                                          int mode_j) {
    RawFieldData data;
    data.mode_i = mode_i;
    data.mode_j = mode_j;
    data.psi1 = modes[mode_i].psi;
    data.psi2 = modes[mode_j].psi;
    data.Apsi1 = modes[mode_i].Apsi;
    data.Apsi2 = modes[mode_j].Apsi;
    data.g1x = modes[mode_i].gx;
    data.g1y = modes[mode_i].gy;
    data.g1z = modes[mode_i].gz;
    data.g2x = modes[mode_j].gx;
    data.g2y = modes[mode_j].gy;
    data.g2z = modes[mode_j].gz;
    return data;
}

static RawFieldData prepare_subspace_pair_data(const std::vector<ModeFieldData>& modes,
                                               const std::array<int, 4>& mode_ids,
                                               const std::array<double, 4>& coeff1,
                                               const std::array<double, 4>& coeff2) {
    RawFieldData data;
    data.mode_i = -1;
    data.mode_j = -1;
    const size_t n = modes.at(static_cast<size_t>(mode_ids[0])).psi.size();
    data.psi1.assign(n, 0.0);
    data.psi2.assign(n, 0.0);
    data.Apsi1.assign(n, 0.0);
    data.Apsi2.assign(n, 0.0);
    data.g1x.assign(n, 0.0);
    data.g1y.assign(n, 0.0);
    data.g1z.assign(n, 0.0);
    data.g2x.assign(n, 0.0);
    data.g2y.assign(n, 0.0);
    data.g2z.assign(n, 0.0);

    for (int local = 0; local < 4; ++local) {
        const int mode = mode_ids[local];
        if (mode < 0)
            continue;
        const auto& src = modes.at(static_cast<size_t>(mode));
        const double a = coeff1[local];
        const double b = coeff2[local];
        for (size_t idx = 0; idx < n; ++idx) {
            data.psi1[idx] += a * src.psi[idx];
            data.psi2[idx] += b * src.psi[idx];
            data.Apsi1[idx] += a * src.Apsi[idx];
            data.Apsi2[idx] += b * src.Apsi[idx];
            data.g1x[idx] += a * src.gx[idx];
            data.g1y[idx] += a * src.gy[idx];
            data.g1z[idx] += a * src.gz[idx];
            data.g2x[idx] += b * src.gx[idx];
            data.g2y[idx] += b * src.gy[idx];
            data.g2z[idx] += b * src.gz[idx];
        }
    }

    return data;
}

static bool orthonormalize_frame(std::array<double, 4>& a, std::array<double, 4>& b) {
    auto normalize = [](std::array<double, 4>& v) -> bool {
        double norm_sq = 0.0;
        for (double value : v)
            norm_sq += value * value;
        if (norm_sq <= 1.0e-20)
            return false;
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (double& value : v)
            value *= inv_norm;
        return true;
    };

    if (!normalize(a))
        return false;
    double dot_ab = 0.0;
    for (int i = 0; i < 4; ++i)
        dot_ab += a[i] * b[i];
    for (int i = 0; i < 4; ++i)
        b[i] -= dot_ab * a[i];
    if (!normalize(b))
        return false;
    return true;
}

static std::vector<Subspace4GaugeCandidate>
generate_subspace4_candidates(const Grid3D& grid, const HostVelocity& hv,
                              const std::vector<ModeFieldData>& mode_data,
                              const std::array<int, 4>& mode_ids, int n_random, unsigned seed) {
    std::vector<Subspace4GaugeCandidate> out;
    out.reserve(static_cast<size_t>(n_random) + 6);

    auto push_candidate = [&](std::array<double, 4> c1, std::array<double, 4> c2) {
        if (!orthonormalize_frame(c1, c2))
            return;
        Subspace4GaugeCandidate candidate;
        candidate.mode_ids = mode_ids;
        candidate.coeff1 = c1;
        candidate.coeff2 = c2;
        RawFieldData raw = prepare_subspace_pair_data(mode_data, mode_ids, c1, c2);
        candidate.host = evaluate_rotation(grid, hv, raw, 0.0);
        out.push_back(std::move(candidate));
    };

    for (int axis_i = 0; axis_i < 4; ++axis_i) {
        for (int axis_j = axis_i + 1; axis_j < 4; ++axis_j) {
            std::array<double, 4> c1{{0.0, 0.0, 0.0, 0.0}};
            std::array<double, 4> c2{{0.0, 0.0, 0.0, 0.0}};
            c1[axis_i] = 1.0;
            c2[axis_j] = 1.0;
            push_candidate(c1, c2);
        }
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int sample = 0; sample < n_random; ++sample) {
        std::array<double, 4> c1;
        std::array<double, 4> c2;
        for (int i = 0; i < 4; ++i) {
            c1[i] = dist(rng);
            c2[i] = dist(rng);
        }
        push_candidate(c1, c2);
    }

    return out;
}

static PairSearchCandidate make_host_candidate_summary(const RotatedBasisMetrics& host) {
    PairSearchCandidate out;
    out.mode_i = host.mode_i;
    out.mode_j = host.mode_j;
    out.angle_deg = host.angle_deg;
    out.min_gradient_rms = host.min_gradient_rms;
    out.min_field_range = host.min_field_range;
    out.rel_rms_mismatch = host.quality.rel_rms_mismatch;
    out.rms_invariance_sum = host.quality.rms_ri1 + host.quality.rms_ri2;
    out.degeneracy_fraction = host.quality.degeneracy_fraction;
    out.final_drift_max = 0.0;
    out.total_fail = 0;
    out.n_nonzero_fail = 0;
    out.max_fail_count = 0;
    return out;
}

static PairSearchCandidate make_consumed_candidate_summary(const Subspace4GaugeCandidate& candidate,
                                                           const InvariantQualityReport& quality) {
    PairSearchCandidate out;
    out.mode_i = -1;
    out.mode_j = -1;
    out.angle_deg = 0.0;
    out.min_gradient_rms = candidate.host.min_gradient_rms;
    out.min_field_range = candidate.host.min_field_range;
    out.rel_rms_mismatch = quality.cross_product.rel_rms_mismatch;
    out.rms_invariance_sum = quality.invariance.rms_r1 + quality.invariance.rms_r2;
    out.degeneracy_fraction = quality.independence.degeneracy_score;
    out.final_drift_max = 0.0;
    out.total_fail = 0;
    out.n_nonzero_fail = 0;
    out.max_fail_count = 0;
    return out;
}

static CandidateCollapseReference
build_host_collapse_reference(const std::vector<Subspace4GaugeCandidate>& candidates) {
    CandidateCollapseReference ref;
    for (const auto& candidate : candidates) {
        ref.reference_min_gradient_rms =
            std::max(ref.reference_min_gradient_rms, candidate.host.min_gradient_rms);
        ref.reference_min_field_range =
            std::max(ref.reference_min_field_range, candidate.host.min_field_range);
    }
    return ref;
}

static std::vector<Subspace4TransportPlan> evaluate_subspace4_consumed_candidates(
    const CaseSpec& case_spec, const SolveSummary& solve,
    const std::vector<Subspace4GaugeCandidate>& candidates, const CandidateCollapseReference& ref,
    const std::vector<ModeFieldData>& mode_data, CudaContext& ctx) {
    std::vector<Subspace4TransportPlan> out;
    out.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        Subspace4TransportPlan plan;
        plan.raw = prepare_subspace_pair_data(mode_data, candidate.mode_ids, candidate.coeff1,
                                              candidate.coeff2);
        plan.candidate = candidate;
        plan.host_decision =
            evaluate_pair_candidate(make_host_candidate_summary(candidate.host), ref);

        StrategyATransformSpec variant{
            "subspace4_consumed_eval", -1, -1, 0.0, 1.0, 0.0, 1.0, 0.0, false};
        PsptaInvariantField inv = build_strategy_a_field(case_spec, solve, plan.raw, variant, ctx);
        plan.consumed_quality = inv.compute_quality(case_spec.velocity, ctx.cuda_stream());
        plan.consumed_summary = make_consumed_candidate_summary(candidate, plan.consumed_quality);
        plan.consumed_decision = evaluate_pair_candidate(plan.consumed_summary, ref);
        out.push_back(std::move(plan));
    }

    auto rank_less = [&](const Subspace4TransportPlan& lhs, const Subspace4TransportPlan& rhs) {
        return candidate_is_preferred(lhs.consumed_summary, ref, rhs.consumed_summary, ref);
    };
    std::stable_sort(out.begin(), out.end(), rank_less);
    return out;
}

static std::vector<Subspace4TransportPlan>
choose_subspace4_transport_plans(const std::vector<Subspace4TransportPlan>& ranked_consumed_plans,
                                 int max_candidates) {
    std::vector<Subspace4TransportPlan> out;
    out.reserve(static_cast<size_t>(std::max(0, max_candidates)));

    for (const auto& plan : ranked_consumed_plans) {
        if (!plan.consumed_decision.admissible)
            continue;
        out.push_back(plan);
        if (static_cast<int>(out.size()) >= max_candidates)
            break;
    }

    if (!out.empty())
        return out;

    for (size_t idx = 0;
         idx < ranked_consumed_plans.size() && static_cast<int>(out.size()) < max_candidates;
         ++idx) {
        out.push_back(ranked_consumed_plans[idx]);
    }

    return out;
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
    out.mode_i = raw.mode_i;
    out.mode_j = raw.mode_j;
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
    double ssq_grad1 = 0.0;
    double ssq_grad2 = 0.0;
    double ssq_resid1 = 0.0;
    double ssq_resid2 = 0.0;
    double dot_phi1_Aphi1 = 0.0;
    double dot_phi2_Aphi2 = 0.0;
    double sum_cross_vdot = 0.0;
    double sum_cross_sq = 0.0;
    double sum_abs_align = 0.0;

    double psi1_min = std::numeric_limits<double>::infinity();
    double psi1_max = -std::numeric_limits<double>::infinity();
    double psi2_min = std::numeric_limits<double>::infinity();
    double psi2_max = -std::numeric_limits<double>::infinity();

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

        ssq_grad1 += g1mag * g1mag;
        ssq_grad2 += g2mag * g2mag;
        psi1_min = std::min(psi1_min, psi1);
        psi1_max = std::max(psi1_max, psi1);
        psi2_min = std::min(psi2_min, psi2);
        psi2_max = std::max(psi2_max, psi2);

        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        const double cross_mag = std::sqrt(cx * cx + cy * cy + cz * cz);
        const double mismatch =
            std::sqrt((vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) + (vz - cz) * (vz - cz));
        const double rel_mismatch = mismatch / (vmag + 1e-12);
        const double abs_cos =
            std::fabs(g1x * g2x + g1y * g2y + g1z * g2z) / (g1mag * g2mag + 1e-12);
        const double abs_align = (vmag > 1e-12 && cross_mag > 1e-12)
                                     ? std::fabs(vx * cx + vy * cy + vz * cz) / (vmag * cross_mag)
                                     : 0.0;

        ssq_r1 += d1 * d1;
        ssq_r2 += d2 * d2;
        ssq_ri1 += ri1 * ri1;
        ssq_ri2 += ri2 * ri2;
        ssq_mismatch += mismatch * mismatch;
        ssq_rel_mismatch += rel_mismatch * rel_mismatch;
        sum_cos += abs_cos;
        sum_abs_align += abs_align;

        out.quality.max_vdotgrad1 = std::max(out.quality.max_vdotgrad1, std::fabs(d1));
        out.quality.max_vdotgrad2 = std::max(out.quality.max_vdotgrad2, std::fabs(d2));
        out.quality.max_ri1 = std::max(out.quality.max_ri1, ri1);
        out.quality.max_ri2 = std::max(out.quality.max_ri2, ri2);
        out.quality.max_mismatch = std::max(out.quality.max_mismatch, mismatch);
        out.quality.max_abs_cos = std::max(out.quality.max_abs_cos, abs_cos);
        out.quality.max_abs_v_cross_cos = std::max(out.quality.max_abs_v_cross_cos, abs_align);

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
        sum_cross_vdot += vx * cx + vy * cy + vz * cz;
        sum_cross_sq += cx * cx + cy * cy + cz * cz;
    }

    out.quality.rms_vdotgrad1 = std::sqrt(ssq_r1 / std::max<double>(n, 1.0));
    out.quality.rms_vdotgrad2 = std::sqrt(ssq_r2 / std::max<double>(n, 1.0));
    out.quality.rms_ri1 = std::sqrt(ssq_ri1 / std::max<double>(n, 1.0));
    out.quality.rms_ri2 = std::sqrt(ssq_ri2 / std::max<double>(n, 1.0));
    out.quality.rms_mismatch = std::sqrt(ssq_mismatch / std::max<double>(n, 1.0));
    out.quality.rel_rms_mismatch = std::sqrt(ssq_rel_mismatch / std::max<double>(n, 1.0));
    out.quality.mean_abs_v_cross_cos = sum_abs_align / std::max<double>(n, 1.0);
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
    out.gradient_rms1 = std::sqrt(ssq_grad1 / std::max<double>(n, 1.0));
    out.gradient_rms2 = std::sqrt(ssq_grad2 / std::max<double>(n, 1.0));
    out.min_gradient_rms = std::min(out.gradient_rms1, out.gradient_rms2);
    out.field_range1 = psi1_max - psi1_min;
    out.field_range2 = psi2_max - psi2_min;
    out.min_field_range = std::min(out.field_range1, out.field_range2);
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
    if (sum_cross_sq > 1.0e-30)
        out.crossfit_alpha = std::max(0.0, sum_cross_vdot / sum_cross_sq);
    out.combined_score = out.quality.rms_ri1 + out.quality.rms_ri2 + out.quality.rel_rms_mismatch +
                         0.1 * out.quality.mean_abs_cos + 0.1 * out.quality.degeneracy_fraction;
    return out;
}

static void write_summary_header(std::ofstream& os) {
    os << "case,mu,basis_kind,mode_i,mode_j,angle_deg,eig0,eig1,eig2,eig3,eig4,eig5,gap01,gap12,"
          "gap34,pair_gap,"
          "prefix2_similarity,prefix4_similarity,cluster_count_rel_1p3,cluster_count_rel_2p0,"
          "matrix_free_min_rayleigh,exact_min_rayleigh,probed_min_rayleigh,"
          "exact_symmetry_defect,probed_symmetry_defect,"
          "exact_action_relerr_mean,exact_action_relerr_max,probed_action_relerr_mean,probed_"
          "action_relerr_max,"
          "exact_rayleigh_relerr_mean,exact_rayleigh_relerr_max,probed_rayleigh_relerr_mean,probed_"
          "rayleigh_relerr_max,"
          "expected_full_subspace_capture,expected_pair_subspace_capture,"
          "modal_ortho,residual1,residual2,gauge_ready,expected_capture_0,expected_capture_1,"
          "rms_vdotgrad1,max_vdotgrad1,rms_vdotgrad2,max_vdotgrad2,"
          "rms_ri1,max_ri1,rms_ri2,max_ri2,mean_abs_v_cross_cos,max_abs_v_cross_cos,"
          "rms_mismatch,max_mismatch,rel_rms_mismatch,"
          "mean_abs_cos,max_abs_cos,degeneracy_fraction,masked_fraction,mean_speed,max_speed,"
          "low_vel_fraction,low_vel_threshold,gradient_rms1,gradient_rms2,min_gradient_rms,"
          "field_range1,field_range2,min_field_range,combined_score\n";
}

static void write_rotation_header(std::ofstream& os) {
    os << "case,mu,mode_i,mode_j,angle_deg,lambda1,lambda2,residual1,residual2,norm1,norm2,"
          "gradient_rms1,gradient_rms2,min_gradient_rms,field_range1,field_range2,min_field_range,"
          "expected_pair_subspace_capture,"
          "orthogonality,"
          "gauge_ready,rms_ri1,rms_ri2,mean_abs_v_cross_cos,rel_rms_mismatch,mean_abs_cos,"
          "degeneracy_fraction,"
          "combined_score\n";
}

static void write_local_header(std::ofstream& os) {
    os << "case,mu,basis_kind,mode_i,mode_j,angle_deg,region,fraction,rms_vdotgrad1,rms_vdotgrad2,"
          "rel_rms_mismatch,mean_abs_cos\n";
}

static void write_transport_header(std::ofstream& os) {
    os << "case,mu,consumed_field,mode_i,mode_j,engine_semantics,rank_candidate_admissible,"
          "rank_rejection_reason,rank_candidate_score,rank_min_gradient_rms,rank_min_field_range,"
          "rank_rms_r1,rank_rms_r2,rank_rel_rms_mismatch,rank_mean_abs_alignment,rank_degeneracy,"
          "inlet_gauge_applied,"
          "rotation_deg,scale1,shift1,scale2,shift2,wrapped_to_periods,"
          "quality_rms_r1,quality_max_r1,quality_rms_r2,quality_max_r2,"
          "quality_rms_mismatch,quality_max_mismatch,quality_rel_rms_mismatch,"
          "quality_mean_abs_alignment,quality_max_abs_alignment,"
          "quality_mean_cos,quality_max_cos,quality_degeneracy,quality_masked_fraction,"
          "prepare_rms_psi1_drift,prepare_max_psi1_drift,prepare_rms_psi2_drift,prepare_max_psi2_"
          "drift,"
          "final_rms_psi1_drift,final_max_psi1_drift,final_rms_psi2_drift,final_max_psi2_drift,"
          "transport_active,transport_exited,transport_other,total_fail,n_nonzero_fail,max_fail_"
          "count,"
          "n_particles,n_steps,dt\n";
}

static void write_subspace4_header(std::ofstream& os) {
    os << "case,mu,candidate_rank,selection_kind,host_candidate_admissible,host_rejection_reason,"
          "host_candidate_score,mode0,mode1,mode2,mode3,"
          "coeff1_0,coeff1_1,coeff1_2,coeff1_3,coeff2_0,coeff2_1,coeff2_2,coeff2_3,"
          "host_rms_r1,host_rms_r2,host_rel_rms_mismatch,host_mean_abs_alignment,host_degeneracy,"
          "host_min_gradient_rms,host_min_field_range,host_combined_score,"
          "quality_rms_r1,quality_rms_r2,quality_rel_rms_mismatch,quality_mean_abs_alignment,"
          "quality_degeneracy,prepare_drift_max,final_drift_max,total_fail,n_nonzero_fail,max_fail_"
          "count\n";
}

static void write_strategy_c_header(std::ofstream& os) {
    os << "case,mu,init_rank,init_name,initial_rms_r1,initial_rms_r2,initial_rel_rms_mismatch,"
          "initial_mean_abs_alignment,initial_degeneracy,initial_min_gradient_rms,"
          "initial_min_field_range,initial_proj_rel_vx_det_mismatch,initial_proj_mean_recip_"
          "condition,"
          "initial_proj_min_recip_condition,initial_proj_low_recip_condition_fraction,"
          "initial_proj_combined_score,initial_eng_fail_fraction,initial_eng_mean_fail_count,"
          "initial_eng_fail_x_fraction,initial_eng_fail_mid_fraction,initial_eng_fail_new_fraction,"
          "initial_eng_mean_newton_iterations,initial_eng_mean_normalized_final_residual,"
          "initial_eng_low_recip_condition_fraction,initial_eng_combined_score,"
          "iterations_done,converged,stop_reason,total_time_ms,"
          "best_trial_phase,best_trial_rejection_reason,best_trial_rel_mismatch,"
          "best_trial_invariance_sum,best_trial_degeneracy,best_trial_min_gradient_rms,"
          "best_trial_min_field_range,"
          "final_rms_r1,final_rms_r2,final_rel_rms_mismatch,final_mean_abs_alignment,"
          "final_degeneracy,final_min_gradient_rms,final_min_field_range,"
          "final_proj_rel_vx_det_mismatch,final_proj_mean_recip_condition,"
          "final_proj_min_recip_condition,final_proj_low_recip_condition_fraction,"
          "final_proj_combined_score,final_eng_fail_fraction,final_eng_mean_fail_count,"
          "final_eng_fail_x_fraction,final_eng_fail_mid_fraction,final_eng_fail_new_fraction,"
          "final_eng_mean_newton_iterations,final_eng_mean_normalized_final_residual,"
          "final_eng_low_recip_condition_fraction,final_eng_combined_score,"
          "prepare_rms_psi1_drift,prepare_max_psi1_drift,prepare_rms_psi2_drift,prepare_max_psi2_"
          "drift,"
          "final_rms_psi1_drift,final_max_psi1_drift,final_rms_psi2_drift,final_max_psi2_drift,"
          "transport_active,transport_exited,transport_other,total_fail,n_nonzero_fail,max_fail_"
          "count,"
          "n_particles,n_steps,dt\n";
}

} // namespace

int main() {
    runtime::PetscSlepcInit::ensure();
    CudaContext ctx(0);

    {
        std::filesystem::create_directories("artifacts/gate3");
        std::ofstream summary("artifacts/gate3/invariant_quality_summary.csv");
        std::ofstream rotation("artifacts/gate3/invariant_quality_rotation_scan.csv");
        std::ofstream local("artifacts/gate3/invariant_quality_localization.csv");
        std::ofstream transport("artifacts/gate3/invariant_transport_consumed.csv");
        std::ofstream subspace4("artifacts/gate3/invariant_subspace4_gauge_scan.csv");
        std::ofstream strategy_c("artifacts/gate3/strategy_c_refinement.csv");
        write_summary_header(summary);
        write_rotation_header(rotation);
        write_local_header(local);
        write_transport_header(transport);
        write_subspace4_header(subspace4);
        write_strategy_c_header(strategy_c);

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
        std::printf("  artifacts/gate3/invariant_transport_consumed.csv\n\n");
        std::printf("  artifacts/gate3/invariant_subspace4_gauge_scan.csv\n\n");
        std::printf("  artifacts/gate3/strategy_c_refinement.csv\n\n");

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

                const std::vector<ModeFieldData> mode_data =
                    prepare_mode_field_data(case_spec, A, solve.eigenvectors, ctx);
                const HostVelocity hv = copy_cell_center_velocity(case_spec.velocity);
                const std::vector<DeviceBuffer<real>> expected_basis =
                    case_spec.expected_yz_subspace ? make_expected_yz_basis(case_spec.grid)
                                                   : std::vector<DeviceBuffer<real>>{};

                if (!baseline_set) {
                    baseline_subspace.clear();
                    baseline_subspace.reserve(solve.eigenvectors.size());
                    for (const auto& eigenvector : solve.eigenvectors) {
                        baseline_subspace.emplace_back(eigenvector.size());
                        cudaMemcpy(baseline_subspace.back().data(), eigenvector.data(),
                                   eigenvector.size() * sizeof(real), cudaMemcpyDeviceToDevice);
                    }
                    baseline_set = true;
                }

                solve.prefix2_similarity =
                    baseline_set ? eigenspace_similarity_prefix(ctx, baseline_subspace,
                                                                solve.eigenvectors, 2)
                                 : 1.0;
                solve.prefix4_similarity =
                    baseline_set ? eigenspace_similarity_prefix(ctx, baseline_subspace,
                                                                solve.eigenvectors, 4)
                                 : 1.0;

                const auto write_summary_row = [&](const char* basis_kind,
                                                   const RotatedBasisMetrics& m) {
                    const double eig0 =
                        solve.result.eigenvalues.size() > 0 ? solve.result.eigenvalues[0] : 0.0;
                    const double eig1 =
                        solve.result.eigenvalues.size() > 1 ? solve.result.eigenvalues[1] : 0.0;
                    const double eig2 =
                        solve.result.eigenvalues.size() > 2 ? solve.result.eigenvalues[2] : 0.0;
                    const double eig3 =
                        solve.result.eigenvalues.size() > 3 ? solve.result.eigenvalues[3] : 0.0;
                    const double eig4 =
                        solve.result.eigenvalues.size() > 4 ? solve.result.eigenvalues[4] : 0.0;
                    const double eig5 =
                        solve.result.eigenvalues.size() > 5 ? solve.result.eigenvalues[5] : 0.0;
                    const double gap01 = eig1 - eig0;
                    const double gap12 = eig2 - eig1;
                    const double gap34 = eig4 - eig3;
                    const double pair_gap =
                        (m.mode_j >= 0 &&
                         static_cast<size_t>(m.mode_j) < solve.result.eigenvalues.size())
                            ? solve.result.eigenvalues[m.mode_j] -
                                  solve.result.eigenvalues[std::max(m.mode_i, 0)]
                            : 0.0;
                    const double cap0 = (m.mode_i >= 0 && static_cast<size_t>(m.mode_i) <
                                                              solve.expected_captures.size())
                                            ? solve.expected_captures[m.mode_i]
                                            : -1.0;
                    const double cap1 = (m.mode_j >= 0 && static_cast<size_t>(m.mode_j) <
                                                              solve.expected_captures.size())
                                            ? solve.expected_captures[m.mode_j]
                                            : -1.0;

                    summary << case_spec.name << "," << mu << "," << basis_kind << "," << m.mode_i
                            << "," << m.mode_j << "," << m.angle_deg << "," << eig0 << "," << eig1
                            << "," << eig2 << "," << eig3 << "," << eig4 << "," << eig5 << ","
                            << gap01 << "," << gap12 << "," << gap34 << "," << pair_gap << ","
                            << solve.prefix2_similarity << "," << solve.prefix4_similarity << ","
                            << solve.cluster_count_rel_1p3 << "," << solve.cluster_count_rel_2p0
                            << "," << solve.matrix_free_min_rayleigh << ","
                            << solve.exact_min_rayleigh << "," << solve.probed_min_rayleigh << ","
                            << solve.exact_symmetry_defect << "," << solve.probed_symmetry_defect
                            << "," << solve.exact_action_relerr_mean << ","
                            << solve.exact_action_relerr_max << ","
                            << solve.probed_action_relerr_mean << ","
                            << solve.probed_action_relerr_max << ","
                            << solve.exact_rayleigh_relerr_mean << ","
                            << solve.exact_rayleigh_relerr_max << ","
                            << solve.probed_rayleigh_relerr_mean << ","
                            << solve.probed_rayleigh_relerr_max << ","
                            << solve.expected_full_subspace_capture << ","
                            << m.expected_pair_subspace_capture << ","
                            << solve.modal_quality.orthogonality << "," << m.residual1 << ","
                            << m.residual2 << "," << (m.gauge_ready ? 1 : 0) << "," << cap0 << ","
                            << cap1 << "," << m.quality.rms_vdotgrad1 << ","
                            << m.quality.max_vdotgrad1 << "," << m.quality.rms_vdotgrad2 << ","
                            << m.quality.max_vdotgrad2 << "," << m.quality.rms_ri1 << ","
                            << m.quality.max_ri1 << "," << m.quality.rms_ri2 << ","
                            << m.quality.max_ri2 << "," << m.quality.mean_abs_v_cross_cos << ","
                            << m.quality.max_abs_v_cross_cos << "," << m.quality.rms_mismatch << ","
                            << m.quality.max_mismatch << "," << m.quality.rel_rms_mismatch << ","
                            << m.quality.mean_abs_cos << "," << m.quality.max_abs_cos << ","
                            << m.quality.degeneracy_fraction << "," << m.quality.masked_fraction
                            << "," << m.quality.mean_speed << "," << m.quality.max_speed << ","
                            << m.quality.low_vel_fraction << "," << m.quality.low_vel_threshold
                            << "," << m.gradient_rms1 << "," << m.gradient_rms2 << ","
                            << m.min_gradient_rms << "," << m.field_range1 << "," << m.field_range2
                            << "," << m.min_field_range << "," << m.combined_score << "\n";

                    local << case_spec.name << "," << mu << "," << basis_kind << "," << m.mode_i
                          << "," << m.mode_j << "," << m.angle_deg << ",low_velocity,"
                          << m.quality.low_vel_fraction << "," << m.quality.rms_vdotgrad1_low_vel
                          << "," << m.quality.rms_vdotgrad2_low_vel << ","
                          << m.quality.rel_rms_mismatch_low_vel << ","
                          << m.quality.mean_abs_cos_low_vel << "\n";
                    local << case_spec.name << "," << mu << "," << basis_kind << "," << m.mode_i
                          << "," << m.mode_j << "," << m.angle_deg << ",high_velocity,"
                          << (1.0 - m.quality.low_vel_fraction) << ","
                          << m.quality.rms_vdotgrad1_high_vel << ","
                          << m.quality.rms_vdotgrad2_high_vel << ","
                          << m.quality.rel_rms_mismatch_high_vel << ","
                          << m.quality.mean_abs_cos_high_vel << "\n";
                    local << case_spec.name << "," << mu << "," << basis_kind << "," << m.mode_i
                          << "," << m.mode_j << "," << m.angle_deg << ",degenerate,"
                          << m.quality.degeneracy_fraction << ",0,0,"
                          << m.quality.rel_rms_mismatch_degenerate << ","
                          << m.quality.mean_abs_cos_degenerate << "\n";
                    local << case_spec.name << "," << mu << "," << basis_kind << "," << m.mode_i
                          << "," << m.mode_j << "," << m.angle_deg << ",nondegenerate,"
                          << (1.0 - m.quality.degeneracy_fraction) << ",0,0,"
                          << m.quality.rel_rms_mismatch_nondegenerate << ","
                          << m.quality.mean_abs_cos_nondegenerate << "\n";
                };

                const int n_modes = std::min<int>(
                    6, std::min<int>(solve.result.n_converged, static_cast<int>(mode_data.size())));
                std::vector<PairTransportPlan> pair_plans;
                pair_plans.reserve(std::max(0, (n_modes * (n_modes - 1)) / 2));

                for (int mode_i = 0; mode_i < n_modes; ++mode_i) {
                    for (int mode_j = mode_i + 1; mode_j < n_modes; ++mode_j) {
                        RawFieldData raw = prepare_raw_pair_data(mode_data, mode_i, mode_j);
                        RotatedBasisMetrics original =
                            evaluate_rotation(case_spec.grid, hv, raw, 0.0);
                        RotatedBasisMetrics best = original;
                        if (!expected_basis.empty()) {
                            std::vector<DeviceBuffer<real>> pair_subspace;
                            pair_subspace.reserve(2);
                            for (int mode_idx : {mode_i, mode_j}) {
                                pair_subspace.emplace_back(solve.eigenvectors[mode_idx].size());
                                cudaMemcpy(pair_subspace.back().data(),
                                           solve.eigenvectors[mode_idx].data(),
                                           solve.eigenvectors[mode_idx].size() * sizeof(real),
                                           cudaMemcpyDeviceToDevice);
                            }
                            const double pair_capture =
                                subspace_capture_of_basis(ctx, pair_subspace, expected_basis);
                            original.expected_pair_subspace_capture = pair_capture;
                            best.expected_pair_subspace_capture = pair_capture;
                        }

                        for (int angle = 0; angle < 180; ++angle) {
                            RotatedBasisMetrics trial = evaluate_rotation(
                                case_spec.grid, hv, raw, static_cast<double>(angle));
                            trial.expected_pair_subspace_capture =
                                original.expected_pair_subspace_capture;
                            rotation
                                << case_spec.name << "," << mu << "," << trial.mode_i << ","
                                << trial.mode_j << "," << trial.angle_deg << "," << trial.lambda1
                                << "," << trial.lambda2 << "," << trial.residual1 << ","
                                << trial.residual2 << "," << trial.norm1 << "," << trial.norm2
                                << "," << trial.gradient_rms1 << "," << trial.gradient_rms2 << ","
                                << trial.min_gradient_rms << "," << trial.field_range1 << ","
                                << trial.field_range2 << "," << trial.min_field_range << ","
                                << trial.expected_pair_subspace_capture << ","
                                << trial.orthogonality << "," << (trial.gauge_ready ? 1 : 0) << ","
                                << trial.quality.rms_ri1 << "," << trial.quality.rms_ri2 << ","
                                << trial.quality.mean_abs_v_cross_cos << ","
                                << trial.quality.rel_rms_mismatch << ","
                                << trial.quality.mean_abs_cos << ","
                                << trial.quality.degeneracy_fraction << "," << trial.combined_score
                                << "\n";
                            if (trial.combined_score < best.combined_score)
                                best = trial;
                        }

                        write_summary_row("pair_original", original);
                        write_summary_row("pair_best_rotation", best);
                        pair_plans.push_back(PairTransportPlan{std::move(raw), best});
                    }
                }

                CandidateCollapseReference collapse_reference;
                for (const auto& plan : pair_plans) {
                    collapse_reference.reference_min_gradient_rms = std::max(
                        collapse_reference.reference_min_gradient_rms, plan.best.min_gradient_rms);
                    collapse_reference.reference_min_field_range = std::max(
                        collapse_reference.reference_min_field_range, plan.best.min_field_range);
                }

                std::printf(
                    "    eig=[%.4e, %.4e, %.4e] n_modes=%d gap12=%.3e subspace(mu_ref)=%.3f\n",
                    solve.result.eigenvalues.size() > 0 ? solve.result.eigenvalues[0] : 0.0,
                    solve.result.eigenvalues.size() > 1 ? solve.result.eigenvalues[1] : 0.0,
                    solve.result.eigenvalues.size() > 2 ? solve.result.eigenvalues[2] : 0.0,
                    n_modes,
                    (solve.result.eigenvalues.size() > 2
                         ? solve.result.eigenvalues[2] - solve.result.eigenvalues[1]
                         : 0.0),
                    solve.prefix2_similarity);
                std::printf("    low-cluster counts: <=1.3*eig0 -> %d, <=2.0*eig0 -> %d, "
                            "prefix4(mu_ref)=%.3f\n",
                            solve.cluster_count_rel_1p3, solve.cluster_count_rel_2p0,
                            solve.prefix4_similarity);
                std::printf(
                    "    collapse-ref: min_grad>=%.3e min_range>=%.3e across %zu pair planes\n",
                    collapse_reference.reference_min_gradient_rms *
                        collapse_reference.min_relative_gradient_rms,
                    collapse_reference.reference_min_field_range *
                        collapse_reference.min_relative_field_range,
                    pair_plans.size());

                bool have_best_transport = false;
                PairSearchCandidate best_transport_candidate;
                CandidateDecision best_transport_decision;
                TransportProbeMetrics best_transport_probe;

                for (const auto& plan : pair_plans) {
                    StrategyATransformSpec variant{"pair_search_best_rotation",
                                                   plan.best.mode_i,
                                                   plan.best.mode_j,
                                                   plan.best.angle_deg,
                                                   1.0,
                                                   0.0,
                                                   1.0,
                                                   0.0,
                                                   false};
                    PsptaInvariantField transport_inv =
                        build_strategy_a_field(case_spec, solve, plan.raw, variant, ctx);
                    TransportProbeMetrics probe =
                        run_transport_probe(case_spec, hv, transport_inv, variant, ctx);

                    PairSearchCandidate candidate;
                    candidate.mode_i = plan.best.mode_i;
                    candidate.mode_j = plan.best.mode_j;
                    candidate.angle_deg = plan.best.angle_deg;
                    candidate.min_gradient_rms = plan.best.min_gradient_rms;
                    candidate.min_field_range = plan.best.min_field_range;
                    candidate.rel_rms_mismatch = probe.quality.cross_product.rel_rms_mismatch;
                    candidate.rms_invariance_sum =
                        probe.quality.invariance.rms_r1 + probe.quality.invariance.rms_r2;
                    candidate.degeneracy_fraction = probe.quality.independence.degeneracy_score;
                    candidate.final_drift_max = std::max(probe.preservation_final.max_psi1_drift,
                                                         probe.preservation_final.max_psi2_drift);
                    candidate.total_fail = probe.transport.total_fail;
                    candidate.n_nonzero_fail = probe.transport.n_nonzero_fail;
                    candidate.max_fail_count = probe.transport.max_fail_count;

                    const CandidateDecision decision =
                        evaluate_pair_candidate(candidate, collapse_reference);
                    probe.host_candidate_admissible = decision.admissible;
                    probe.host_rejection_reason = decision.rejection_reason;
                    probe.host_candidate_score = decision.score;
                    probe.host_min_gradient_rms = candidate.min_gradient_rms;
                    probe.host_min_field_range = candidate.min_field_range;
                    probe.host_rms_r1 = plan.best.quality.rms_ri1;
                    probe.host_rms_r2 = plan.best.quality.rms_ri2;
                    probe.host_rel_rms_mismatch = plan.best.quality.rel_rms_mismatch;
                    probe.host_mean_abs_alignment = plan.best.quality.mean_abs_v_cross_cos;
                    probe.host_degeneracy_fraction = plan.best.quality.degeneracy_fraction;

                    transport << case_spec.name << "," << mu << "," << probe.variant_name << ","
                              << probe.mode_i << "," << probe.mode_j << ","
                              << probe.engine_semantics << ","
                              << (probe.host_candidate_admissible ? 1 : 0) << ","
                              << probe.host_rejection_reason << "," << probe.host_candidate_score
                              << "," << probe.host_min_gradient_rms << ","
                              << probe.host_min_field_range << "," << probe.host_rms_r1 << ","
                              << probe.host_rms_r2 << "," << probe.host_rel_rms_mismatch << ","
                              << probe.host_mean_abs_alignment << ","
                              << probe.host_degeneracy_fraction << ","
                              << (probe.inlet_gauge_applied ? 1 : 0) << "," << probe.rotation_deg
                              << "," << probe.scale1 << "," << probe.shift1 << "," << probe.scale2
                              << "," << probe.shift2 << "," << (probe.wrapped_to_periods ? 1 : 0)
                              << "," << probe.quality.invariance.rms_r1 << ","
                              << probe.quality.invariance.max_r1 << ","
                              << probe.quality.invariance.rms_r2 << ","
                              << probe.quality.invariance.max_r2 << ","
                              << probe.quality.cross_product.rms_mismatch << ","
                              << probe.quality.cross_product.max_mismatch << ","
                              << probe.quality.cross_product.rel_rms_mismatch << ","
                              << probe.quality.cross_product.mean_abs_alignment << ","
                              << probe.quality.cross_product.max_abs_alignment << ","
                              << probe.quality.independence.mean_cos_angle << ","
                              << probe.quality.independence.max_cos_angle << ","
                              << probe.quality.independence.degeneracy_score << ","
                              << probe.quality.masked_fraction << ","
                              << probe.preservation_prepare.rms_psi1_drift << ","
                              << probe.preservation_prepare.max_psi1_drift << ","
                              << probe.preservation_prepare.rms_psi2_drift << ","
                              << probe.preservation_prepare.max_psi2_drift << ","
                              << probe.preservation_final.rms_psi1_drift << ","
                              << probe.preservation_final.max_psi1_drift << ","
                              << probe.preservation_final.rms_psi2_drift << ","
                              << probe.preservation_final.max_psi2_drift << ","
                              << probe.transport.n_active << "," << probe.transport.n_exited << ","
                              << probe.transport.n_other << "," << probe.transport.total_fail << ","
                              << probe.transport.n_nonzero_fail << ","
                              << probe.transport.max_fail_count << "," << probe.n_particles << ","
                              << probe.n_steps << "," << probe.dt << "\n";

                    std::printf("    pair(%d,%d) angle=%.1f admissible=%d rel_mismatch=%.3e "
                                "deg=%.3f driftN=[%.3e, %.3e] fail=%lld nonzero_fail=%u "
                                "min_grad=%.3e min_range=%.3e%s%s\n",
                                probe.mode_i, probe.mode_j, probe.rotation_deg,
                                probe.host_candidate_admissible ? 1 : 0,
                                probe.quality.cross_product.rel_rms_mismatch,
                                probe.quality.independence.degeneracy_score,
                                probe.preservation_final.rms_psi1_drift,
                                probe.preservation_final.rms_psi2_drift, probe.transport.total_fail,
                                probe.transport.n_nonzero_fail, probe.host_min_gradient_rms,
                                probe.host_min_field_range,
                                probe.host_rejection_reason.empty() ? "" : " reject=",
                                probe.host_rejection_reason.empty()
                                    ? ""
                                    : probe.host_rejection_reason.c_str());

                    if (!have_best_transport ||
                        candidate_is_preferred(candidate, collapse_reference,
                                               best_transport_candidate, collapse_reference)) {
                        best_transport_candidate = candidate;
                        best_transport_decision = decision;
                        best_transport_probe = probe;
                        have_best_transport = true;
                    }
                }

                if (have_best_transport) {
                    std::printf("    best transported pair: modes=(%d,%d) angle=%.1f admissible=%d "
                                "reject=%s rel_mismatch=%.3e q_r_sum=%.3e drift_max=%.3e "
                                "fail=%lld nonzero_fail=%u\n",
                                best_transport_candidate.mode_i, best_transport_candidate.mode_j,
                                best_transport_candidate.angle_deg,
                                best_transport_decision.admissible ? 1 : 0,
                                best_transport_decision.rejection_reason.empty()
                                    ? "none"
                                    : best_transport_decision.rejection_reason.c_str(),
                                best_transport_candidate.rel_rms_mismatch,
                                best_transport_candidate.rms_invariance_sum,
                                best_transport_candidate.final_drift_max,
                                best_transport_candidate.total_fail,
                                best_transport_candidate.n_nonzero_fail);
                }

                if (n_modes >= 4) {
                    const std::array<int, 4> subspace4_mode_ids{{0, 1, 2, 3}};
                    const int n_random_subspace = (case_spec.name == "darcy_small") ? 512 : 96;
                    std::vector<Subspace4GaugeCandidate> subspace_candidates =
                        generate_subspace4_candidates(
                            case_spec.grid, hv, mode_data, subspace4_mode_ids, n_random_subspace,
                            static_cast<unsigned>(20260420u +
                                                  static_cast<unsigned>(std::llround(mu * 1.0e8))));
                    for (auto& candidate : subspace_candidates) {
                        candidate.host.mode_i = -1;
                        candidate.host.mode_j = -1;
                        candidate.host.expected_pair_subspace_capture =
                            solve.expected_full_subspace_capture;
                    }

                    const CandidateCollapseReference subspace_ref =
                        build_host_collapse_reference(subspace_candidates);
                    const std::vector<Subspace4TransportPlan> ranked_consumed_plans =
                        evaluate_subspace4_consumed_candidates(
                            case_spec, solve, subspace_candidates, subspace_ref, mode_data, ctx);
                    const std::vector<Subspace4TransportPlan> subspace_plans =
                        choose_subspace4_transport_plans(ranked_consumed_plans, 8);

                    size_t best_host_idx = 0;
                    bool have_host_best = false;
                    PairSearchCandidate best_host_summary;
                    CandidateDecision best_host_decision;

                    for (size_t idx = 0; idx < subspace_candidates.size(); ++idx) {
                        const PairSearchCandidate host_summary =
                            make_host_candidate_summary(subspace_candidates[idx].host);
                        const CandidateDecision host_decision =
                            evaluate_pair_candidate(host_summary, subspace_ref);
                        if (!have_host_best ||
                            candidate_is_preferred(host_summary, subspace_ref, best_host_summary,
                                                   subspace_ref)) {
                            best_host_idx = idx;
                            best_host_summary = host_summary;
                            best_host_decision = host_decision;
                            have_host_best = true;
                        }
                    }

                    if (have_host_best) {
                        write_summary_row("subspace4_host_best",
                                          subspace_candidates[best_host_idx].host);
                        std::printf("    subspace4 host-best: admissible=%d reject=%s "
                                    "rel_mismatch=%.3e q_r_sum=%.3e deg=%.3f "
                                    "min_grad=%.3e min_range=%.3e\n",
                                    best_host_decision.admissible ? 1 : 0,
                                    best_host_decision.rejection_reason.empty()
                                        ? "none"
                                        : best_host_decision.rejection_reason.c_str(),
                                    best_host_summary.rel_rms_mismatch,
                                    best_host_summary.rms_invariance_sum,
                                    best_host_summary.degeneracy_fraction,
                                    best_host_summary.min_gradient_rms,
                                    best_host_summary.min_field_range);
                    }

                    bool have_consumed_best = !ranked_consumed_plans.empty();
                    const Subspace4TransportPlan* best_consumed_plan =
                        have_consumed_best ? &ranked_consumed_plans.front() : nullptr;
                    if (best_consumed_plan) {
                        std::printf(
                            "    subspace4 consumed-best: admissible=%d reject=%s "
                            "rel_mismatch=%.3e q_r_sum=%.3e deg=%.3f "
                            "min_grad=%.3e min_range=%.3e\n",
                            best_consumed_plan->consumed_decision.admissible ? 1 : 0,
                            best_consumed_plan->consumed_decision.rejection_reason.empty()
                                ? "none"
                                : best_consumed_plan->consumed_decision.rejection_reason.c_str(),
                            best_consumed_plan->consumed_summary.rel_rms_mismatch,
                            best_consumed_plan->consumed_summary.rms_invariance_sum,
                            best_consumed_plan->consumed_summary.degeneracy_fraction,
                            best_consumed_plan->consumed_summary.min_gradient_rms,
                            best_consumed_plan->consumed_summary.min_field_range);
                    }

                    for (size_t rank = 0; rank < ranked_consumed_plans.size(); ++rank) {
                        const auto& plan = ranked_consumed_plans[rank];
                        subspace4 << case_spec.name << "," << mu << "," << rank
                                  << ",consumed_ranked,"
                                  << (plan.consumed_decision.admissible ? 1 : 0) << ","
                                  << plan.consumed_decision.rejection_reason << ","
                                  << plan.consumed_decision.score << ","
                                  << plan.candidate.mode_ids[0] << "," << plan.candidate.mode_ids[1]
                                  << "," << plan.candidate.mode_ids[2] << ","
                                  << plan.candidate.mode_ids[3] << "," << plan.candidate.coeff1[0]
                                  << "," << plan.candidate.coeff1[1] << ","
                                  << plan.candidate.coeff1[2] << "," << plan.candidate.coeff1[3]
                                  << "," << plan.candidate.coeff2[0] << ","
                                  << plan.candidate.coeff2[1] << "," << plan.candidate.coeff2[2]
                                  << "," << plan.candidate.coeff2[3] << ","
                                  << plan.candidate.host.quality.rms_ri1 << ","
                                  << plan.candidate.host.quality.rms_ri2 << ","
                                  << plan.candidate.host.quality.rel_rms_mismatch << ","
                                  << plan.candidate.host.quality.mean_abs_v_cross_cos << ","
                                  << plan.candidate.host.quality.degeneracy_fraction << ","
                                  << plan.candidate.host.min_gradient_rms << ","
                                  << plan.candidate.host.min_field_range << ","
                                  << plan.candidate.host.combined_score << ","
                                  << plan.consumed_quality.invariance.rms_r1 << ","
                                  << plan.consumed_quality.invariance.rms_r2 << ","
                                  << plan.consumed_quality.cross_product.rel_rms_mismatch << ","
                                  << plan.consumed_quality.cross_product.mean_abs_alignment << ","
                                  << plan.consumed_quality.independence.degeneracy_score
                                  << ",,,,,\n";
                    }

                    bool have_best_subspace_transport = false;
                    PairSearchCandidate best_subspace_transport_candidate;
                    CandidateDecision best_subspace_transport_decision;
                    TransportProbeMetrics best_subspace_transport_probe;

                    for (size_t rank = 0; rank < subspace_plans.size(); ++rank) {
                        const auto& plan = subspace_plans[rank];
                        StrategyATransformSpec variant{
                            "subspace4_candidate_" + std::to_string(rank),
                            -1,
                            -1,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            false};
                        PsptaInvariantField transport_inv =
                            build_strategy_a_field(case_spec, solve, plan.raw, variant, ctx);
                        TransportProbeMetrics probe =
                            run_transport_probe(case_spec, hv, transport_inv, variant, ctx);

                        PairSearchCandidate candidate;
                        candidate.mode_i = -1;
                        candidate.mode_j = -1;
                        candidate.angle_deg = 0.0;
                        candidate.min_gradient_rms = plan.candidate.host.min_gradient_rms;
                        candidate.min_field_range = plan.candidate.host.min_field_range;
                        candidate.rel_rms_mismatch = probe.quality.cross_product.rel_rms_mismatch;
                        candidate.rms_invariance_sum =
                            probe.quality.invariance.rms_r1 + probe.quality.invariance.rms_r2;
                        candidate.degeneracy_fraction = probe.quality.independence.degeneracy_score;
                        candidate.final_drift_max =
                            std::max(probe.preservation_final.max_psi1_drift,
                                     probe.preservation_final.max_psi2_drift);
                        candidate.total_fail = probe.transport.total_fail;
                        candidate.n_nonzero_fail = probe.transport.n_nonzero_fail;
                        candidate.max_fail_count = probe.transport.max_fail_count;

                        probe.host_candidate_admissible = plan.consumed_decision.admissible;
                        probe.host_rejection_reason = plan.consumed_decision.rejection_reason;
                        probe.host_candidate_score = plan.consumed_decision.score;
                        probe.host_min_gradient_rms = plan.candidate.host.min_gradient_rms;
                        probe.host_min_field_range = plan.candidate.host.min_field_range;
                        probe.host_rms_r1 = plan.consumed_quality.invariance.rms_r1;
                        probe.host_rms_r2 = plan.consumed_quality.invariance.rms_r2;
                        probe.host_rel_rms_mismatch =
                            plan.consumed_quality.cross_product.rel_rms_mismatch;
                        probe.host_mean_abs_alignment =
                            plan.consumed_quality.cross_product.mean_abs_alignment;
                        probe.host_degeneracy_fraction =
                            plan.consumed_quality.independence.degeneracy_score;

                        transport << case_spec.name << "," << mu << "," << probe.variant_name << ","
                                  << probe.mode_i << "," << probe.mode_j << ","
                                  << probe.engine_semantics << ","
                                  << (probe.host_candidate_admissible ? 1 : 0) << ","
                                  << probe.host_rejection_reason << ","
                                  << probe.host_candidate_score << ","
                                  << probe.host_min_gradient_rms << ","
                                  << probe.host_min_field_range << "," << probe.host_rms_r1 << ","
                                  << probe.host_rms_r2 << "," << probe.host_rel_rms_mismatch << ","
                                  << probe.host_mean_abs_alignment << ","
                                  << probe.host_degeneracy_fraction << ","
                                  << (probe.inlet_gauge_applied ? 1 : 0) << ","
                                  << probe.rotation_deg << "," << probe.scale1 << ","
                                  << probe.shift1 << "," << probe.scale2 << "," << probe.shift2
                                  << "," << (probe.wrapped_to_periods ? 1 : 0) << ","
                                  << probe.quality.invariance.rms_r1 << ","
                                  << probe.quality.invariance.max_r1 << ","
                                  << probe.quality.invariance.rms_r2 << ","
                                  << probe.quality.invariance.max_r2 << ","
                                  << probe.quality.cross_product.rms_mismatch << ","
                                  << probe.quality.cross_product.max_mismatch << ","
                                  << probe.quality.cross_product.rel_rms_mismatch << ","
                                  << probe.quality.cross_product.mean_abs_alignment << ","
                                  << probe.quality.cross_product.max_abs_alignment << ","
                                  << probe.quality.independence.mean_cos_angle << ","
                                  << probe.quality.independence.max_cos_angle << ","
                                  << probe.quality.independence.degeneracy_score << ","
                                  << probe.quality.masked_fraction << ","
                                  << probe.preservation_prepare.rms_psi1_drift << ","
                                  << probe.preservation_prepare.max_psi1_drift << ","
                                  << probe.preservation_prepare.rms_psi2_drift << ","
                                  << probe.preservation_prepare.max_psi2_drift << ","
                                  << probe.preservation_final.rms_psi1_drift << ","
                                  << probe.preservation_final.max_psi1_drift << ","
                                  << probe.preservation_final.rms_psi2_drift << ","
                                  << probe.preservation_final.max_psi2_drift << ","
                                  << probe.transport.n_active << "," << probe.transport.n_exited
                                  << "," << probe.transport.n_other << ","
                                  << probe.transport.total_fail << ","
                                  << probe.transport.n_nonzero_fail << ","
                                  << probe.transport.max_fail_count << "," << probe.n_particles
                                  << "," << probe.n_steps << "," << probe.dt << "\n";

                        const double prepare_drift_max =
                            std::max(probe.preservation_prepare.max_psi1_drift,
                                     probe.preservation_prepare.max_psi2_drift);
                        const double final_drift_max =
                            std::max(probe.preservation_final.max_psi1_drift,
                                     probe.preservation_final.max_psi2_drift);
                        subspace4 << case_spec.name << "," << mu << "," << rank
                                  << ",transport_selected,"
                                  << (probe.host_candidate_admissible ? 1 : 0) << ","
                                  << probe.host_rejection_reason << ","
                                  << probe.host_candidate_score << "," << plan.candidate.mode_ids[0]
                                  << "," << plan.candidate.mode_ids[1] << ","
                                  << plan.candidate.mode_ids[2] << "," << plan.candidate.mode_ids[3]
                                  << "," << plan.candidate.coeff1[0] << ","
                                  << plan.candidate.coeff1[1] << "," << plan.candidate.coeff1[2]
                                  << "," << plan.candidate.coeff1[3] << ","
                                  << plan.candidate.coeff2[0] << "," << plan.candidate.coeff2[1]
                                  << "," << plan.candidate.coeff2[2] << ","
                                  << plan.candidate.coeff2[3] << ","
                                  << plan.candidate.host.quality.rms_ri1 << ","
                                  << plan.candidate.host.quality.rms_ri2 << ","
                                  << plan.candidate.host.quality.rel_rms_mismatch << ","
                                  << plan.candidate.host.quality.mean_abs_v_cross_cos << ","
                                  << plan.candidate.host.quality.degeneracy_fraction << ","
                                  << probe.host_min_gradient_rms << ","
                                  << probe.host_min_field_range << ","
                                  << plan.candidate.host.combined_score << ","
                                  << probe.quality.invariance.rms_r1 << ","
                                  << probe.quality.invariance.rms_r2 << ","
                                  << probe.quality.cross_product.rel_rms_mismatch << ","
                                  << probe.quality.cross_product.mean_abs_alignment << ","
                                  << probe.quality.independence.degeneracy_score << ","
                                  << prepare_drift_max << "," << final_drift_max << ","
                                  << probe.transport.total_fail << ","
                                  << probe.transport.n_nonzero_fail << ","
                                  << probe.transport.max_fail_count << "\n";

                        std::printf("    subspace4[%zu] host_rel=%.3e consumed_rel=%.3e "
                                    "host_deg=%.3f consumed_deg=%.3f driftN=[%.3e, %.3e] "
                                    "fail=%lld nonzero_fail=%u\n",
                                    rank, probe.host_rel_rms_mismatch,
                                    probe.quality.cross_product.rel_rms_mismatch,
                                    probe.host_degeneracy_fraction,
                                    probe.quality.independence.degeneracy_score,
                                    probe.preservation_final.rms_psi1_drift,
                                    probe.preservation_final.rms_psi2_drift,
                                    probe.transport.total_fail, probe.transport.n_nonzero_fail);

                        if (!have_best_subspace_transport ||
                            candidate_is_preferred(candidate, subspace_ref,
                                                   best_subspace_transport_candidate,
                                                   subspace_ref)) {
                            best_subspace_transport_candidate = candidate;
                            best_subspace_transport_decision = plan.consumed_decision;
                            best_subspace_transport_probe = probe;
                            have_best_subspace_transport = true;
                        }
                    }

                    if (have_host_best) {
                        const auto& best_host_candidate = subspace_candidates[best_host_idx];
                        subspace4 << case_spec.name << "," << mu << "," << -1 << ",host_best,"
                                  << (best_host_decision.admissible ? 1 : 0) << ","
                                  << best_host_decision.rejection_reason << ","
                                  << best_host_decision.score << ","
                                  << best_host_candidate.mode_ids[0] << ","
                                  << best_host_candidate.mode_ids[1] << ","
                                  << best_host_candidate.mode_ids[2] << ","
                                  << best_host_candidate.mode_ids[3] << ","
                                  << best_host_candidate.coeff1[0] << ","
                                  << best_host_candidate.coeff1[1] << ","
                                  << best_host_candidate.coeff1[2] << ","
                                  << best_host_candidate.coeff1[3] << ","
                                  << best_host_candidate.coeff2[0] << ","
                                  << best_host_candidate.coeff2[1] << ","
                                  << best_host_candidate.coeff2[2] << ","
                                  << best_host_candidate.coeff2[3] << ","
                                  << best_host_candidate.host.quality.rms_ri1 << ","
                                  << best_host_candidate.host.quality.rms_ri2 << ","
                                  << best_host_candidate.host.quality.rel_rms_mismatch << ","
                                  << best_host_candidate.host.quality.mean_abs_v_cross_cos << ","
                                  << best_host_candidate.host.quality.degeneracy_fraction << ","
                                  << best_host_candidate.host.min_gradient_rms << ","
                                  << best_host_candidate.host.min_field_range << ","
                                  << best_host_candidate.host.combined_score << ",,,,,,,,,,\n";
                    }

                    if (have_best_subspace_transport) {
                        std::printf("    best transported subspace4: admissible=%d reject=%s "
                                    "host_rel=%.3e consumed_rel=%.3e host_deg=%.3f "
                                    "consumed_deg=%.3f drift_max=%.3e fail=%lld "
                                    "nonzero_fail=%u\n",
                                    best_subspace_transport_decision.admissible ? 1 : 0,
                                    best_subspace_transport_decision.rejection_reason.empty()
                                        ? "none"
                                        : best_subspace_transport_decision.rejection_reason.c_str(),
                                    best_subspace_transport_probe.host_rel_rms_mismatch,
                                    best_subspace_transport_candidate.rel_rms_mismatch,
                                    best_subspace_transport_probe.host_degeneracy_fraction,
                                    best_subspace_transport_candidate.degeneracy_fraction,
                                    best_subspace_transport_candidate.final_drift_max,
                                    best_subspace_transport_candidate.total_fail,
                                    best_subspace_transport_candidate.n_nonzero_fail);
                    }

                    const int n_strategy_c_inits = (case_spec.name == "darcy_small") ? 3 : 1;
                    for (int rank = 0;
                         rank < std::min<int>(n_strategy_c_inits,
                                              static_cast<int>(ranked_consumed_plans.size()));
                         ++rank) {
                        const auto& plan = ranked_consumed_plans[static_cast<size_t>(rank)];
                        StrategyATransformSpec init_variant{
                            "subspace4_consumed_rank_" + std::to_string(rank),
                            -1,
                            -1,
                            0.0,
                            1.0,
                            0.0,
                            1.0,
                            0.0,
                            false};
                        PsptaInvariantField strategy_c_inv =
                            build_strategy_a_field(case_spec, solve, plan.raw, init_variant, ctx);
                        StrategyCProbeMetrics strategy_c_probe =
                            run_strategy_c_probe(case_spec, hv, strategy_c_inv, init_variant.name,
                                                 rank, mode_data, subspace4_mode_ids, ctx);
                        const RefinementIterReport* last_iter =
                            strategy_c_probe.refinement.history.empty()
                                ? nullptr
                                : &strategy_c_probe.refinement.history.back();

                        strategy_c
                            << case_spec.name << "," << mu << "," << rank << ","
                            << strategy_c_probe.init_name << ","
                            << plan.consumed_quality.invariance.rms_r1 << ","
                            << plan.consumed_quality.invariance.rms_r2 << ","
                            << plan.consumed_quality.cross_product.rel_rms_mismatch << ","
                            << plan.consumed_quality.cross_product.mean_abs_alignment << ","
                            << plan.consumed_quality.independence.degeneracy_score << ","
                            << strategy_c_probe.refinement.initial_min_gradient_rms << ","
                            << strategy_c_probe.refinement.initial_min_field_range << ","
                            << strategy_c_probe.refinement.initial_projection
                                   .rel_rms_vx_det_mismatch
                            << ","
                            << strategy_c_probe.refinement.initial_projection.mean_recip_condition
                            << ","
                            << strategy_c_probe.refinement.initial_projection.min_recip_condition
                            << ","
                            << strategy_c_probe.refinement.initial_projection
                                   .low_recip_condition_fraction
                            << "," << strategy_c_probe.refinement.initial_projection.combined_score
                            << "," << strategy_c_probe.refinement.initial_engine.fail_fraction
                            << "," << strategy_c_probe.refinement.initial_engine.mean_fail_count
                            << "," << strategy_c_probe.refinement.initial_engine.fail_x_fraction
                            << "," << strategy_c_probe.refinement.initial_engine.fail_mid_fraction
                            << "," << strategy_c_probe.refinement.initial_engine.fail_new_fraction
                            << ","
                            << strategy_c_probe.refinement.initial_engine.mean_newton_iterations
                            << ","
                            << strategy_c_probe.refinement.initial_engine
                                   .mean_normalized_final_residual
                            << ","
                            << strategy_c_probe.refinement.initial_engine
                                   .low_recip_condition_fraction
                            << "," << strategy_c_probe.refinement.initial_engine.combined_score
                            << "," << strategy_c_probe.refinement.iterations_done << ","
                            << (strategy_c_probe.refinement.converged ? 1 : 0) << ","
                            << strategy_c_probe.refinement.stop_reason << ","
                            << strategy_c_probe.refinement.total_time_ms << ","
                            << (last_iter ? last_iter->best_trial_phase : "") << ","
                            << (last_iter ? last_iter->best_trial_rejection_reason : "") << ","
                            << (last_iter ? last_iter->best_trial_rel_mismatch : 0.0) << ","
                            << (last_iter ? last_iter->best_trial_invariance_sum : 0.0) << ","
                            << (last_iter ? last_iter->best_trial_degeneracy : 0.0) << ","
                            << (last_iter ? last_iter->best_trial_min_gradient_rms : 0.0) << ","
                            << (last_iter ? last_iter->best_trial_min_field_range : 0.0) << ","
                            << strategy_c_probe.refinement.final_quality.invariance.rms_r1 << ","
                            << strategy_c_probe.refinement.final_quality.invariance.rms_r2 << ","
                            << strategy_c_probe.refinement.final_quality.cross_product
                                   .rel_rms_mismatch
                            << ","
                            << strategy_c_probe.refinement.final_quality.cross_product
                                   .mean_abs_alignment
                            << ","
                            << strategy_c_probe.refinement.final_quality.independence
                                   .degeneracy_score
                            << "," << strategy_c_probe.refinement.final_min_gradient_rms << ","
                            << strategy_c_probe.refinement.final_min_field_range << ","
                            << strategy_c_probe.refinement.final_projection.rel_rms_vx_det_mismatch
                            << ","
                            << strategy_c_probe.refinement.final_projection.mean_recip_condition
                            << ","
                            << strategy_c_probe.refinement.final_projection.min_recip_condition
                            << ","
                            << strategy_c_probe.refinement.final_projection
                                   .low_recip_condition_fraction
                            << "," << strategy_c_probe.refinement.final_projection.combined_score
                            << "," << strategy_c_probe.refinement.final_engine.fail_fraction << ","
                            << strategy_c_probe.refinement.final_engine.mean_fail_count << ","
                            << strategy_c_probe.refinement.final_engine.fail_x_fraction << ","
                            << strategy_c_probe.refinement.final_engine.fail_mid_fraction << ","
                            << strategy_c_probe.refinement.final_engine.fail_new_fraction << ","
                            << strategy_c_probe.refinement.final_engine.mean_newton_iterations
                            << ","
                            << strategy_c_probe.refinement.final_engine
                                   .mean_normalized_final_residual
                            << ","
                            << strategy_c_probe.refinement.final_engine.low_recip_condition_fraction
                            << "," << strategy_c_probe.refinement.final_engine.combined_score << ","
                            << strategy_c_probe.transport.preservation_prepare.rms_psi1_drift << ","
                            << strategy_c_probe.transport.preservation_prepare.max_psi1_drift << ","
                            << strategy_c_probe.transport.preservation_prepare.rms_psi2_drift << ","
                            << strategy_c_probe.transport.preservation_prepare.max_psi2_drift << ","
                            << strategy_c_probe.transport.preservation_final.rms_psi1_drift << ","
                            << strategy_c_probe.transport.preservation_final.max_psi1_drift << ","
                            << strategy_c_probe.transport.preservation_final.rms_psi2_drift << ","
                            << strategy_c_probe.transport.preservation_final.max_psi2_drift << ","
                            << strategy_c_probe.transport.transport.n_active << ","
                            << strategy_c_probe.transport.transport.n_exited << ","
                            << strategy_c_probe.transport.transport.n_other << ","
                            << strategy_c_probe.transport.transport.total_fail << ","
                            << strategy_c_probe.transport.transport.n_nonzero_fail << ","
                            << strategy_c_probe.transport.transport.max_fail_count << ","
                            << strategy_c_probe.transport.n_particles << ","
                            << strategy_c_probe.transport.n_steps << ","
                            << strategy_c_probe.transport.dt << "\n";

                        std::printf("    StrategyC[%d] init_rel=%.3e -> refined_rel=%.3e "
                                    "engine=%.3e -> %.3e proj=%.3e -> %.3e "
                                    "init_q_r_sum=%.3e -> refined_q_r_sum=%.3e "
                                    "fail=%lld nonzero_fail=%u stop=%s best_trial=%s/%s %.3e\n",
                                    rank, plan.consumed_quality.cross_product.rel_rms_mismatch,
                                    strategy_c_probe.refinement.final_quality.cross_product
                                        .rel_rms_mismatch,
                                    strategy_c_probe.refinement.initial_engine.combined_score,
                                    strategy_c_probe.refinement.final_engine.combined_score,
                                    strategy_c_probe.refinement.initial_projection.combined_score,
                                    strategy_c_probe.refinement.final_projection.combined_score,
                                    plan.consumed_quality.invariance.rms_r1 +
                                        plan.consumed_quality.invariance.rms_r2,
                                    strategy_c_probe.refinement.final_quality.invariance.rms_r1 +
                                        strategy_c_probe.refinement.final_quality.invariance.rms_r2,
                                    strategy_c_probe.transport.transport.total_fail,
                                    strategy_c_probe.transport.transport.n_nonzero_fail,
                                    strategy_c_probe.refinement.stop_reason.c_str(),
                                    last_iter ? last_iter->best_trial_phase.c_str() : "none",
                                    last_iter ? last_iter->best_trial_rejection_reason.c_str()
                                              : "none",
                                    last_iter ? last_iter->best_trial_rel_mismatch : 0.0);
                    }
                }
            }
            std::printf("\n");
        }
    }

    ctx.synchronize();
    runtime::PetscSlepcInit::finalize();
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
