/**
 * @file RefinementAC.cu
 * @brief First consumed-object-aware implementation of Strategy C refinement.
 */

#include "../../../../numerics/solvers/cg.cuh"
#include "../../../../runtime/cuda_check.cuh"
#include "../PsptaEngine.hpp"
#include "RefinementAC.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

namespace {

struct FieldStats {
    double gradient_rms1 = 0.0;
    double gradient_rms2 = 0.0;
    double min_gradient_rms = 0.0;
    double field_range1 = 0.0;
    double field_range2 = 0.0;
    double min_field_range = 0.0;
};

__device__ __forceinline__ int idx_cc(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

__device__ __forceinline__ int pmod(int n, int N) {
    return ((n % N) + N) % N;
}

__device__ inline void cell_centered_velocity(const real* __restrict__ U,
                                              const real* __restrict__ V,
                                              const real* __restrict__ W, int i, int j, int k,
                                              int nx, int ny, double& vx, double& vy, double& vz) {
    auto idx_U = [nx, ny](int ii, int jj, int kk) {
        return ii + (nx + 1) * jj + (nx + 1) * ny * kk;
    };
    auto idx_V = [nx, ny](int ii, int jj, int kk) { return ii + nx * jj + nx * (ny + 1) * kk; };
    auto idx_W = [nx, ny](int ii, int jj, int kk) { return ii + nx * jj + nx * ny * kk; };

    vx =
        0.5 * (static_cast<double>(U[idx_U(i, j, k)]) + static_cast<double>(U[idx_U(i + 1, j, k)]));
    vy =
        0.5 * (static_cast<double>(V[idx_V(i, j, k)]) + static_cast<double>(V[idx_V(i, j + 1, k)]));
    vz =
        0.5 * (static_cast<double>(W[idx_W(i, j, k)]) + static_cast<double>(W[idx_W(i, j, k + 1)]));
}

__device__ inline void gradient_with_lifting(const float* __restrict__ psi, int i, int j, int k,
                                             int nx, int ny, int nz, double dx, double dy,
                                             double dz, double L_self, double& gx, double& gy,
                                             double& gz) {
    const int c = idx_cc(i, j, k, nx, ny);
    const double psi_c = static_cast<double>(psi[c]);

    if (i == 0) {
        gx = (static_cast<double>(psi[idx_cc(1, j, k, nx, ny)]) - psi_c) / dx;
    } else if (i == nx - 1) {
        gx = (psi_c - static_cast<double>(psi[idx_cc(nx - 2, j, k, nx, ny)])) / dx;
    } else {
        gx = (static_cast<double>(psi[idx_cc(i + 1, j, k, nx, ny)]) -
              static_cast<double>(psi[idx_cc(i - 1, j, k, nx, ny)])) /
             (2.0 * dx);
    }

    const int jm = pmod(j - 1, ny);
    const int jp = pmod(j + 1, ny);
    double psi_jm = static_cast<double>(psi[idx_cc(i, jm, k, nx, ny)]);
    double psi_jp = static_cast<double>(psi[idx_cc(i, jp, k, nx, ny)]);
    psi_jm += L_self * round((psi_c - psi_jm) / L_self);
    psi_jp += L_self * round((psi_c - psi_jp) / L_self);
    gy = (psi_jp - psi_jm) / (2.0 * dy);

    const int km = pmod(k - 1, nz);
    const int kp = pmod(k + 1, nz);
    double psi_km = static_cast<double>(psi[idx_cc(i, j, km, nx, ny)]);
    double psi_kp = static_cast<double>(psi[idx_cc(i, j, kp, nx, ny)]);
    psi_km += L_self * round((psi_c - psi_km) / L_self);
    psi_kp += L_self * round((psi_c - psi_kp) / L_self);
    gz = (psi_kp - psi_km) / (2.0 * dz);
}

__device__ bool solve_spd_3x3(double H[3][3], double b[3], double x[3]) {
    double A[3][4] = {
        {H[0][0], H[0][1], H[0][2], b[0]},
        {H[1][0], H[1][1], H[1][2], b[1]},
        {H[2][0], H[2][1], H[2][2], b[2]},
    };

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        double pivot_abs = fabs(A[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            const double cand = fabs(A[row][col]);
            if (cand > pivot_abs) {
                pivot = row;
                pivot_abs = cand;
            }
        }
        if (pivot_abs < 1.0e-14) {
            x[0] = x[1] = x[2] = 0.0;
            return false;
        }
        if (pivot != col) {
            for (int k = col; k < 4; ++k) {
                const double tmp = A[col][k];
                A[col][k] = A[pivot][k];
                A[pivot][k] = tmp;
            }
        }
        const double diag = A[col][col];
        for (int k = col; k < 4; ++k)
            A[col][k] /= diag;
        for (int row = 0; row < 3; ++row) {
            if (row == col)
                continue;
            const double factor = A[row][col];
            for (int k = col; k < 4; ++k)
                A[row][k] -= factor * A[col][k];
        }
    }

    x[0] = A[0][3];
    x[1] = A[1][3];
    x[2] = A[2][3];
    return true;
}

__global__ void kernel_compute_delta_gradient_psi1(
    const float* __restrict__ psi1, const float* __restrict__ psi2, const real* __restrict__ U,
    const real* __restrict__ V, const real* __restrict__ W, real* __restrict__ dgx,
    real* __restrict__ dgy, real* __restrict__ dgz, int nx, int ny, int nz, double dx, double dy,
    double dz, double Ly, double Lz, double alpha, double beta) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    double g1x, g1y, g1z, g2x, g2y, g2z;
    gradient_with_lifting(psi1, i, j, k, nx, ny, nz, dx, dy, dz, Ly, g1x, g1y, g1z);
    gradient_with_lifting(psi2, i, j, k, nx, ny, nz, dx, dy, dz, Lz, g2x, g2y, g2z);

    double vx, vy, vz;
    cell_centered_velocity(U, V, W, i, j, k, nx, ny, vx, vy, vz);

    const double cx = g1y * g2z - g1z * g2y;
    const double cy = g1z * g2x - g1x * g2z;
    const double cz = g1x * g2y - g1y * g2x;

    const double r[3] = {vx - cx, vy - cy, vz - cz};
    const double inv0 = vx * g1x + vy * g1y + vz * g1z;

    const double M[3][3] = {
        {0.0, g2z, -g2y},
        {-g2z, 0.0, g2x},
        {g2y, -g2x, 0.0},
    };

    double H[3][3] = {};
    double bvec[3] = {-alpha * inv0 * vx, -alpha * inv0 * vy, -alpha * inv0 * vz};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double value = beta * (row == col ? 1.0 : 0.0) + alpha * (&vx)[row] * (&vx)[col];
            for (int p = 0; p < 3; ++p)
                value += M[p][row] * M[p][col];
            H[row][col] = value;
        }
        for (int p = 0; p < 3; ++p)
            bvec[row] += M[p][row] * r[p];
    }

    double delta[3];
    solve_spd_3x3(H, bvec, delta);
    if (i == 0 || i == nx - 1)
        delta[0] = 0.0;

    dgx[c] = delta[0];
    dgy[c] = delta[1];
    dgz[c] = delta[2];
}

__global__ void kernel_compute_delta_gradient_psi2(
    const float* __restrict__ psi1, const float* __restrict__ psi2, const real* __restrict__ U,
    const real* __restrict__ V, const real* __restrict__ W, real* __restrict__ dgx,
    real* __restrict__ dgy, real* __restrict__ dgz, int nx, int ny, int nz, double dx, double dy,
    double dz, double Ly, double Lz, double alpha, double beta) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    double g1x, g1y, g1z, g2x, g2y, g2z;
    gradient_with_lifting(psi1, i, j, k, nx, ny, nz, dx, dy, dz, Ly, g1x, g1y, g1z);
    gradient_with_lifting(psi2, i, j, k, nx, ny, nz, dx, dy, dz, Lz, g2x, g2y, g2z);

    double vx, vy, vz;
    cell_centered_velocity(U, V, W, i, j, k, nx, ny, vx, vy, vz);

    const double cx = g1y * g2z - g1z * g2y;
    const double cy = g1z * g2x - g1x * g2z;
    const double cz = g1x * g2y - g1y * g2x;

    const double r[3] = {vx - cx, vy - cy, vz - cz};
    const double inv0 = vx * g2x + vy * g2y + vz * g2z;

    const double M[3][3] = {
        {0.0, -g1z, g1y},
        {g1z, 0.0, -g1x},
        {-g1y, g1x, 0.0},
    };

    double H[3][3] = {};
    double bvec[3] = {-alpha * inv0 * vx, -alpha * inv0 * vy, -alpha * inv0 * vz};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double value = beta * (row == col ? 1.0 : 0.0) + alpha * (&vx)[row] * (&vx)[col];
            for (int p = 0; p < 3; ++p)
                value += M[p][row] * M[p][col];
            H[row][col] = value;
        }
        for (int p = 0; p < 3; ++p)
            bvec[row] += M[p][row] * r[p];
    }

    double delta[3];
    solve_spd_3x3(H, bvec, delta);
    if (i == 0 || i == nx - 1)
        delta[0] = 0.0;

    dgx[c] = delta[0];
    dgy[c] = delta[1];
    dgz[c] = delta[2];
}

__global__ void kernel_compute_divergence_rhs(const real* __restrict__ gx,
                                              const real* __restrict__ gy,
                                              const real* __restrict__ gz, real* __restrict__ rhs,
                                              int nx, int ny, int nz, double dx, double dy,
                                              double dz) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const int i = c % nx;
    const int j = (c / nx) % ny;
    const int k = c / (nx * ny);

    double ddx;
    if (i == 0) {
        ddx = (gx[idx_cc(1, j, k, nx, ny)] - gx[idx_cc(0, j, k, nx, ny)]) / dx;
    } else if (i == nx - 1) {
        ddx = (gx[idx_cc(nx - 1, j, k, nx, ny)] - gx[idx_cc(nx - 2, j, k, nx, ny)]) / dx;
    } else {
        ddx = (gx[idx_cc(i + 1, j, k, nx, ny)] - gx[idx_cc(i - 1, j, k, nx, ny)]) / (2.0 * dx);
    }

    const int jm = pmod(j - 1, ny);
    const int jp = pmod(j + 1, ny);
    const int km = pmod(k - 1, nz);
    const int kp = pmod(k + 1, nz);

    const double ddy = (gy[idx_cc(i, jp, k, nx, ny)] - gy[idx_cc(i, jm, k, nx, ny)]) / (2.0 * dy);
    const double ddz = (gz[idx_cc(i, j, kp, nx, ny)] - gz[idx_cc(i, j, km, nx, ny)]) / (2.0 * dz);

    rhs[c] = -(ddx + ddy + ddz);
}

__global__ void kernel_apply_delta_to_field(const float* __restrict__ base,
                                            const real* __restrict__ delta, float* __restrict__ out,
                                            int n, double omega) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = static_cast<float>(static_cast<double>(base[idx]) + omega * delta[idx]);
}

__global__ void kernel_pin_identity_row(const real* __restrict__ x, real* __restrict__ y) {
    if (threadIdx.x + blockIdx.x * blockDim.x != 0)
        return;
    y[0] = x[0];
}

struct PinnedLaplacianCGOperator {
    const LaplacianOperator3D* L = nullptr;

    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        L->apply_L(x, y, ctx.cuda_stream());
        kernel_pin_identity_row<<<1, 1, 0, ctx.cuda_stream()>>>(x.data(), y.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
};

FieldStats compute_field_stats_host(const Grid3D& grid, const PsptaInvariantField& inv,
                                    cudaStream_t stream) {
    FieldStats out;
    const size_t n = inv.num_cells();
    std::vector<float> psi1(n, 0.0f);
    std::vector<float> psi2(n, 0.0f);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(psi1.data(), inv.psi1_ptr(), n * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(psi2.data(), inv.psi2_ptr(), n * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto idx = [grid](int i, int j, int k) {
        return static_cast<size_t>(i) +
               static_cast<size_t>(grid.nx) *
                   (static_cast<size_t>(j) + static_cast<size_t>(grid.ny) * k);
    };
    auto periodic_lift = [](double center, double neighbor, double period) {
        return neighbor + period * std::round((center - neighbor) / period);
    };

    double ssq_grad1 = 0.0;
    double ssq_grad2 = 0.0;
    double psi1_min = std::numeric_limits<double>::infinity();
    double psi1_max = -std::numeric_limits<double>::infinity();
    double psi2_min = std::numeric_limits<double>::infinity();
    double psi2_max = -std::numeric_limits<double>::infinity();

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = idx(i, j, k);
                const double p1 = static_cast<double>(psi1[c]);
                const double p2 = static_cast<double>(psi2[c]);
                psi1_min = std::min(psi1_min, p1);
                psi1_max = std::max(psi1_max, p1);
                psi2_min = std::min(psi2_min, p2);
                psi2_max = std::max(psi2_max, p2);

                auto dx_grad = [&](const std::vector<float>& psi) {
                    if (i == 0)
                        return (static_cast<double>(psi[idx(1, j, k)]) -
                                static_cast<double>(psi[idx(0, j, k)])) /
                               static_cast<double>(grid.dx);
                    if (i == grid.nx - 1)
                        return (static_cast<double>(psi[idx(grid.nx - 1, j, k)]) -
                                static_cast<double>(psi[idx(grid.nx - 2, j, k)])) /
                               static_cast<double>(grid.dx);
                    return (static_cast<double>(psi[idx(i + 1, j, k)]) -
                            static_cast<double>(psi[idx(i - 1, j, k)])) /
                           (2.0 * static_cast<double>(grid.dx));
                };
                auto dy_grad = [&](const std::vector<float>& psi, double center, double period) {
                    const int jm = (j - 1 + grid.ny) % grid.ny;
                    const int jp = (j + 1) % grid.ny;
                    const double pm =
                        periodic_lift(center, static_cast<double>(psi[idx(i, jm, k)]), period);
                    const double pp =
                        periodic_lift(center, static_cast<double>(psi[idx(i, jp, k)]), period);
                    return (pp - pm) / (2.0 * static_cast<double>(grid.dy));
                };
                auto dz_grad = [&](const std::vector<float>& psi, double center, double period) {
                    const int km = (k - 1 + grid.nz) % grid.nz;
                    const int kp = (k + 1) % grid.nz;
                    const double pm =
                        periodic_lift(center, static_cast<double>(psi[idx(i, j, km)]), period);
                    const double pp =
                        periodic_lift(center, static_cast<double>(psi[idx(i, j, kp)]), period);
                    return (pp - pm) / (2.0 * static_cast<double>(grid.dz));
                };

                const double g1x = dx_grad(psi1);
                const double g1y = dy_grad(psi1, p1, inv.Ly());
                const double g1z = dz_grad(psi1, p1, inv.Ly());
                const double g2x = dx_grad(psi2);
                const double g2y = dy_grad(psi2, p2, inv.Lz());
                const double g2z = dz_grad(psi2, p2, inv.Lz());

                ssq_grad1 += g1x * g1x + g1y * g1y + g1z * g1z;
                ssq_grad2 += g2x * g2x + g2y * g2y + g2z * g2z;
            }
        }
    }

    out.gradient_rms1 = std::sqrt(ssq_grad1 / std::max<double>(n, 1.0));
    out.gradient_rms2 = std::sqrt(ssq_grad2 / std::max<double>(n, 1.0));
    out.min_gradient_rms = std::min(out.gradient_rms1, out.gradient_rms2);
    out.field_range1 = psi1_max - psi1_min;
    out.field_range2 = psi2_max - psi2_min;
    out.min_field_range = std::min(out.field_range1, out.field_range2);
    return out;
}

bool trial_admissible(const InvariantQualityReport& trial_quality, const FieldStats& trial_stats,
                      const InvariantQualityReport& current_quality,
                      const InvariantQualityReport& initial_quality,
                      const FieldStats& initial_stats, const RefinementACConfig& cfg) {
    const double current_rel = current_quality.cross_product.rel_rms_mismatch;
    const double trial_rel = trial_quality.cross_product.rel_rms_mismatch;
    const double initial_inv =
        initial_quality.invariance.rms_r1 + initial_quality.invariance.rms_r2;
    const double trial_inv = trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
    const double initial_deg = initial_quality.independence.degeneracy_score;
    const double trial_deg = trial_quality.independence.degeneracy_score;

    const bool mismatch_improved = trial_rel + 1.0e-8 < current_rel;
    const bool invariance_ok =
        trial_inv <= initial_inv * (1.0 + cfg.max_invariance_growth) + 1.0e-12;
    const bool degeneracy_ok = trial_deg <= initial_deg + cfg.max_degeneracy_growth + 1.0e-12;
    const bool gradients_ok = trial_stats.min_gradient_rms >=
                              initial_stats.min_gradient_rms * cfg.min_relative_gradient_rms;
    const bool ranges_ok =
        trial_stats.min_field_range >= initial_stats.min_field_range * cfg.min_relative_field_range;

    return mismatch_improved && invariance_ok && degeneracy_ok && gradients_ok && ranges_ok;
}

std::string rejection_reason(const InvariantQualityReport& trial_quality,
                             const FieldStats& trial_stats,
                             const InvariantQualityReport& current_quality,
                             const InvariantQualityReport& initial_quality,
                             const FieldStats& initial_stats, const RefinementACConfig& cfg) {
    const double current_rel = current_quality.cross_product.rel_rms_mismatch;
    const double trial_rel = trial_quality.cross_product.rel_rms_mismatch;
    if (!(trial_rel + 1.0e-8 < current_rel))
        return "mismatch_not_improved";

    const double initial_inv =
        initial_quality.invariance.rms_r1 + initial_quality.invariance.rms_r2;
    const double trial_inv = trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
    if (!(trial_inv <= initial_inv * (1.0 + cfg.max_invariance_growth) + 1.0e-12))
        return "invariance_growth";

    const double initial_deg = initial_quality.independence.degeneracy_score;
    const double trial_deg = trial_quality.independence.degeneracy_score;
    if (!(trial_deg <= initial_deg + cfg.max_degeneracy_growth + 1.0e-12))
        return "degeneracy_growth";

    if (!(trial_stats.min_gradient_rms >=
          initial_stats.min_gradient_rms * cfg.min_relative_gradient_rms))
        return "gradient_collapse";
    if (!(trial_stats.min_field_range >=
          initial_stats.min_field_range * cfg.min_relative_field_range))
        return "range_collapse";
    return "admissible";
}

bool solve_dense_system(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double pivot_abs = std::fabs(A[static_cast<size_t>(col) * n + col]);
        for (int row = col + 1; row < n; ++row) {
            const double cand = std::fabs(A[static_cast<size_t>(row) * n + col]);
            if (cand > pivot_abs) {
                pivot = row;
                pivot_abs = cand;
            }
        }
        if (pivot_abs < 1.0e-12)
            return false;
        if (pivot != col) {
            for (int k = col; k < n; ++k)
                std::swap(A[static_cast<size_t>(col) * n + k],
                          A[static_cast<size_t>(pivot) * n + k]);
            std::swap(b[col], b[pivot]);
        }
        const double diag = A[static_cast<size_t>(col) * n + col];
        for (int k = col; k < n; ++k)
            A[static_cast<size_t>(col) * n + k] /= diag;
        b[col] /= diag;
        for (int row = 0; row < n; ++row) {
            if (row == col)
                continue;
            const double factor = A[static_cast<size_t>(row) * n + col];
            for (int k = col; k < n; ++k)
                A[static_cast<size_t>(row) * n + k] -= factor * A[static_cast<size_t>(col) * n + k];
            b[row] -= factor * b[col];
        }
    }
    return true;
}

std::vector<float> download_field(const float* d_ptr, size_t n, cudaStream_t stream) {
    std::vector<float> out(n, 0.0f);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(out.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

void upload_field_pair(const std::vector<float>& psi1, const std::vector<float>& psi2,
                       PsptaInvariantField& inv, cudaStream_t stream) {
    const size_t n = psi1.size();
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi1_ptr(), psi1.data(), n * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi2_ptr(), psi2.data(), n * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
}

struct SubspaceFeatureBank {
    int n_modes = 0;
    size_t n_cells = 0;
    std::vector<std::vector<double>> linear_features;
    std::vector<std::vector<double>> quadratic_features;
    std::vector<std::pair<int, int>> quadratic_pairs;
};

struct HostVelocityCache {
    std::vector<real> U;
    std::vector<real> V;
    std::vector<real> W;
};

bool build_subspace_feature_bank(const std::vector<std::vector<float>>& basis_modes,
                                 SubspaceFeatureBank& bank) {
    if (basis_modes.size() < 2)
        return false;
    const size_t n_cells = basis_modes.front().size();
    if (n_cells == 0)
        return false;
    for (const auto& mode : basis_modes) {
        if (mode.size() != n_cells)
            return false;
    }

    bank = {};
    bank.n_modes = static_cast<int>(basis_modes.size());
    bank.n_cells = n_cells;
    bank.linear_features.resize(static_cast<size_t>(bank.n_modes));
    for (int mode = 0; mode < bank.n_modes; ++mode) {
        auto& dst = bank.linear_features[static_cast<size_t>(mode)];
        dst.resize(n_cells, 0.0);
        double mean = 0.0;
        for (float value : basis_modes[static_cast<size_t>(mode)])
            mean += static_cast<double>(value);
        mean /= static_cast<double>(n_cells);
        double rms = 0.0;
        for (size_t idx = 0; idx < n_cells; ++idx) {
            const double centered =
                static_cast<double>(basis_modes[static_cast<size_t>(mode)][idx]) - mean;
            dst[idx] = centered;
            rms += centered * centered;
        }
        rms = std::sqrt(rms / static_cast<double>(n_cells));
        if (rms < 1.0e-12)
            return false;
        for (double& value : dst)
            value /= rms;
    }

    for (int i = 0; i < bank.n_modes; ++i) {
        for (int j = i; j < bank.n_modes; ++j) {
            std::vector<double> q(n_cells, 0.0);
            double mean = 0.0;
            for (size_t idx = 0; idx < n_cells; ++idx) {
                q[idx] = bank.linear_features[static_cast<size_t>(i)][idx] *
                         bank.linear_features[static_cast<size_t>(j)][idx];
                mean += q[idx];
            }
            mean /= static_cast<double>(n_cells);
            double rms = 0.0;
            for (double& value : q) {
                value -= mean;
                rms += value * value;
            }
            rms = std::sqrt(rms / static_cast<double>(n_cells));
            if (rms < 1.0e-12)
                continue;
            for (double& value : q)
                value /= rms;
            bank.quadratic_pairs.emplace_back(i, j);
            bank.quadratic_features.push_back(std::move(q));
        }
    }
    return !bank.quadratic_features.empty();
}

bool project_field_to_linear_features(const SubspaceFeatureBank& bank,
                                      const std::vector<float>& field,
                                      std::vector<double>& coeffs) {
    const int m = bank.n_modes;
    std::vector<double> gram(static_cast<size_t>(m) * m, 0.0);
    std::vector<double> rhs(static_cast<size_t>(m), 0.0);
    for (int row = 0; row < m; ++row) {
        const auto& fr = bank.linear_features[static_cast<size_t>(row)];
        for (size_t idx = 0; idx < bank.n_cells; ++idx)
            rhs[static_cast<size_t>(row)] += fr[idx] * static_cast<double>(field[idx]);
        for (int col = 0; col < m; ++col) {
            const auto& fc = bank.linear_features[static_cast<size_t>(col)];
            double value = 0.0;
            for (size_t idx = 0; idx < bank.n_cells; ++idx)
                value += fr[idx] * fc[idx];
            gram[static_cast<size_t>(row) * m + col] = value;
        }
    }
    if (!solve_dense_system(gram, rhs, m))
        return false;
    coeffs = std::move(rhs);
    return true;
}

bool project_field_to_all_features(const SubspaceFeatureBank& bank, const std::vector<float>& field,
                                   std::vector<double>& linear_coeffs,
                                   std::vector<double>& quad_coeffs) {
    const int n_linear = bank.n_modes;
    const int n_quad = static_cast<int>(bank.quadratic_features.size());
    const int n_feat = n_linear + n_quad;
    std::vector<double> gram(static_cast<size_t>(n_feat) * n_feat, 0.0);
    std::vector<double> rhs(static_cast<size_t>(n_feat), 0.0);

    auto feature_value = [&bank, n_linear](int feat_idx, size_t cell) {
        if (feat_idx < n_linear)
            return bank.linear_features[static_cast<size_t>(feat_idx)][cell];
        return bank.quadratic_features[static_cast<size_t>(feat_idx - n_linear)][cell];
    };

    for (int row = 0; row < n_feat; ++row) {
        for (size_t idx = 0; idx < bank.n_cells; ++idx)
            rhs[static_cast<size_t>(row)] +=
                feature_value(row, idx) * static_cast<double>(field[idx]);
        for (int col = 0; col < n_feat; ++col) {
            double value = 0.0;
            for (size_t idx = 0; idx < bank.n_cells; ++idx)
                value += feature_value(row, idx) * feature_value(col, idx);
            gram[static_cast<size_t>(row) * n_feat + col] = value;
        }
        gram[static_cast<size_t>(row) * n_feat + row] += 1.0e-10;
    }

    if (!solve_dense_system(gram, rhs, n_feat))
        return false;

    linear_coeffs.assign(rhs.begin(), rhs.begin() + n_linear);
    quad_coeffs.assign(rhs.begin() + n_linear, rhs.end());
    return true;
}

void pack_subspace_coefficients(const std::vector<double>& linear1,
                                const std::vector<double>& quad1,
                                const std::vector<double>& linear2,
                                const std::vector<double>& quad2, std::vector<double>& coeffs) {
    coeffs.clear();
    coeffs.reserve(linear1.size() + quad1.size() + linear2.size() + quad2.size());
    coeffs.insert(coeffs.end(), linear1.begin(), linear1.end());
    coeffs.insert(coeffs.end(), quad1.begin(), quad1.end());
    coeffs.insert(coeffs.end(), linear2.begin(), linear2.end());
    coeffs.insert(coeffs.end(), quad2.begin(), quad2.end());
}

bool unpack_subspace_coefficients(const SubspaceFeatureBank& bank,
                                  const std::vector<double>& coeffs, std::vector<double>& linear1,
                                  std::vector<double>& quad1, std::vector<double>& linear2,
                                  std::vector<double>& quad2) {
    const size_t n_linear = static_cast<size_t>(bank.n_modes);
    const size_t n_quad = bank.quadratic_features.size();
    const size_t expected = 2 * (n_linear + n_quad);
    if (coeffs.size() != expected)
        return false;

    auto it = coeffs.begin();
    linear1.assign(it, it + static_cast<std::ptrdiff_t>(n_linear));
    it += static_cast<std::ptrdiff_t>(n_linear);
    quad1.assign(it, it + static_cast<std::ptrdiff_t>(n_quad));
    it += static_cast<std::ptrdiff_t>(n_quad);
    linear2.assign(it, it + static_cast<std::ptrdiff_t>(n_linear));
    it += static_cast<std::ptrdiff_t>(n_linear);
    quad2.assign(it, it + static_cast<std::ptrdiff_t>(n_quad));
    return true;
}

void synthesize_subspace_fields(const SubspaceFeatureBank& bank, const std::vector<double>& linear1,
                                const std::vector<double>& linear2,
                                const std::vector<double>& quad1, const std::vector<double>& quad2,
                                std::vector<float>& psi1, std::vector<float>& psi2) {
    psi1.assign(bank.n_cells, 0.0f);
    psi2.assign(bank.n_cells, 0.0f);
    for (size_t idx = 0; idx < bank.n_cells; ++idx) {
        double v1 = 0.0;
        double v2 = 0.0;
        for (int mode = 0; mode < bank.n_modes; ++mode) {
            const double feat = bank.linear_features[static_cast<size_t>(mode)][idx];
            v1 += linear1[static_cast<size_t>(mode)] * feat;
            v2 += linear2[static_cast<size_t>(mode)] * feat;
        }
        for (size_t q = 0; q < bank.quadratic_features.size(); ++q) {
            const double feat = bank.quadratic_features[q][idx];
            v1 += quad1[q] * feat;
            v2 += quad2[q] * feat;
        }
        psi1[idx] = static_cast<float>(v1);
        psi2[idx] = static_cast<float>(v2);
    }
}

bool download_velocity_cache(const Grid3D& grid, const VelocityField& vel, cudaStream_t stream,
                             HostVelocityCache& cache) {
    cache.U.resize(static_cast<size_t>(grid.nx + 1) * grid.ny * grid.nz);
    cache.V.resize(static_cast<size_t>(grid.nx) * (grid.ny + 1) * grid.nz);
    cache.W.resize(static_cast<size_t>(grid.nx) * grid.ny * (grid.nz + 1));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(cache.U.data(), vel.U.data(),
                                           cache.U.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                           stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(cache.V.data(), vel.V.data(),
                                           cache.V.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                           stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(cache.W.data(), vel.W.data(),
                                           cache.W.size() * sizeof(real), cudaMemcpyDeviceToHost,
                                           stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

inline double periodic_lift_host(double center, double neighbor, double period) {
    return neighbor + period * std::round((center - neighbor) / period);
}

inline size_t idx_host(const Grid3D& grid, int i, int j, int k) {
    return static_cast<size_t>(i) + static_cast<size_t>(grid.nx) *
                                        (static_cast<size_t>(j) + static_cast<size_t>(grid.ny) * k);
}

void host_cell_center_velocity(const Grid3D& grid, const HostVelocityCache& cache, int i, int j,
                               int k, double& vx, double& vy, double& vz) {
    auto idx_U = [&grid](int ii, int jj, int kk) {
        return static_cast<size_t>(ii) + static_cast<size_t>(grid.nx + 1) * jj +
               static_cast<size_t>(grid.nx + 1) * grid.ny * kk;
    };
    auto idx_V = [&grid](int ii, int jj, int kk) {
        return static_cast<size_t>(ii) + static_cast<size_t>(grid.nx) * jj +
               static_cast<size_t>(grid.nx) * (grid.ny + 1) * kk;
    };
    auto idx_W = [&grid](int ii, int jj, int kk) {
        return static_cast<size_t>(ii) + static_cast<size_t>(grid.nx) * jj +
               static_cast<size_t>(grid.nx) * grid.ny * kk;
    };

    vx = 0.5 * (static_cast<double>(cache.U[idx_U(i, j, k)]) +
                static_cast<double>(cache.U[idx_U(i + 1, j, k)]));
    vy = 0.5 * (static_cast<double>(cache.V[idx_V(i, j, k)]) +
                static_cast<double>(cache.V[idx_V(i, j + 1, k)]));
    vz = 0.5 * (static_cast<double>(cache.W[idx_W(i, j, k)]) +
                static_cast<double>(cache.W[idx_W(i, j, k + 1)]));
}

void host_gradient_with_self_period(const Grid3D& grid, const std::vector<float>& psi, int i, int j,
                                    int k, double L_self, double& gx, double& gy, double& gz) {
    const size_t c = idx_host(grid, i, j, k);
    const double psi_c = static_cast<double>(psi[c]);

    if (i == 0) {
        gx = (static_cast<double>(psi[idx_host(grid, 1, j, k)]) - psi_c) /
             static_cast<double>(grid.dx);
    } else if (i == grid.nx - 1) {
        gx = (psi_c - static_cast<double>(psi[idx_host(grid, grid.nx - 2, j, k)])) /
             static_cast<double>(grid.dx);
    } else {
        gx = (static_cast<double>(psi[idx_host(grid, i + 1, j, k)]) -
              static_cast<double>(psi[idx_host(grid, i - 1, j, k)])) /
             (2.0 * static_cast<double>(grid.dx));
    }

    const int jm = (j - 1 + grid.ny) % grid.ny;
    const int jp = (j + 1) % grid.ny;
    double psi_jm =
        periodic_lift_host(psi_c, static_cast<double>(psi[idx_host(grid, i, jm, k)]), L_self);
    double psi_jp =
        periodic_lift_host(psi_c, static_cast<double>(psi[idx_host(grid, i, jp, k)]), L_self);
    gy = (psi_jp - psi_jm) / (2.0 * static_cast<double>(grid.dy));

    const int km = (k - 1 + grid.nz) % grid.nz;
    const int kp = (k + 1) % grid.nz;
    double psi_km =
        periodic_lift_host(psi_c, static_cast<double>(psi[idx_host(grid, i, j, km)]), L_self);
    double psi_kp =
        periodic_lift_host(psi_c, static_cast<double>(psi[idx_host(grid, i, j, kp)]), L_self);
    gz = (psi_kp - psi_km) / (2.0 * static_cast<double>(grid.dz));
}

void build_subspace_residual_vector(const Grid3D& grid, const HostVelocityCache& vel_cache,
                                    const std::vector<float>& psi1, const std::vector<float>& psi2,
                                    double Ly, double Lz, double invariance_weight,
                                    std::vector<double>& residuals) {
    const size_t n = static_cast<size_t>(grid.nx) * grid.ny * grid.nz;
    residuals.assign(5 * n, 0.0);
    const double inv_scale = std::sqrt(std::max(invariance_weight, 0.0));

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = idx_host(grid, i, j, k);

                double vx, vy, vz;
                host_cell_center_velocity(grid, vel_cache, i, j, k, vx, vy, vz);

                double g1x, g1y, g1z, g2x, g2y, g2z;
                host_gradient_with_self_period(grid, psi1, i, j, k, Ly, g1x, g1y, g1z);
                host_gradient_with_self_period(grid, psi2, i, j, k, Lz, g2x, g2y, g2z);

                const double cx = g1y * g2z - g1z * g2y;
                const double cy = g1z * g2x - g1x * g2z;
                const double cz = g1x * g2y - g1y * g2x;

                residuals[5 * c + 0] = vx - cx;
                residuals[5 * c + 1] = vy - cy;
                residuals[5 * c + 2] = vz - cz;
                residuals[5 * c + 3] = inv_scale * (vx * g1x + vy * g1y + vz * g1z);
                residuals[5 * c + 4] = inv_scale * (vx * g2x + vy * g2y + vz * g2z);
            }
        }
    }
}

double l2_norm(const std::vector<double>& x) {
    double ssq = 0.0;
    for (double value : x)
        ssq += value * value;
    return std::sqrt(ssq);
}

ProjectionProxyReport
compute_projection_proxy_host(const Grid3D& grid, const HostVelocityCache& vel_cache,
                              const std::vector<float>& psi1, const std::vector<float>& psi2,
                              double Ly, double Lz, double cond_floor, double cond_weight) {
    ProjectionProxyReport out;
    const size_t n = static_cast<size_t>(grid.nx) * grid.ny * grid.nz;
    if (n == 0)
        return out;

    double ssq_det_mismatch = 0.0;
    double sum_abs_vx = 0.0;
    double sum_recip = 0.0;
    double min_recip = std::numeric_limits<double>::infinity();
    double low_recip = 0.0;

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                double vx, vy, vz;
                host_cell_center_velocity(grid, vel_cache, i, j, k, vx, vy, vz);

                double g1x, g1y, g1z, g2x, g2y, g2z;
                host_gradient_with_self_period(grid, psi1, i, j, k, Ly, g1x, g1y, g1z);
                host_gradient_with_self_period(grid, psi2, i, j, k, Lz, g2x, g2y, g2z);

                const double det = g1y * g2z - g1z * g2y;
                const double fro2 = g1y * g1y + g1z * g1z + g2y * g2y + g2z * g2z;
                const double recip = (fro2 > 1.0e-18) ? (2.0 * std::fabs(det) / fro2) : 0.0;

                const double det_mismatch = vx - det;
                ssq_det_mismatch += det_mismatch * det_mismatch;
                sum_abs_vx += std::fabs(vx);
                sum_recip += recip;
                min_recip = std::min(min_recip, recip);
                if (recip < cond_floor)
                    low_recip += 1.0;
            }
        }
    }

    out.rel_rms_vx_det_mismatch = std::sqrt(ssq_det_mismatch / static_cast<double>(n)) /
                                  std::max(sum_abs_vx / static_cast<double>(n), 1.0e-12);
    out.mean_recip_condition = sum_recip / static_cast<double>(n);
    out.min_recip_condition = std::isfinite(min_recip) ? min_recip : 0.0;
    out.low_recip_condition_fraction = low_recip / static_cast<double>(n);
    out.combined_score =
        out.rel_rms_vx_det_mismatch + cond_weight * out.low_recip_condition_fraction;
    out.valid = true;
    return out;
}

inline int imod_host(int n, int N) {
    return ((n % N) + N) % N;
}

inline double wrap_to_L_host(double x, double L) {
    return x - std::floor(x / L) * L;
}

inline double wrap_diff_host(double f, double L) {
    return f - std::round(f / L) * L;
}

double sample_psi_and_partials_host(const std::vector<float>& psi, double x, double y, double z,
                                    const Grid3D& grid, double Ly, double Lz, double L_self,
                                    double& dpsi_dy, double& dpsi_dz) {
    const double dx = static_cast<double>(grid.dx);
    const double dy = static_cast<double>(grid.dy);
    const double dz = static_cast<double>(grid.dz);
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;

    const double xf = x / dx - 0.5;
    int i0 = static_cast<int>(std::floor(xf));
    i0 = std::max(0, std::min(nx - 2, i0));
    const int i1 = i0 + 1;
    const double tx = std::clamp(xf - static_cast<double>(i0), 0.0, 1.0);

    const double yf = y / dy - 0.5;
    const int j0_raw = static_cast<int>(std::floor(yf));
    const int j0 = imod_host(j0_raw, ny);
    const int j1 = imod_host(j0_raw + 1, ny);
    const double ty = yf - static_cast<double>(j0_raw);

    const double zf = z / dz - 0.5;
    const int k0_raw = static_cast<int>(std::floor(zf));
    const int k0 = imod_host(k0_raw, nz);
    const int k1 = imod_host(k0_raw + 1, nz);
    const double tz = zf - static_cast<double>(k0_raw);

    double c000 = static_cast<double>(psi[idx_host(grid, i0, j0, k0)]);
    double c100 = static_cast<double>(psi[idx_host(grid, i1, j0, k0)]);
    double c010 = static_cast<double>(psi[idx_host(grid, i0, j1, k0)]);
    double c110 = static_cast<double>(psi[idx_host(grid, i1, j1, k0)]);
    double c001 = static_cast<double>(psi[idx_host(grid, i0, j0, k1)]);
    double c101 = static_cast<double>(psi[idx_host(grid, i1, j0, k1)]);
    double c011 = static_cast<double>(psi[idx_host(grid, i0, j1, k1)]);
    double c111 = static_cast<double>(psi[idx_host(grid, i1, j1, k1)]);

    const double ref = c000;
    c100 += L_self * std::round((ref - c100) / L_self);
    c010 += L_self * std::round((ref - c010) / L_self);
    c110 += L_self * std::round((ref - c110) / L_self);
    c001 += L_self * std::round((ref - c001) / L_self);
    c101 += L_self * std::round((ref - c101) / L_self);
    c011 += L_self * std::round((ref - c011) / L_self);
    c111 += L_self * std::round((ref - c111) / L_self);

    const double tx0 = 1.0 - tx, tx1 = tx;
    const double ty0 = 1.0 - ty, ty1 = ty;
    const double tz0 = 1.0 - tz, tz1 = tz;

    const double psi_val = c000 * tx0 * ty0 * tz0 + c100 * tx1 * ty0 * tz0 +
                           c010 * tx0 * ty1 * tz0 + c110 * tx1 * ty1 * tz0 +
                           c001 * tx0 * ty0 * tz1 + c101 * tx1 * ty0 * tz1 +
                           c011 * tx0 * ty1 * tz1 + c111 * tx1 * ty1 * tz1;

    dpsi_dy =
        (1.0 / dy) * (-c000 * tx0 * tz0 - c100 * tx1 * tz0 - c001 * tx0 * tz1 - c101 * tx1 * tz1 +
                      c010 * tx0 * tz0 + c110 * tx1 * tz0 + c011 * tx0 * tz1 + c111 * tx1 * tz1);

    dpsi_dz =
        (1.0 / dz) * (-c000 * tx0 * ty0 - c100 * tx1 * ty0 - c010 * tx0 * ty1 - c110 * tx1 * ty1 +
                      c001 * tx0 * ty0 + c101 * tx1 * ty0 + c011 * tx0 * ty1 + c111 * tx1 * ty1);

    (void)Ly;
    (void)Lz;
    return wrap_to_L_host(psi_val, L_self);
}

double sample_vx_host(const Grid3D& grid, const HostVelocityCache& vel_cache, double x, double y,
                      double z) {
    const double dx = static_cast<double>(grid.dx);
    const double dy = static_cast<double>(grid.dy);
    const double dz = static_cast<double>(grid.dz);
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;

    const double fx = x / dx;
    int i0 = static_cast<int>(std::floor(fx));
    i0 = (i0 < 0) ? 0 : (i0 >= nx ? nx - 1 : i0);
    const int i1 = i0 + 1;
    const double tx = std::clamp(fx - static_cast<double>(i0), 0.0, 1.0);

    const double fy = y / dy - 0.5;
    const int j0_raw = static_cast<int>(std::floor(fy));
    const int j0 = imod_host(j0_raw, ny);
    const int j1 = imod_host(j0_raw + 1, ny);
    const double ty = fy - static_cast<double>(j0_raw);

    const double fz = z / dz - 0.5;
    const int k0_raw = static_cast<int>(std::floor(fz));
    const int k0 = imod_host(k0_raw, nz);
    const int k1 = imod_host(k0_raw + 1, nz);
    const double tz = fz - static_cast<double>(k0_raw);

    auto idx_U_host = [&grid](int ii, int jj, int kk) {
        return static_cast<size_t>(ii) + static_cast<size_t>(grid.nx + 1) * jj +
               static_cast<size_t>(grid.nx + 1) * grid.ny * kk;
    };

    const double u000 = static_cast<double>(vel_cache.U[idx_U_host(i0, j0, k0)]);
    const double u100 = static_cast<double>(vel_cache.U[idx_U_host(i1, j0, k0)]);
    const double u010 = static_cast<double>(vel_cache.U[idx_U_host(i0, j1, k0)]);
    const double u110 = static_cast<double>(vel_cache.U[idx_U_host(i1, j1, k0)]);
    const double u001 = static_cast<double>(vel_cache.U[idx_U_host(i0, j0, k1)]);
    const double u101 = static_cast<double>(vel_cache.U[idx_U_host(i1, j0, k1)]);
    const double u011 = static_cast<double>(vel_cache.U[idx_U_host(i0, j1, k1)]);
    const double u111 = static_cast<double>(vel_cache.U[idx_U_host(i1, j1, k1)]);

    return u000 * (1.0 - tx) * (1.0 - ty) * (1.0 - tz) + u100 * tx * (1.0 - ty) * (1.0 - tz) +
           u010 * (1.0 - tx) * ty * (1.0 - tz) + u110 * tx * ty * (1.0 - tz) +
           u001 * (1.0 - tx) * (1.0 - ty) * tz + u101 * tx * (1.0 - ty) * tz +
           u011 * (1.0 - tx) * ty * tz + u111 * tx * ty * tz;
}

struct HostNewtonSolveReport {
    bool converged = false;
    int iterations = 0;
    double normalized_final_residual = 0.0;
    double min_recip_condition = 1.0;
};

HostNewtonSolveReport newton_solve_yz_host(const std::vector<float>& psi1,
                                           const std::vector<float>& psi2, double x, double psi1_c,
                                           double psi2_c, double y0, double z0, const Grid3D& grid,
                                           double Ly, double Lz, double& y_out, double& z_out) {
    HostNewtonSolveReport report;
    double y = y0;
    double z = z0;
    const double dy = static_cast<double>(grid.dy);
    const double dz = static_cast<double>(grid.dz);
    const double tol = PSPTA_TOL_FACTOR * std::min(dy, dz);
    const double trust_y = PSPTA_TRUST_FACTOR * dy;
    const double trust_z = PSPTA_TRUST_FACTOR * dz;

    for (int it = 0; it < PSPTA_N_NEWTON; ++it) {
        double dp1_dy = 0.0, dp1_dz = 0.0, dp2_dy = 0.0, dp2_dz = 0.0;
        const double p1 =
            sample_psi_and_partials_host(psi1, x, y, z, grid, Ly, Lz, Ly, dp1_dy, dp1_dz);
        const double p2 =
            sample_psi_and_partials_host(psi2, x, y, z, grid, Ly, Lz, Lz, dp2_dy, dp2_dz);

        const double f1 = wrap_diff_host(p1 - psi1_c, Ly);
        const double f2 = wrap_diff_host(p2 - psi2_c, Lz);
        const double res = std::max(std::fabs(f1), std::fabs(f2));
        if (res < tol) {
            report.iterations = it;
            break;
        }

        const double a = dp1_dy, b = dp1_dz;
        const double c = dp2_dy, d = dp2_dz;
        const double det = a * d - b * c;
        const double fro2 = a * a + b * b + c * c + d * d;
        const double recip = (fro2 > 1.0e-18) ? (2.0 * std::fabs(det) / fro2) : 0.0;
        report.min_recip_condition = std::min(report.min_recip_condition, recip);

        if (std::fabs(det) < PSPTA_DET_MIN || det != det) {
            report.iterations = it + 1;
            break;
        }

        const double inv_det = 1.0 / det;
        double dy_step = -(d * f1 - b * f2) * inv_det * PSPTA_DAMPING;
        double dz_step = -(-c * f1 + a * f2) * inv_det * PSPTA_DAMPING;
        dy_step = std::clamp(dy_step, -trust_y, trust_y);
        dz_step = std::clamp(dz_step, -trust_z, trust_z);
        y = wrap_to_L_host(y + dy_step, Ly);
        z = wrap_to_L_host(z + dz_step, Lz);
        report.iterations = it + 1;
    }

    double dp1_dy = 0.0, dp1_dz = 0.0, dp2_dy = 0.0, dp2_dz = 0.0;
    const double p1_fin =
        sample_psi_and_partials_host(psi1, x, y, z, grid, Ly, Lz, Ly, dp1_dy, dp1_dz);
    const double p2_fin =
        sample_psi_and_partials_host(psi2, x, y, z, grid, Ly, Lz, Lz, dp2_dy, dp2_dz);
    const double f1_fin = std::fabs(wrap_diff_host(p1_fin - psi1_c, Ly));
    const double f2_fin = std::fabs(wrap_diff_host(p2_fin - psi2_c, Lz));
    const double res_fin = std::max(f1_fin, f2_fin);

    report.converged = (res_fin < tol);
    report.normalized_final_residual = res_fin / std::max(tol, 1.0e-18);
    if (!std::isfinite(report.min_recip_condition))
        report.min_recip_condition = 1.0;

    y_out = y;
    z_out = z;
    return report;
}

double max_speed_host(const Grid3D& grid, const HostVelocityCache& vel_cache) {
    double vmax = 0.0;
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                double vx = 0.0, vy = 0.0, vz = 0.0;
                host_cell_center_velocity(grid, vel_cache, i, j, k, vx, vy, vz);
                const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);
                vmax = std::max(vmax, speed);
            }
        }
    }
    return vmax;
}

EngineSampledProxyReport
compute_engine_sampled_proxy_host(const Grid3D& grid, const HostVelocityCache& vel_cache,
                                  const std::vector<float>& psi1, const std::vector<float>& psi2,
                                  double Ly, double Lz, const RefinementACConfig& cfg) {
    EngineSampledProxyReport out;
    const int sample_count = std::max(cfg.engine_proxy_sample_count, 1);
    const int sample_steps = std::max(cfg.engine_proxy_sample_steps, 1);
    const double vmax = max_speed_host(grid, vel_cache);
    const double dt = (vmax > 0.0) ? (0.25 * static_cast<double>(grid.dx) / vmax) : 0.0;
    if (!(dt > 0.0))
        return out;

    const int ny_s =
        std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(sample_count)))));
    const int nz_s =
        std::max(1, static_cast<int>(std::ceil(static_cast<double>(sample_count) / ny_s)));
    const double x0 = 0.25 * static_cast<double>(grid.Lx());

    double total_fail = 0.0;
    double n_nonzero_fail = 0.0;
    double fail_x = 0.0;
    double fail_mid = 0.0;
    double fail_new = 0.0;
    double total_iters = 0.0;
    double total_norm_res = 0.0;
    double low_recip = 0.0;
    double attempted_newton = 0.0;
    int actual_samples = 0;

    for (int sample = 0; sample < sample_count; ++sample) {
        const int sj = sample % ny_s;
        const int sk = sample / ny_s;
        if (sk >= nz_s)
            break;
        ++actual_samples;

        double y = (static_cast<double>(sj) + 0.5) * (Ly / static_cast<double>(ny_s));
        double z = (static_cast<double>(sk) + 0.5) * (Lz / static_cast<double>(nz_s));
        double x = x0;
        double y_guess = y;
        double z_guess = z;
        uint32_t fail_count = 0;

        double dp1_dy = 0.0, dp1_dz = 0.0, dp2_dy = 0.0, dp2_dz = 0.0;
        const double psi1_c =
            sample_psi_and_partials_host(psi1, x, y, z, grid, Ly, Lz, Ly, dp1_dy, dp1_dz);
        const double psi2_c =
            sample_psi_and_partials_host(psi2, x, y, z, grid, Ly, Lz, Lz, dp2_dy, dp2_dz);

        for (int step = 0; step < sample_steps; ++step) {
            double y0 = y_guess;
            double z0 = z_guess;
            HostNewtonSolveReport rep0 = newton_solve_yz_host(
                psi1, psi2, x, psi1_c, psi2_c, y_guess, z_guess, grid, Ly, Lz, y0, z0);
            attempted_newton += 1.0;
            total_iters += rep0.iterations;
            total_norm_res += rep0.normalized_final_residual;
            if (rep0.min_recip_condition < cfg.projection_proxy_cond_floor)
                low_recip += 1.0;
            if (!rep0.converged) {
                fail_count++;
                fail_x += 1.0;
                continue;
            }

            const double vx0 = sample_vx_host(grid, vel_cache, x, y0, z0);
            if (!(vx0 > 0.0) || !std::isfinite(vx0)) {
                fail_count++;
                fail_x += 1.0;
                continue;
            }
            const double x_mid = x + 0.5 * dt * vx0;
            if (x_mid < 0.0 || x_mid >= static_cast<double>(grid.Lx())) {
                break;
            }

            double y_mid = y0;
            double z_mid = z0;
            HostNewtonSolveReport rep_mid = newton_solve_yz_host(
                psi1, psi2, x_mid, psi1_c, psi2_c, y0, z0, grid, Ly, Lz, y_mid, z_mid);
            attempted_newton += 1.0;
            total_iters += rep_mid.iterations;
            total_norm_res += rep_mid.normalized_final_residual;
            if (rep_mid.min_recip_condition < cfg.projection_proxy_cond_floor)
                low_recip += 1.0;
            if (!rep_mid.converged) {
                fail_count++;
                fail_mid += 1.0;
                continue;
            }

            const double vx_mid = sample_vx_host(grid, vel_cache, x_mid, y_mid, z_mid);
            if (!(vx_mid > 0.0) || !std::isfinite(vx_mid)) {
                fail_count++;
                fail_mid += 1.0;
                continue;
            }
            const double x_new_raw = x + dt * vx_mid;
            if (x_new_raw < 0.0 || x_new_raw >= static_cast<double>(grid.Lx())) {
                break;
            }
            const double x_new = x_new_raw;

            double y_new = y_mid;
            double z_new = z_mid;
            HostNewtonSolveReport rep_new = newton_solve_yz_host(
                psi1, psi2, x_new, psi1_c, psi2_c, y_mid, z_mid, grid, Ly, Lz, y_new, z_new);
            attempted_newton += 1.0;
            total_iters += rep_new.iterations;
            total_norm_res += rep_new.normalized_final_residual;
            if (rep_new.min_recip_condition < cfg.projection_proxy_cond_floor)
                low_recip += 1.0;
            if (!rep_new.converged) {
                fail_count++;
                fail_new += 1.0;
                continue;
            }

            x = x_new;
            y = y_new;
            z = z_new;
            y_guess = y_new;
            z_guess = z_new;
        }

        total_fail += static_cast<double>(fail_count);
        if (fail_count > 0)
            n_nonzero_fail += 1.0;
    }

    const double n_samples = std::max(actual_samples, 1);
    const double step_attempts = n_samples * sample_steps;
    out.fail_fraction = n_nonzero_fail / n_samples;
    out.mean_fail_count = total_fail / n_samples;
    out.fail_x_fraction = fail_x / step_attempts;
    out.fail_mid_fraction = fail_mid / step_attempts;
    out.fail_new_fraction = fail_new / step_attempts;
    out.mean_newton_iterations = (attempted_newton > 0.0) ? (total_iters / attempted_newton) : 0.0;
    out.mean_normalized_final_residual =
        (attempted_newton > 0.0) ? (total_norm_res / attempted_newton) : 0.0;
    out.low_recip_condition_fraction =
        (attempted_newton > 0.0) ? (low_recip / attempted_newton) : 0.0;
    out.combined_score = cfg.engine_proxy_fail_weight * out.mean_fail_count +
                         cfg.engine_proxy_iter_weight *
                             (out.mean_newton_iterations / static_cast<double>(PSPTA_N_NEWTON)) +
                         cfg.engine_proxy_residual_weight * out.mean_normalized_final_residual +
                         cfg.engine_proxy_low_recip_weight * out.low_recip_condition_fraction;
    out.valid = true;
    return out;
}

struct EngineProxySelectorValue {
    double primary = 0.0;
    double secondary = 0.0;
    double tertiary = 0.0;
};

EngineProxySelectorValue make_engine_proxy_selector(const InvariantQualityReport& quality,
                                                    const EngineSampledProxyReport& engine,
                                                    const RefinementACConfig& cfg) {
    EngineProxySelectorValue out;
    switch (cfg.engine_proxy_selector_mode) {
    case EngineProxySelectorMode::CombinedScore:
        out.primary = quality.cross_product.rel_rms_mismatch +
                      cfg.engine_proxy_acceptance_weight * engine.combined_score;
        break;
    case EngineProxySelectorMode::FailFractionLexicographic:
        out.primary = engine.fail_fraction;
        out.secondary = engine.mean_normalized_final_residual;
        out.tertiary = quality.cross_product.rel_rms_mismatch;
        break;
    }
    return out;
}

bool engine_proxy_selector_better(const EngineProxySelectorValue& lhs,
                                  const EngineProxySelectorValue& rhs) {
    if (lhs.primary + 1.0e-8 < rhs.primary)
        return true;
    if (rhs.primary + 1.0e-8 < lhs.primary)
        return false;
    if (lhs.secondary + 1.0e-10 < rhs.secondary)
        return true;
    if (rhs.secondary + 1.0e-10 < lhs.secondary)
        return false;
    return lhs.tertiary + 1.0e-8 < rhs.tertiary;
}

void build_projection_aware_residual_vector(const Grid3D& grid, const HostVelocityCache& vel_cache,
                                            const std::vector<float>& psi1,
                                            const std::vector<float>& psi2, double Ly, double Lz,
                                            double invariance_weight, double yz_weight,
                                            double cond_weight, double cond_floor,
                                            std::vector<double>& residuals,
                                            ProjectionProxyReport* proxy_stats = nullptr) {
    const size_t n = static_cast<size_t>(grid.nx) * grid.ny * grid.nz;
    residuals.assign(6 * n, 0.0);
    const double inv_scale = std::sqrt(std::max(invariance_weight, 0.0));
    const double yz_scale = std::sqrt(std::max(yz_weight, 0.0));
    const double cond_scale = std::sqrt(std::max(cond_weight, 0.0));

    double ssq_det_mismatch = 0.0;
    double sum_abs_vx = 0.0;
    double sum_recip = 0.0;
    double min_recip = std::numeric_limits<double>::infinity();
    double low_recip = 0.0;

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = idx_host(grid, i, j, k);

                double vx, vy, vz;
                host_cell_center_velocity(grid, vel_cache, i, j, k, vx, vy, vz);

                double g1x, g1y, g1z, g2x, g2y, g2z;
                host_gradient_with_self_period(grid, psi1, i, j, k, Ly, g1x, g1y, g1z);
                host_gradient_with_self_period(grid, psi2, i, j, k, Lz, g2x, g2y, g2z);

                const double cx = g1y * g2z - g1z * g2y;
                const double cy = g1z * g2x - g1x * g2z;
                const double cz = g1x * g2y - g1y * g2x;
                const double inv1 = vx * g1x + vy * g1y + vz * g1z;
                const double inv2 = vx * g2x + vy * g2y + vz * g2z;

                const double fro2 = g1y * g1y + g1z * g1z + g2y * g2y + g2z * g2z;
                const double recip = (fro2 > 1.0e-18) ? (2.0 * std::fabs(cx) / fro2) : 0.0;
                const double cond_barrier = std::max(0.0, cond_floor - recip);

                residuals[6 * c + 0] = vx - cx;
                residuals[6 * c + 1] = yz_scale * (vy - cy);
                residuals[6 * c + 2] = yz_scale * (vz - cz);
                residuals[6 * c + 3] = inv_scale * inv1;
                residuals[6 * c + 4] = inv_scale * inv2;
                residuals[6 * c + 5] = cond_scale * cond_barrier;

                ssq_det_mismatch += (vx - cx) * (vx - cx);
                sum_abs_vx += std::fabs(vx);
                sum_recip += recip;
                min_recip = std::min(min_recip, recip);
                if (recip < cond_floor)
                    low_recip += 1.0;
            }
        }
    }

    if (proxy_stats) {
        proxy_stats->rel_rms_vx_det_mismatch =
            std::sqrt(ssq_det_mismatch / static_cast<double>(n)) /
            std::max(sum_abs_vx / static_cast<double>(n), 1.0e-12);
        proxy_stats->mean_recip_condition = sum_recip / static_cast<double>(n);
        proxy_stats->min_recip_condition = std::isfinite(min_recip) ? min_recip : 0.0;
        proxy_stats->low_recip_condition_fraction = low_recip / static_cast<double>(n);
        proxy_stats->combined_score = proxy_stats->rel_rms_vx_det_mismatch +
                                      cond_weight * proxy_stats->low_recip_condition_fraction;
        proxy_stats->valid = true;
    }
}

} // namespace

RefinementAC::RefinementAC(const Grid3D& grid, const VelocityField* vel,
                           const RefinementACConfig& config)
    : grid_(grid), vel_(vel), config_(config) {
    GaugeFixerConfig gf_cfg;
    gf_cfg.method = GaugeMethod::None;
    gauge_fixer_ = std::make_unique<GaugeFixer>(gf_cfg);
}

void RefinementAC::set_gauge_fixer(std::unique_ptr<GaugeFixer> gf) {
    gauge_fixer_ = std::move(gf);
}

void RefinementAC::set_subspace_basis_host(std::vector<std::vector<float>> basis_modes) {
    subspace_basis_host_ = std::move(basis_modes);
}

RefinementACReport RefinementAC::refine(PsptaInvariantField& inv, CudaContext& ctx) {
    switch (config_.strategy) {
    case RefinementACStrategy::AlternatingProjection:
        return refine_alternating_projection(inv, ctx);
    case RefinementACStrategy::SubspaceQuadraticMap:
        return refine_subspace_quadratic_map(inv, ctx);
    case RefinementACStrategy::SubspaceQuadraticGaussNewton:
        return refine_subspace_quadratic_gauss_newton(inv, ctx);
    case RefinementACStrategy::SubspaceQuadraticGaussNewtonProjectionProxy:
        return refine_subspace_quadratic_gauss_newton_projection_proxy(inv, ctx);
    case RefinementACStrategy::SubspaceQuadraticGaussNewtonEngineProxy:
        return refine_subspace_quadratic_gauss_newton_engine_proxy(inv, ctx);
    }

    RefinementACReport report;
    report.enabled = config_.enabled;
    report.stop_reason = "invalid_strategy";
    return report;
}

RefinementACReport RefinementAC::refine_alternating_projection(PsptaInvariantField& inv,
                                                               CudaContext& ctx) {
    RefinementACReport report;
    report.enabled = config_.enabled;

    if (!config_.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (!inv.is_valid() || vel_ == nullptr) {
        report.stop_reason = "invalid_input";
        return report;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = inv.num_cells();
    d_delta_psi1_.resize(n);
    d_delta_psi2_.resize(n);
    d_rhs_.resize(n);
    d_target_gx_.resize(n);
    d_target_gy_.resize(n);
    d_target_gz_.resize(n);
    d_trial_psi1_.resize(n);
    d_trial_psi2_.resize(n);
    d_base_psi1_.resize(n);
    d_base_psi2_.resize(n);

    report.initial_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    FieldStats initial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.initial_min_gradient_rms = initial_stats.min_gradient_rms;
    report.initial_min_field_range = initial_stats.min_field_range;
    InvariantQualityReport current_quality = report.initial_quality;

    LaplacianOperator3D lap(grid_);
    PinnedLaplacianCGOperator pinned_lap{&lap};
    solvers::CGWorkspace cg_ws;
    solvers::CGConfig cg_cfg;
    cg_cfg.max_iter = config_.poisson_max_iter;
    cg_cfg.rtol = static_cast<real>(config_.poisson_tol);
    cg_cfg.atol = 0.0;
    cg_cfg.check_every = 5;
    cg_cfg.verbose = false;

    auto copy_current_to_base = [&]() {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_base_psi1_.data(), inv.psi1_ptr(),
                                               n * sizeof(float), cudaMemcpyDeviceToDevice,
                                               ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_base_psi2_.data(), inv.psi2_ptr(),
                                               n * sizeof(float), cudaMemcpyDeviceToDevice,
                                               ctx.cuda_stream()));
    };

    auto restore_base_into_inv = [&]() {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi1_ptr(), d_base_psi1_.data(),
                                               n * sizeof(float), cudaMemcpyDeviceToDevice,
                                               ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi2_ptr(), d_base_psi2_.data(),
                                               n * sizeof(float), cudaMemcpyDeviceToDevice,
                                               ctx.cuda_stream()));
    };

    auto prepare_rhs = [&](DeviceBuffer<real>& rhs) {
        std::vector<real> h_rhs(n, real(0));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_rhs.data(), rhs.data(), n * sizeof(real),
                                               cudaMemcpyDeviceToHost, ctx.cuda_stream()));
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
        double mean = 0.0;
        for (real value : h_rhs)
            mean += static_cast<double>(value);
        mean /= std::max<double>(n, 1.0);
        for (real& value : h_rhs)
            value = static_cast<real>(static_cast<double>(value) - mean);
        h_rhs[0] = real(0);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(rhs.data(), h_rhs.data(), n * sizeof(real),
                                               cudaMemcpyHostToDevice, ctx.cuda_stream()));
    };

    auto try_half_step = [&](bool update_psi1, RefinementIterReport& iter_report) -> bool {
        copy_current_to_base();
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));

        const int block = 256;
        const int grid_k = (static_cast<int>(n) + block - 1) / block;
        if (update_psi1) {
            kernel_compute_delta_gradient_psi1<<<grid_k, block, 0, ctx.cuda_stream()>>>(
                inv.psi1_ptr(), inv.psi2_ptr(), vel_->U.data(), vel_->V.data(), vel_->W.data(),
                d_target_gx_.data(), d_target_gy_.data(), d_target_gz_.data(), grid_.nx, grid_.ny,
                grid_.nz, static_cast<double>(grid_.dx), static_cast<double>(grid_.dy),
                static_cast<double>(grid_.dz), inv.Ly(), inv.Lz(), config_.invariance_weight,
                config_.local_tikhonov);
        } else {
            kernel_compute_delta_gradient_psi2<<<grid_k, block, 0, ctx.cuda_stream()>>>(
                inv.psi1_ptr(), inv.psi2_ptr(), vel_->U.data(), vel_->V.data(), vel_->W.data(),
                d_target_gx_.data(), d_target_gy_.data(), d_target_gz_.data(), grid_.nx, grid_.ny,
                grid_.nz, static_cast<double>(grid_.dx), static_cast<double>(grid_.dy),
                static_cast<double>(grid_.dz), inv.Ly(), inv.Lz(), config_.invariance_weight,
                config_.local_tikhonov);
        }
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

        kernel_compute_divergence_rhs<<<grid_k, block, 0, ctx.cuda_stream()>>>(
            d_target_gx_.data(), d_target_gy_.data(), d_target_gz_.data(), d_rhs_.data(), grid_.nx,
            grid_.ny, grid_.nz, static_cast<double>(grid_.dx), static_cast<double>(grid_.dy),
            static_cast<double>(grid_.dz));
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        prepare_rhs(d_rhs_);
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));

        DeviceBuffer<real>& delta = update_psi1 ? d_delta_psi1_ : d_delta_psi2_;
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(delta.data(), 0, n * sizeof(real), ctx.cuda_stream()));
        solvers::CGResult cg_result = solvers::cg_solve(
            ctx, pinned_lap, DeviceSpan<const real>(d_rhs_.span()), delta.span(), cg_cfg, cg_ws);
        if (update_psi1)
            iter_report.poisson_residual_1 = cg_result.r_norm;
        else
            iter_report.poisson_residual_2 = cg_result.r_norm;

        double omega = config_.omega;
        bool have_best_trial = false;
        for (int backtrack = 0; backtrack < config_.max_backtracks && omega >= config_.omega_min;
             ++backtrack) {
            if (update_psi1) {
                kernel_apply_delta_to_field<<<grid_k, block, 0, ctx.cuda_stream()>>>(
                    d_base_psi1_.data(), delta.data(), d_trial_psi1_.data(), static_cast<int>(n),
                    omega);
                MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi1_ptr(), d_trial_psi1_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       ctx.cuda_stream()));
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi2_ptr(), d_base_psi2_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       ctx.cuda_stream()));
            } else {
                kernel_apply_delta_to_field<<<grid_k, block, 0, ctx.cuda_stream()>>>(
                    d_base_psi2_.data(), delta.data(), d_trial_psi2_.data(), static_cast<int>(n),
                    omega);
                MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi1_ptr(), d_base_psi1_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       ctx.cuda_stream()));
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(inv.psi2_ptr(), d_trial_psi2_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       ctx.cuda_stream()));
            }
            MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));

            const InvariantQualityReport trial_quality =
                inv.compute_quality(*vel_, ctx.cuda_stream());
            const FieldStats trial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
            const std::string trial_rejection =
                rejection_reason(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_);

            if (!have_best_trial || trial_quality.cross_product.rel_rms_mismatch <
                                        iter_report.best_trial_rel_mismatch) {
                have_best_trial = true;
                iter_report.best_trial_phase = update_psi1 ? "psi1" : "psi2";
                iter_report.best_trial_rejection_reason = trial_rejection;
                iter_report.best_trial_rel_mismatch = trial_quality.cross_product.rel_rms_mismatch;
                iter_report.best_trial_invariance_sum =
                    trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
                iter_report.best_trial_degeneracy = trial_quality.independence.degeneracy_score;
                iter_report.best_trial_min_gradient_rms = trial_stats.min_gradient_rms;
                iter_report.best_trial_min_field_range = trial_stats.min_field_range;
            }

            if (trial_admissible(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_)) {
                current_quality = trial_quality;
                iter_report.omega_accepted = omega;
                iter_report.backtracks = backtrack;
                iter_report.accepted = true;
                return true;
            }

            restore_base_into_inv();
            MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
            omega *= 0.5;
        }

        restore_base_into_inv();
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(ctx.cuda_stream()));
        return false;
    };

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        RefinementIterReport iter_report;
        iter_report.iter = iter + 1;
        iter_report.quality_before = current_quality;
        const double before_rel = current_quality.cross_product.rel_rms_mismatch;

        const bool accepted_psi1 = try_half_step(true, iter_report);
        const bool accepted_psi2 = try_half_step(false, iter_report);

        if (gauge_fixer_ && gauge_fixer_->method() != GaugeMethod::None) {
            gauge_fixer_->apply(inv, *vel_, ctx.cuda_stream());
        }

        iter_report.quality_after = inv.compute_quality(*vel_, ctx.cuda_stream());
        current_quality = iter_report.quality_after;
        const double after_rel = current_quality.cross_product.rel_rms_mismatch;
        iter_report.rel_improvement =
            (before_rel > 1.0e-12) ? (before_rel - after_rel) / before_rel : 0.0;

        report.history.push_back(iter_report);
        report.iterations_done = iter + 1;

        if (config_.verbose) {
            std::printf("  [StrategyC] iter=%d accept=(%d,%d) omega=%.3e rel=%.6e -> %.6e "
                        "inv_sum=%.3e deg=%.3f\n",
                        iter + 1, accepted_psi1 ? 1 : 0, accepted_psi2 ? 1 : 0,
                        iter_report.omega_accepted, before_rel, after_rel,
                        current_quality.invariance.rms_r1 + current_quality.invariance.rms_r2,
                        current_quality.independence.degeneracy_score);
        }

        if (!accepted_psi1 && !accepted_psi2) {
            report.stop_reason = "no_accepted_update";
            break;
        }
        if (current_quality.cross_product.rel_rms_mismatch <= config_.stop_abs_quality) {
            report.converged = true;
            report.stop_reason = "abs_quality";
            break;
        }
        if (iter_report.rel_improvement < config_.stop_rel_quality) {
            report.converged = true;
            report.stop_reason = "rel_improvement";
            break;
        }
    }

    if (report.stop_reason.empty())
        report.stop_reason = report.converged ? "converged" : "max_iterations";

    report.final_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats final_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.final_min_gradient_rms = final_stats.min_gradient_rms;
    report.final_min_field_range = final_stats.min_field_range;

    auto info = inv.construction_info();
    info.method = InvariantConstructionMethod::StrategyAC;
    info.refinement_iterations = report.iterations_done;
    info.refinement_omega = config_.omega;
    info.refinement_final_rms = report.final_quality.cross_product.rel_rms_mismatch;
    info.refinement_stop_reason = report.stop_reason;
    info.gauge_method =
        gauge_fixer_ ? (gauge_fixer_->method() == GaugeMethod::None ? "none" : "custom") : "none";
    inv.set_construction_info(info);

    const auto t1 = std::chrono::steady_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return report;
}

RefinementACReport RefinementAC::refine_subspace_quadratic_map(PsptaInvariantField& inv,
                                                               CudaContext& ctx) {
    RefinementACReport report;
    report.enabled = config_.enabled;

    if (!config_.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (!inv.is_valid() || vel_ == nullptr) {
        report.stop_reason = "invalid_input";
        return report;
    }
    if (subspace_basis_host_.size() < 2) {
        report.stop_reason = "invalid_subspace";
        return report;
    }
    if (subspace_basis_host_.size() > 4) {
        subspace_basis_host_.resize(4);
    }
    for (const auto& mode : subspace_basis_host_) {
        if (mode.size() != inv.num_cells()) {
            report.stop_reason = "invalid_subspace";
            return report;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = inv.num_cells();
    report.initial_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats initial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.initial_min_gradient_rms = initial_stats.min_gradient_rms;
    report.initial_min_field_range = initial_stats.min_field_range;

    SubspaceFeatureBank bank;
    if (!build_subspace_feature_bank(subspace_basis_host_, bank)) {
        report.stop_reason = "invalid_subspace";
        return report;
    }

    const std::vector<float> initial_psi1 = download_field(inv.psi1_ptr(), n, ctx.cuda_stream());
    const std::vector<float> initial_psi2 = download_field(inv.psi2_ptr(), n, ctx.cuda_stream());

    std::vector<double> linear1;
    std::vector<double> linear2;
    if (!project_field_to_linear_features(bank, initial_psi1, linear1) ||
        !project_field_to_linear_features(bank, initial_psi2, linear2)) {
        report.stop_reason = "projection_failed";
        return report;
    }

    std::vector<double> quad1(bank.quadratic_features.size(), 0.0);
    std::vector<double> quad2(bank.quadratic_features.size(), 0.0);
    std::vector<float> psi1_host;
    std::vector<float> psi2_host;
    synthesize_subspace_fields(bank, linear1, linear2, quad1, quad2, psi1_host, psi2_host);
    upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());

    InvariantQualityReport current_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    report.initial_quality = current_quality;

    double step = config_.subspace_initial_step;
    if (step <= 0.0)
        step = std::max(config_.omega, 0.1);

    auto apply_best_state = [&](const std::vector<float>& psi1, const std::vector<float>& psi2) {
        upload_field_pair(psi1, psi2, inv, ctx.cuda_stream());
    };

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        RefinementIterReport iter_report;
        iter_report.iter = iter + 1;
        iter_report.quality_before = current_quality;
        const double before_rel = current_quality.cross_product.rel_rms_mismatch;

        bool accepted = false;
        InvariantQualityReport accepted_quality;
        FieldStats accepted_stats;
        std::vector<double> accepted_quad1 = quad1;
        std::vector<double> accepted_quad2 = quad2;
        std::vector<float> accepted_psi1 = psi1_host;
        std::vector<float> accepted_psi2 = psi2_host;
        double accepted_step = 0.0;
        int accepted_feature = -1;
        int accepted_field = -1;
        int accepted_sign = 0;

        bool have_best_trial = false;

        auto consider_trial = [&](int which_field, size_t q_idx, int sign) {
            std::vector<double> trial_quad1 = quad1;
            std::vector<double> trial_quad2 = quad2;
            if (which_field == 0) {
                trial_quad1[q_idx] += static_cast<double>(sign) * step;
            } else {
                trial_quad2[q_idx] += static_cast<double>(sign) * step;
            }

            std::vector<float> trial_psi1;
            std::vector<float> trial_psi2;
            synthesize_subspace_fields(bank, linear1, linear2, trial_quad1, trial_quad2, trial_psi1,
                                       trial_psi2);
            upload_field_pair(trial_psi1, trial_psi2, inv, ctx.cuda_stream());

            const InvariantQualityReport trial_quality =
                inv.compute_quality(*vel_, ctx.cuda_stream());
            const FieldStats trial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
            const std::string trial_rejection =
                rejection_reason(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_);

            if (!have_best_trial || trial_quality.cross_product.rel_rms_mismatch <
                                        iter_report.best_trial_rel_mismatch) {
                have_best_trial = true;
                iter_report.best_trial_phase = std::string(which_field == 0 ? "q1[" : "q2[") +
                                               std::to_string(q_idx) + (sign > 0 ? "]+" : "]-");
                iter_report.best_trial_rejection_reason = trial_rejection;
                iter_report.best_trial_rel_mismatch = trial_quality.cross_product.rel_rms_mismatch;
                iter_report.best_trial_invariance_sum =
                    trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
                iter_report.best_trial_degeneracy = trial_quality.independence.degeneracy_score;
                iter_report.best_trial_min_gradient_rms = trial_stats.min_gradient_rms;
                iter_report.best_trial_min_field_range = trial_stats.min_field_range;
            }

            if (trial_admissible(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_)) {
                if (!accepted || trial_quality.cross_product.rel_rms_mismatch <
                                     accepted_quality.cross_product.rel_rms_mismatch) {
                    accepted = true;
                    accepted_quality = trial_quality;
                    accepted_stats = trial_stats;
                    accepted_quad1 = std::move(trial_quad1);
                    accepted_quad2 = std::move(trial_quad2);
                    accepted_psi1 = std::move(trial_psi1);
                    accepted_psi2 = std::move(trial_psi2);
                    accepted_step = step;
                    accepted_feature = static_cast<int>(q_idx);
                    accepted_field = which_field;
                    accepted_sign = sign;
                }
            }
        };

        for (size_t q_idx = 0; q_idx < bank.quadratic_features.size(); ++q_idx) {
            consider_trial(0, q_idx, +1);
            consider_trial(0, q_idx, -1);
            consider_trial(1, q_idx, +1);
            consider_trial(1, q_idx, -1);
        }

        if (accepted) {
            quad1 = std::move(accepted_quad1);
            quad2 = std::move(accepted_quad2);
            psi1_host = std::move(accepted_psi1);
            psi2_host = std::move(accepted_psi2);
            apply_best_state(psi1_host, psi2_host);
            current_quality = accepted_quality;
            iter_report.accepted = true;
            iter_report.omega_accepted = accepted_step;
            iter_report.backtracks = 0;
            iter_report.best_trial_phase = std::string(accepted_field == 0 ? "q1[" : "q2[") +
                                           std::to_string(accepted_feature) +
                                           (accepted_sign > 0 ? "]+" : "]-");
        } else {
            apply_best_state(psi1_host, psi2_host);
        }

        iter_report.quality_after = inv.compute_quality(*vel_, ctx.cuda_stream());
        current_quality = iter_report.quality_after;
        const double after_rel = current_quality.cross_product.rel_rms_mismatch;
        iter_report.rel_improvement =
            (before_rel > 1.0e-12) ? (before_rel - after_rel) / before_rel : 0.0;

        report.history.push_back(iter_report);
        report.iterations_done = iter + 1;

        if (config_.verbose) {
            std::printf("  [StrategyC-Subspace] iter=%d accepted=%d step=%.3e rel=%.6e -> %.6e "
                        "inv_sum=%.3e deg=%.3f\n",
                        iter + 1, iter_report.accepted ? 1 : 0, step, before_rel, after_rel,
                        current_quality.invariance.rms_r1 + current_quality.invariance.rms_r2,
                        current_quality.independence.degeneracy_score);
        }

        if (!accepted) {
            step *= 0.5;
            if (step < config_.subspace_min_step) {
                report.stop_reason = "subspace_step_exhausted";
                break;
            }
        }
        if (current_quality.cross_product.rel_rms_mismatch <= config_.stop_abs_quality) {
            report.converged = true;
            report.stop_reason = "abs_quality";
            break;
        }
        if (iter_report.rel_improvement < config_.stop_rel_quality && accepted) {
            report.converged = true;
            report.stop_reason = "rel_improvement";
            break;
        }
    }

    if (report.stop_reason.empty())
        report.stop_reason = report.converged ? "converged" : "max_iterations";

    report.final_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats final_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.final_min_gradient_rms = final_stats.min_gradient_rms;
    report.final_min_field_range = final_stats.min_field_range;

    auto info = inv.construction_info();
    info.method = InvariantConstructionMethod::StrategyAC;
    info.refinement_iterations = report.iterations_done;
    info.refinement_omega = config_.subspace_initial_step;
    info.refinement_final_rms = report.final_quality.cross_product.rel_rms_mismatch;
    info.refinement_stop_reason = report.stop_reason;
    info.gauge_method = "subspace_quadratic_map";
    inv.set_construction_info(info);

    const auto t1 = std::chrono::steady_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return report;
}

RefinementACReport RefinementAC::refine_subspace_quadratic_gauss_newton(PsptaInvariantField& inv,
                                                                        CudaContext& ctx) {
    RefinementACReport report;
    report.enabled = config_.enabled;

    if (!config_.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (!inv.is_valid() || vel_ == nullptr) {
        report.stop_reason = "invalid_input";
        return report;
    }
    if (subspace_basis_host_.size() < 2) {
        report.stop_reason = "invalid_subspace";
        return report;
    }
    if (subspace_basis_host_.size() > 4) {
        subspace_basis_host_.resize(4);
    }
    for (const auto& mode : subspace_basis_host_) {
        if (mode.size() != inv.num_cells()) {
            report.stop_reason = "invalid_subspace";
            return report;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = inv.num_cells();
    report.initial_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats initial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.initial_min_gradient_rms = initial_stats.min_gradient_rms;
    report.initial_min_field_range = initial_stats.min_field_range;

    SubspaceFeatureBank bank;
    if (!build_subspace_feature_bank(subspace_basis_host_, bank)) {
        report.stop_reason = "invalid_subspace";
        return report;
    }

    HostVelocityCache vel_cache;
    if (!download_velocity_cache(grid_, *vel_, ctx.cuda_stream(), vel_cache)) {
        report.stop_reason = "velocity_download_failed";
        return report;
    }

    const std::vector<float> initial_psi1 = download_field(inv.psi1_ptr(), n, ctx.cuda_stream());
    const std::vector<float> initial_psi2 = download_field(inv.psi2_ptr(), n, ctx.cuda_stream());

    std::vector<double> linear1, quad1, linear2, quad2;
    if (!project_field_to_all_features(bank, initial_psi1, linear1, quad1) ||
        !project_field_to_all_features(bank, initial_psi2, linear2, quad2)) {
        report.stop_reason = "projection_failed";
        return report;
    }

    std::vector<double> coeffs;
    pack_subspace_coefficients(linear1, quad1, linear2, quad2, coeffs);

    std::vector<float> psi1_host;
    std::vector<float> psi2_host;
    synthesize_subspace_fields(bank, linear1, linear2, quad1, quad2, psi1_host, psi2_host);
    upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());

    InvariantQualityReport current_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    report.initial_quality = current_quality;

    double lambda = std::max(config_.gn_lambda_initial, 1.0e-8);
    double trust_radius = std::max(config_.gn_trust_radius_initial, config_.gn_trust_radius_min);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        RefinementIterReport iter_report;
        iter_report.iter = iter + 1;
        iter_report.quality_before = current_quality;
        const double before_rel = current_quality.cross_product.rel_rms_mismatch;

        std::vector<double> residual0;
        build_subspace_residual_vector(grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(),
                                       config_.invariance_weight, residual0);
        const size_t residual_dim = residual0.size();
        const int n_coeff = static_cast<int>(coeffs.size());
        std::vector<double> jacobian(residual_dim * static_cast<size_t>(n_coeff), 0.0);
        std::vector<double> fd_steps(static_cast<size_t>(n_coeff), 0.0);

        std::vector<double> trial_linear1, trial_quad1, trial_linear2, trial_quad2;
        std::vector<float> trial_psi1;
        std::vector<float> trial_psi2;

        for (int ci = 0; ci < n_coeff; ++ci) {
            std::vector<double> perturbed = coeffs;
            const double step = config_.gn_fd_absolute_step +
                                config_.gn_fd_relative_step * std::max(1.0, std::fabs(coeffs[ci]));
            fd_steps[static_cast<size_t>(ci)] = step;
            perturbed[ci] += step;
            if (!unpack_subspace_coefficients(bank, perturbed, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                report.stop_reason = "projection_failed";
                return report;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            std::vector<double> residual_fd;
            build_subspace_residual_vector(grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(),
                                           inv.Lz(), config_.invariance_weight, residual_fd);
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(ci)] =
                    (residual_fd[ri] - residual0[ri]) / step;
            }
        }

        std::vector<double> jtj(static_cast<size_t>(n_coeff) * n_coeff, 0.0);
        std::vector<double> jtr(static_cast<size_t>(n_coeff), 0.0);
        for (int row = 0; row < n_coeff; ++row) {
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                const double Jr =
                    jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)];
                jtr[static_cast<size_t>(row)] += Jr * residual0[ri];
            }
            for (int col = 0; col < n_coeff; ++col) {
                double value = 0.0;
                for (size_t ri = 0; ri < residual_dim; ++ri) {
                    value +=
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)] *
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(col)];
                }
                jtj[static_cast<size_t>(row) * n_coeff + col] = value;
            }
        }

        std::vector<double> system = jtj;
        std::vector<double> rhs(static_cast<size_t>(n_coeff), 0.0);
        for (int i = 0; i < n_coeff; ++i) {
            const double diag = jtj[static_cast<size_t>(i) * n_coeff + i];
            system[static_cast<size_t>(i) * n_coeff + i] += lambda * (diag + 1.0);
            rhs[static_cast<size_t>(i)] = -jtr[static_cast<size_t>(i)];
        }
        if (!solve_dense_system(system, rhs, n_coeff)) {
            trust_radius *= 0.5;
            lambda *= config_.gn_lambda_up;
            iter_report.best_trial_phase = "gn";
            iter_report.best_trial_rejection_reason = "linear_solve_failed";
            iter_report.best_trial_rel_mismatch = before_rel;
            iter_report.quality_after = current_quality;
            report.history.push_back(iter_report);
            report.iterations_done = iter + 1;
            if (trust_radius < config_.gn_trust_radius_min) {
                report.stop_reason = "trust_region_exhausted";
                break;
            }
            continue;
        }

        double delta_norm = l2_norm(rhs);
        if (delta_norm > trust_radius && delta_norm > 0.0) {
            const double scale = trust_radius / delta_norm;
            for (double& value : rhs)
                value *= scale;
            delta_norm = trust_radius;
        }

        bool accepted = false;
        InvariantQualityReport accepted_quality;
        std::vector<double> accepted_coeffs = coeffs;
        std::vector<float> accepted_psi1 = psi1_host;
        std::vector<float> accepted_psi2 = psi2_host;
        double accepted_scale = 0.0;

        bool have_best_trial = false;
        for (int backtrack = 0; backtrack < config_.max_backtracks; ++backtrack) {
            const double scale = std::ldexp(1.0, -backtrack);
            std::vector<double> trial_coeffs = coeffs;
            for (int i = 0; i < n_coeff; ++i)
                trial_coeffs[static_cast<size_t>(i)] += scale * rhs[static_cast<size_t>(i)];

            if (!unpack_subspace_coefficients(bank, trial_coeffs, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                continue;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            upload_field_pair(trial_psi1, trial_psi2, inv, ctx.cuda_stream());

            const InvariantQualityReport trial_quality =
                inv.compute_quality(*vel_, ctx.cuda_stream());
            const FieldStats trial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
            const std::string trial_rejection =
                rejection_reason(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_);

            if (!have_best_trial || trial_quality.cross_product.rel_rms_mismatch <
                                        iter_report.best_trial_rel_mismatch) {
                have_best_trial = true;
                iter_report.best_trial_phase = "gn(scale=" + std::to_string(scale) + ")";
                iter_report.best_trial_rejection_reason = trial_rejection;
                iter_report.best_trial_rel_mismatch = trial_quality.cross_product.rel_rms_mismatch;
                iter_report.best_trial_invariance_sum =
                    trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
                iter_report.best_trial_degeneracy = trial_quality.independence.degeneracy_score;
                iter_report.best_trial_min_gradient_rms = trial_stats.min_gradient_rms;
                iter_report.best_trial_min_field_range = trial_stats.min_field_range;
            }

            if (trial_admissible(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_)) {
                accepted = true;
                accepted_quality = trial_quality;
                accepted_coeffs = std::move(trial_coeffs);
                accepted_psi1 = std::move(trial_psi1);
                accepted_psi2 = std::move(trial_psi2);
                accepted_scale = scale;
                break;
            }
        }

        if (accepted) {
            coeffs = std::move(accepted_coeffs);
            psi1_host = std::move(accepted_psi1);
            psi2_host = std::move(accepted_psi2);
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            current_quality = accepted_quality;
            iter_report.accepted = true;
            iter_report.omega_accepted = accepted_scale;
            iter_report.backtracks = 0;
            lambda = std::max(lambda * config_.gn_lambda_down, 1.0e-8);
            trust_radius =
                std::min(std::max(trust_radius, delta_norm) * 1.25, config_.gn_trust_radius_max);
        } else {
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            lambda *= config_.gn_lambda_up;
            trust_radius *= 0.5;
        }

        iter_report.quality_after = inv.compute_quality(*vel_, ctx.cuda_stream());
        current_quality = iter_report.quality_after;
        const double after_rel = current_quality.cross_product.rel_rms_mismatch;
        iter_report.rel_improvement =
            (before_rel > 1.0e-12) ? (before_rel - after_rel) / before_rel : 0.0;

        report.history.push_back(iter_report);
        report.iterations_done = iter + 1;

        if (config_.verbose) {
            std::printf("  [StrategyC-SubspaceGN] iter=%d accepted=%d lambda=%.3e trust=%.3e "
                        "rel=%.6e -> %.6e inv_sum=%.3e deg=%.3f\n",
                        iter + 1, iter_report.accepted ? 1 : 0, lambda, trust_radius, before_rel,
                        after_rel,
                        current_quality.invariance.rms_r1 + current_quality.invariance.rms_r2,
                        current_quality.independence.degeneracy_score);
        }

        if (!accepted && trust_radius < config_.gn_trust_radius_min) {
            report.stop_reason = "trust_region_exhausted";
            break;
        }
        if (current_quality.cross_product.rel_rms_mismatch <= config_.stop_abs_quality) {
            report.converged = true;
            report.stop_reason = "abs_quality";
            break;
        }
        if (iter_report.rel_improvement < config_.stop_rel_quality && accepted) {
            report.converged = true;
            report.stop_reason = "rel_improvement";
            break;
        }
    }

    if (report.stop_reason.empty())
        report.stop_reason = report.converged ? "converged" : "max_iterations";

    report.final_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats final_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.final_min_gradient_rms = final_stats.min_gradient_rms;
    report.final_min_field_range = final_stats.min_field_range;

    auto info = inv.construction_info();
    info.method = InvariantConstructionMethod::StrategyAC;
    info.refinement_iterations = report.iterations_done;
    info.refinement_omega = config_.gn_trust_radius_initial;
    info.refinement_final_rms = report.final_quality.cross_product.rel_rms_mismatch;
    info.refinement_stop_reason = report.stop_reason;
    info.gauge_method = "subspace_quadratic_gn";
    inv.set_construction_info(info);

    const auto t1 = std::chrono::steady_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return report;
}

RefinementACReport
RefinementAC::refine_subspace_quadratic_gauss_newton_projection_proxy(PsptaInvariantField& inv,
                                                                      CudaContext& ctx) {
    RefinementACReport report;
    report.enabled = config_.enabled;

    if (!config_.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (!inv.is_valid() || vel_ == nullptr) {
        report.stop_reason = "invalid_input";
        return report;
    }
    if (subspace_basis_host_.size() < 2) {
        report.stop_reason = "invalid_subspace";
        return report;
    }
    if (subspace_basis_host_.size() > 4)
        subspace_basis_host_.resize(4);
    for (const auto& mode : subspace_basis_host_) {
        if (mode.size() != inv.num_cells()) {
            report.stop_reason = "invalid_subspace";
            return report;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = inv.num_cells();
    report.initial_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats initial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.initial_min_gradient_rms = initial_stats.min_gradient_rms;
    report.initial_min_field_range = initial_stats.min_field_range;

    SubspaceFeatureBank bank;
    if (!build_subspace_feature_bank(subspace_basis_host_, bank)) {
        report.stop_reason = "invalid_subspace";
        return report;
    }

    HostVelocityCache vel_cache;
    if (!download_velocity_cache(grid_, *vel_, ctx.cuda_stream(), vel_cache)) {
        report.stop_reason = "velocity_download_failed";
        return report;
    }

    const std::vector<float> initial_psi1 = download_field(inv.psi1_ptr(), n, ctx.cuda_stream());
    const std::vector<float> initial_psi2 = download_field(inv.psi2_ptr(), n, ctx.cuda_stream());

    std::vector<double> linear1, quad1, linear2, quad2;
    if (!project_field_to_all_features(bank, initial_psi1, linear1, quad1) ||
        !project_field_to_all_features(bank, initial_psi2, linear2, quad2)) {
        report.stop_reason = "projection_failed";
        return report;
    }

    std::vector<double> coeffs;
    pack_subspace_coefficients(linear1, quad1, linear2, quad2, coeffs);

    std::vector<float> psi1_host;
    std::vector<float> psi2_host;
    synthesize_subspace_fields(bank, linear1, linear2, quad1, quad2, psi1_host, psi2_host);
    upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());

    InvariantQualityReport current_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    report.initial_quality = current_quality;
    ProjectionProxyReport current_proxy = compute_projection_proxy_host(
        grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(),
        config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);
    report.initial_projection = current_proxy;

    double lambda = std::max(config_.gn_lambda_initial, 1.0e-8);
    double trust_radius = std::max(config_.gn_trust_radius_initial, config_.gn_trust_radius_min);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        RefinementIterReport iter_report;
        iter_report.iter = iter + 1;
        iter_report.quality_before = current_quality;
        const double before_rel = current_quality.cross_product.rel_rms_mismatch;
        const double current_selector =
            current_quality.cross_product.rel_rms_mismatch +
            config_.projection_proxy_acceptance_weight * current_proxy.combined_score;

        std::vector<double> residual0;
        ProjectionProxyReport residual_proxy;
        build_projection_aware_residual_vector(
            grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(), config_.invariance_weight,
            config_.projection_proxy_yz_weight, config_.projection_proxy_cond_weight,
            config_.projection_proxy_cond_floor, residual0, &residual_proxy);
        const size_t residual_dim = residual0.size();
        const int n_coeff = static_cast<int>(coeffs.size());
        std::vector<double> jacobian(residual_dim * static_cast<size_t>(n_coeff), 0.0);

        std::vector<double> trial_linear1, trial_quad1, trial_linear2, trial_quad2;
        std::vector<float> trial_psi1;
        std::vector<float> trial_psi2;

        for (int ci = 0; ci < n_coeff; ++ci) {
            std::vector<double> perturbed = coeffs;
            const double step = config_.gn_fd_absolute_step +
                                config_.gn_fd_relative_step * std::max(1.0, std::fabs(coeffs[ci]));
            perturbed[ci] += step;
            if (!unpack_subspace_coefficients(bank, perturbed, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                report.stop_reason = "projection_failed";
                return report;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            std::vector<double> residual_fd;
            build_projection_aware_residual_vector(
                grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(), inv.Lz(),
                config_.invariance_weight, config_.projection_proxy_yz_weight,
                config_.projection_proxy_cond_weight, config_.projection_proxy_cond_floor,
                residual_fd, nullptr);
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(ci)] =
                    (residual_fd[ri] - residual0[ri]) / step;
            }
        }

        std::vector<double> jtj(static_cast<size_t>(n_coeff) * n_coeff, 0.0);
        std::vector<double> jtr(static_cast<size_t>(n_coeff), 0.0);
        for (int row = 0; row < n_coeff; ++row) {
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                const double Jr =
                    jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)];
                jtr[static_cast<size_t>(row)] += Jr * residual0[ri];
            }
            for (int col = 0; col < n_coeff; ++col) {
                double value = 0.0;
                for (size_t ri = 0; ri < residual_dim; ++ri) {
                    value +=
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)] *
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(col)];
                }
                jtj[static_cast<size_t>(row) * n_coeff + col] = value;
            }
        }

        std::vector<double> system = jtj;
        std::vector<double> rhs(static_cast<size_t>(n_coeff), 0.0);
        for (int i = 0; i < n_coeff; ++i) {
            const double diag = jtj[static_cast<size_t>(i) * n_coeff + i];
            system[static_cast<size_t>(i) * n_coeff + i] += lambda * (diag + 1.0);
            rhs[static_cast<size_t>(i)] = -jtr[static_cast<size_t>(i)];
        }
        if (!solve_dense_system(system, rhs, n_coeff)) {
            trust_radius *= 0.5;
            lambda *= config_.gn_lambda_up;
            iter_report.best_trial_phase = "proj_gn";
            iter_report.best_trial_rejection_reason = "linear_solve_failed";
            iter_report.best_trial_rel_mismatch = before_rel;
            iter_report.quality_after = current_quality;
            report.history.push_back(iter_report);
            report.iterations_done = iter + 1;
            if (trust_radius < config_.gn_trust_radius_min) {
                report.stop_reason = "trust_region_exhausted";
                break;
            }
            continue;
        }

        double delta_norm = l2_norm(rhs);
        if (delta_norm > trust_radius && delta_norm > 0.0) {
            const double scale = trust_radius / delta_norm;
            for (double& value : rhs)
                value *= scale;
            delta_norm = trust_radius;
        }

        bool accepted = false;
        InvariantQualityReport accepted_quality;
        ProjectionProxyReport accepted_proxy;
        std::vector<double> accepted_coeffs = coeffs;
        std::vector<float> accepted_psi1 = psi1_host;
        std::vector<float> accepted_psi2 = psi2_host;
        double accepted_scale = 0.0;
        double accepted_selector = current_selector;

        bool have_best_trial = false;
        for (int backtrack = 0; backtrack < config_.max_backtracks; ++backtrack) {
            const double scale = std::ldexp(1.0, -backtrack);
            std::vector<double> trial_coeffs = coeffs;
            for (int i = 0; i < n_coeff; ++i)
                trial_coeffs[static_cast<size_t>(i)] += scale * rhs[static_cast<size_t>(i)];

            if (!unpack_subspace_coefficients(bank, trial_coeffs, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                continue;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            upload_field_pair(trial_psi1, trial_psi2, inv, ctx.cuda_stream());

            const InvariantQualityReport trial_quality =
                inv.compute_quality(*vel_, ctx.cuda_stream());
            const FieldStats trial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
            const ProjectionProxyReport trial_proxy = compute_projection_proxy_host(
                grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(), inv.Lz(),
                config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);
            const std::string trial_rejection =
                rejection_reason(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_);
            const double trial_selector =
                trial_quality.cross_product.rel_rms_mismatch +
                config_.projection_proxy_acceptance_weight * trial_proxy.combined_score;

            if (!have_best_trial || trial_quality.cross_product.rel_rms_mismatch <
                                        iter_report.best_trial_rel_mismatch) {
                have_best_trial = true;
                iter_report.best_trial_phase = "proj_gn(scale=" + std::to_string(scale) + ")";
                iter_report.best_trial_rejection_reason = trial_rejection;
                iter_report.best_trial_rel_mismatch = trial_quality.cross_product.rel_rms_mismatch;
                iter_report.best_trial_invariance_sum =
                    trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
                iter_report.best_trial_degeneracy = trial_quality.independence.degeneracy_score;
                iter_report.best_trial_min_gradient_rms = trial_stats.min_gradient_rms;
                iter_report.best_trial_min_field_range = trial_stats.min_field_range;
            }

            const bool selector_improved = trial_selector + 1.0e-8 < current_selector;
            if (trial_admissible(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_) &&
                selector_improved) {
                if (!accepted || trial_selector < accepted_selector) {
                    accepted = true;
                    accepted_quality = trial_quality;
                    accepted_proxy = trial_proxy;
                    accepted_coeffs = std::move(trial_coeffs);
                    accepted_psi1 = std::move(trial_psi1);
                    accepted_psi2 = std::move(trial_psi2);
                    accepted_scale = scale;
                    accepted_selector = trial_selector;
                }
            }
        }

        if (accepted) {
            coeffs = std::move(accepted_coeffs);
            psi1_host = std::move(accepted_psi1);
            psi2_host = std::move(accepted_psi2);
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            current_quality = accepted_quality;
            current_proxy = accepted_proxy;
            iter_report.accepted = true;
            iter_report.omega_accepted = accepted_scale;
            lambda = std::max(lambda * config_.gn_lambda_down, 1.0e-8);
            trust_radius =
                std::min(std::max(trust_radius, delta_norm) * 1.25, config_.gn_trust_radius_max);
        } else {
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            lambda *= config_.gn_lambda_up;
            trust_radius *= 0.5;
            if (have_best_trial && iter_report.best_trial_rejection_reason == "admissible") {
                iter_report.best_trial_rejection_reason = "projection_proxy_not_improved";
            }
        }

        iter_report.quality_after = inv.compute_quality(*vel_, ctx.cuda_stream());
        current_quality = iter_report.quality_after;
        const double after_rel = current_quality.cross_product.rel_rms_mismatch;
        iter_report.rel_improvement =
            (before_rel > 1.0e-12) ? (before_rel - after_rel) / before_rel : 0.0;

        report.history.push_back(iter_report);
        report.iterations_done = iter + 1;

        if (config_.verbose) {
            std::printf("  [StrategyC-ProjGN] iter=%d accepted=%d lambda=%.3e trust=%.3e "
                        "rel=%.6e -> %.6e proxy=%.6e\n",
                        iter + 1, iter_report.accepted ? 1 : 0, lambda, trust_radius, before_rel,
                        after_rel, current_proxy.combined_score);
        }

        if (!accepted && trust_radius < config_.gn_trust_radius_min) {
            report.stop_reason = "trust_region_exhausted";
            break;
        }
        if (current_quality.cross_product.rel_rms_mismatch <= config_.stop_abs_quality) {
            report.converged = true;
            report.stop_reason = "abs_quality";
            break;
        }
        if (iter_report.rel_improvement < config_.stop_rel_quality && accepted) {
            report.converged = true;
            report.stop_reason = "rel_improvement";
            break;
        }
    }

    if (report.stop_reason.empty())
        report.stop_reason = report.converged ? "converged" : "max_iterations";

    report.final_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats final_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.final_min_gradient_rms = final_stats.min_gradient_rms;
    report.final_min_field_range = final_stats.min_field_range;
    report.final_projection = compute_projection_proxy_host(
        grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(),
        config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);

    auto info = inv.construction_info();
    info.method = InvariantConstructionMethod::StrategyAC;
    info.refinement_iterations = report.iterations_done;
    info.refinement_omega = config_.gn_trust_radius_initial;
    info.refinement_final_rms = report.final_quality.cross_product.rel_rms_mismatch;
    info.refinement_stop_reason = report.stop_reason;
    info.gauge_method = "subspace_quadratic_proj_gn";
    inv.set_construction_info(info);

    const auto t1 = std::chrono::steady_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return report;
}

RefinementACReport
RefinementAC::refine_subspace_quadratic_gauss_newton_engine_proxy(PsptaInvariantField& inv,
                                                                  CudaContext& ctx) {
    RefinementACReport report;
    report.enabled = config_.enabled;

    if (!config_.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (!inv.is_valid() || vel_ == nullptr) {
        report.stop_reason = "invalid_input";
        return report;
    }
    if (subspace_basis_host_.size() < 2) {
        report.stop_reason = "invalid_subspace";
        return report;
    }
    if (subspace_basis_host_.size() > 4)
        subspace_basis_host_.resize(4);
    for (const auto& mode : subspace_basis_host_) {
        if (mode.size() != inv.num_cells()) {
            report.stop_reason = "invalid_subspace";
            return report;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = inv.num_cells();
    report.initial_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats initial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.initial_min_gradient_rms = initial_stats.min_gradient_rms;
    report.initial_min_field_range = initial_stats.min_field_range;

    SubspaceFeatureBank bank;
    if (!build_subspace_feature_bank(subspace_basis_host_, bank)) {
        report.stop_reason = "invalid_subspace";
        return report;
    }

    HostVelocityCache vel_cache;
    if (!download_velocity_cache(grid_, *vel_, ctx.cuda_stream(), vel_cache)) {
        report.stop_reason = "velocity_download_failed";
        return report;
    }

    const std::vector<float> initial_psi1 = download_field(inv.psi1_ptr(), n, ctx.cuda_stream());
    const std::vector<float> initial_psi2 = download_field(inv.psi2_ptr(), n, ctx.cuda_stream());

    std::vector<double> linear1, quad1, linear2, quad2;
    if (!project_field_to_all_features(bank, initial_psi1, linear1, quad1) ||
        !project_field_to_all_features(bank, initial_psi2, linear2, quad2)) {
        report.stop_reason = "projection_failed";
        return report;
    }

    std::vector<double> coeffs;
    pack_subspace_coefficients(linear1, quad1, linear2, quad2, coeffs);

    std::vector<float> psi1_host;
    std::vector<float> psi2_host;
    synthesize_subspace_fields(bank, linear1, linear2, quad1, quad2, psi1_host, psi2_host);
    upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());

    InvariantQualityReport current_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    report.initial_quality = current_quality;
    ProjectionProxyReport current_projection = compute_projection_proxy_host(
        grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(),
        config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);
    EngineSampledProxyReport current_engine = compute_engine_sampled_proxy_host(
        grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(), config_);
    report.initial_projection = current_projection;
    report.initial_engine = current_engine;

    double lambda = std::max(config_.gn_lambda_initial, 1.0e-8);
    double trust_radius = std::max(config_.gn_trust_radius_initial, config_.gn_trust_radius_min);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        RefinementIterReport iter_report;
        iter_report.iter = iter + 1;
        iter_report.quality_before = current_quality;
        const double before_rel = current_quality.cross_product.rel_rms_mismatch;
        const EngineProxySelectorValue current_selector =
            make_engine_proxy_selector(current_quality, current_engine, config_);

        std::vector<double> residual0;
        ProjectionProxyReport residual_proxy;
        build_projection_aware_residual_vector(
            grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(), config_.invariance_weight,
            config_.projection_proxy_yz_weight, config_.projection_proxy_cond_weight,
            config_.projection_proxy_cond_floor, residual0, &residual_proxy);
        const size_t residual_dim = residual0.size();
        const int n_coeff = static_cast<int>(coeffs.size());
        std::vector<double> jacobian(residual_dim * static_cast<size_t>(n_coeff), 0.0);

        std::vector<double> trial_linear1, trial_quad1, trial_linear2, trial_quad2;
        std::vector<float> trial_psi1;
        std::vector<float> trial_psi2;

        for (int ci = 0; ci < n_coeff; ++ci) {
            std::vector<double> perturbed = coeffs;
            const double step = config_.gn_fd_absolute_step +
                                config_.gn_fd_relative_step * std::max(1.0, std::fabs(coeffs[ci]));
            perturbed[ci] += step;
            if (!unpack_subspace_coefficients(bank, perturbed, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                report.stop_reason = "projection_failed";
                return report;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            std::vector<double> residual_fd;
            build_projection_aware_residual_vector(
                grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(), inv.Lz(),
                config_.invariance_weight, config_.projection_proxy_yz_weight,
                config_.projection_proxy_cond_weight, config_.projection_proxy_cond_floor,
                residual_fd, nullptr);
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(ci)] =
                    (residual_fd[ri] - residual0[ri]) / step;
            }
        }

        std::vector<double> jtj(static_cast<size_t>(n_coeff) * n_coeff, 0.0);
        std::vector<double> jtr(static_cast<size_t>(n_coeff), 0.0);
        for (int row = 0; row < n_coeff; ++row) {
            for (size_t ri = 0; ri < residual_dim; ++ri) {
                const double Jr =
                    jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)];
                jtr[static_cast<size_t>(row)] += Jr * residual0[ri];
            }
            for (int col = 0; col < n_coeff; ++col) {
                double value = 0.0;
                for (size_t ri = 0; ri < residual_dim; ++ri) {
                    value +=
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(row)] *
                        jacobian[ri * static_cast<size_t>(n_coeff) + static_cast<size_t>(col)];
                }
                jtj[static_cast<size_t>(row) * n_coeff + col] = value;
            }
        }

        std::vector<double> system = jtj;
        std::vector<double> rhs(static_cast<size_t>(n_coeff), 0.0);
        for (int i = 0; i < n_coeff; ++i) {
            const double diag = jtj[static_cast<size_t>(i) * n_coeff + i];
            system[static_cast<size_t>(i) * n_coeff + i] += lambda * (diag + 1.0);
            rhs[static_cast<size_t>(i)] = -jtr[static_cast<size_t>(i)];
        }
        if (!solve_dense_system(system, rhs, n_coeff)) {
            trust_radius *= 0.5;
            lambda *= config_.gn_lambda_up;
            iter_report.best_trial_phase = "engine_gn";
            iter_report.best_trial_rejection_reason = "linear_solve_failed";
            iter_report.best_trial_rel_mismatch = before_rel;
            iter_report.quality_after = current_quality;
            report.history.push_back(iter_report);
            report.iterations_done = iter + 1;
            if (trust_radius < config_.gn_trust_radius_min) {
                report.stop_reason = "trust_region_exhausted";
                break;
            }
            continue;
        }

        double delta_norm = l2_norm(rhs);
        if (delta_norm > trust_radius && delta_norm > 0.0) {
            const double scale = trust_radius / delta_norm;
            for (double& value : rhs)
                value *= scale;
            delta_norm = trust_radius;
        }

        bool accepted = false;
        InvariantQualityReport accepted_quality;
        ProjectionProxyReport accepted_projection;
        EngineSampledProxyReport accepted_engine;
        std::vector<double> accepted_coeffs = coeffs;
        std::vector<float> accepted_psi1 = psi1_host;
        std::vector<float> accepted_psi2 = psi2_host;
        double accepted_scale = 0.0;
        EngineProxySelectorValue accepted_selector = current_selector;

        bool have_best_trial = false;
        for (int backtrack = 0; backtrack < config_.max_backtracks; ++backtrack) {
            const double scale = std::ldexp(1.0, -backtrack);
            std::vector<double> trial_coeffs = coeffs;
            for (int i = 0; i < n_coeff; ++i)
                trial_coeffs[static_cast<size_t>(i)] += scale * rhs[static_cast<size_t>(i)];

            if (!unpack_subspace_coefficients(bank, trial_coeffs, trial_linear1, trial_quad1,
                                              trial_linear2, trial_quad2)) {
                continue;
            }
            synthesize_subspace_fields(bank, trial_linear1, trial_linear2, trial_quad1, trial_quad2,
                                       trial_psi1, trial_psi2);
            upload_field_pair(trial_psi1, trial_psi2, inv, ctx.cuda_stream());

            const InvariantQualityReport trial_quality =
                inv.compute_quality(*vel_, ctx.cuda_stream());
            const FieldStats trial_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
            const ProjectionProxyReport trial_projection = compute_projection_proxy_host(
                grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(), inv.Lz(),
                config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);
            const EngineSampledProxyReport trial_engine = compute_engine_sampled_proxy_host(
                grid_, vel_cache, trial_psi1, trial_psi2, inv.Ly(), inv.Lz(), config_);
            const std::string trial_rejection =
                rejection_reason(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_);
            const EngineProxySelectorValue trial_selector =
                make_engine_proxy_selector(trial_quality, trial_engine, config_);

            if (!have_best_trial || trial_quality.cross_product.rel_rms_mismatch <
                                        iter_report.best_trial_rel_mismatch) {
                have_best_trial = true;
                iter_report.best_trial_phase = "engine_gn(scale=" + std::to_string(scale) + ")";
                iter_report.best_trial_rejection_reason = trial_rejection;
                iter_report.best_trial_rel_mismatch = trial_quality.cross_product.rel_rms_mismatch;
                iter_report.best_trial_invariance_sum =
                    trial_quality.invariance.rms_r1 + trial_quality.invariance.rms_r2;
                iter_report.best_trial_degeneracy = trial_quality.independence.degeneracy_score;
                iter_report.best_trial_min_gradient_rms = trial_stats.min_gradient_rms;
                iter_report.best_trial_min_field_range = trial_stats.min_field_range;
            }

            const bool selector_improved =
                engine_proxy_selector_better(trial_selector, current_selector);
            if (trial_admissible(trial_quality, trial_stats, current_quality,
                                 report.initial_quality, initial_stats, config_) &&
                selector_improved) {
                if (!accepted || engine_proxy_selector_better(trial_selector, accepted_selector)) {
                    accepted = true;
                    accepted_quality = trial_quality;
                    accepted_projection = trial_projection;
                    accepted_engine = trial_engine;
                    accepted_coeffs = std::move(trial_coeffs);
                    accepted_psi1 = std::move(trial_psi1);
                    accepted_psi2 = std::move(trial_psi2);
                    accepted_scale = scale;
                    accepted_selector = trial_selector;
                }
            }
        }

        if (accepted) {
            coeffs = std::move(accepted_coeffs);
            psi1_host = std::move(accepted_psi1);
            psi2_host = std::move(accepted_psi2);
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            current_quality = accepted_quality;
            current_projection = accepted_projection;
            current_engine = accepted_engine;
            iter_report.accepted = true;
            iter_report.omega_accepted = accepted_scale;
            lambda = std::max(lambda * config_.gn_lambda_down, 1.0e-8);
            trust_radius =
                std::min(std::max(trust_radius, delta_norm) * 1.25, config_.gn_trust_radius_max);
        } else {
            upload_field_pair(psi1_host, psi2_host, inv, ctx.cuda_stream());
            lambda *= config_.gn_lambda_up;
            trust_radius *= 0.5;
            if (have_best_trial && iter_report.best_trial_rejection_reason == "admissible") {
                iter_report.best_trial_rejection_reason = "engine_proxy_not_improved";
            }
        }

        iter_report.quality_after = inv.compute_quality(*vel_, ctx.cuda_stream());
        current_quality = iter_report.quality_after;
        const double after_rel = current_quality.cross_product.rel_rms_mismatch;
        iter_report.rel_improvement =
            (before_rel > 1.0e-12) ? (before_rel - after_rel) / before_rel : 0.0;

        report.history.push_back(iter_report);
        report.iterations_done = iter + 1;

        if (config_.verbose) {
            std::printf("  [StrategyC-EngineGN] iter=%d accepted=%d lambda=%.3e trust=%.3e "
                        "rel=%.6e -> %.6e eng_fail=%.3e eng_score=%.6e\n",
                        iter + 1, iter_report.accepted ? 1 : 0, lambda, trust_radius, before_rel,
                        after_rel, current_engine.fail_fraction, current_engine.combined_score);
        }

        if (!accepted && trust_radius < config_.gn_trust_radius_min) {
            report.stop_reason = "trust_region_exhausted";
            break;
        }
        if (current_quality.cross_product.rel_rms_mismatch <= config_.stop_abs_quality) {
            report.converged = true;
            report.stop_reason = "abs_quality";
            break;
        }
        if (iter_report.rel_improvement < config_.stop_rel_quality && accepted) {
            report.converged = true;
            report.stop_reason = "rel_improvement";
            break;
        }
    }

    if (report.stop_reason.empty())
        report.stop_reason = report.converged ? "converged" : "max_iterations";

    report.final_quality = inv.compute_quality(*vel_, ctx.cuda_stream());
    const FieldStats final_stats = compute_field_stats_host(grid_, inv, ctx.cuda_stream());
    report.final_min_gradient_rms = final_stats.min_gradient_rms;
    report.final_min_field_range = final_stats.min_field_range;
    report.final_projection = compute_projection_proxy_host(
        grid_, vel_cache, psi1_host, psi2_host, inv.Ly(), inv.Lz(),
        config_.projection_proxy_cond_floor, config_.projection_proxy_cond_weight);
    report.final_engine = compute_engine_sampled_proxy_host(grid_, vel_cache, psi1_host, psi2_host,
                                                            inv.Ly(), inv.Lz(), config_);

    auto info = inv.construction_info();
    info.method = InvariantConstructionMethod::StrategyAC;
    info.refinement_iterations = report.iterations_done;
    info.refinement_omega = config_.gn_trust_radius_initial;
    info.refinement_final_rms = report.final_quality.cross_product.rel_rms_mismatch;
    info.refinement_stop_reason = report.stop_reason;
    info.gauge_method = "subspace_quadratic_engine_gn";
    inv.set_construction_info(info);

    const auto t1 = std::chrono::steady_clock::now();
    report.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return report;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
