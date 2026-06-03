/**
 * @file test_kh_potential_reconstruction.cu
 * @brief Manufactured-field tests for KH potential-flow velocity evaluation.
 */

#include "src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh"

#include <cuda_runtime.h>
#include <par2_core/grid.hpp>
#include <par2_core/types.hpp>
#include <par2_core/views.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define TEST_CUDA_CHECK(call)                                                                      \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));     \
        }                                                                                          \
    } while (0)

constexpr double kPi = 3.141592653589793238462643383279502884;

struct OrderErrorRow {
    int n = 0;
    double linear_rel_l2 = 0.0;
    double cubic_rel_l2 = 0.0;
    double logk_rel_l2 = 0.0;
};

__global__ void sample_kh_kernel(const par2::GridDesc<double> grid,
                                 const par2::PotentialFlowView<double> potential,
                                 const int velocity_mode, const double* __restrict__ x,
                                 const double* __restrict__ y, const double* __restrict__ z,
                                 double* __restrict__ vx, double* __restrict__ vy,
                                 double* __restrict__ vz, int n) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        par2::internal::sample_velocity_potential_backend(
            potential, grid, static_cast<par2::VelocityEvalMode>(velocity_mode), x[tid], y[tid],
            z[tid], vx[tid], vy[tid], vz[tid]);
    }
}

__global__ void sample_conductivity_kernel(const par2::GridDesc<double> grid,
                                           const par2::PotentialFlowView<double> potential,
                                           const int velocity_mode, const double* __restrict__ x,
                                           const double* __restrict__ y,
                                           const double* __restrict__ z, double* __restrict__ kout,
                                           double* __restrict__ logk_out, int n) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        double logk = 0.0;
        kout[tid] = par2::internal::sample_conductivity_potential_backend(
            potential, grid, static_cast<par2::VelocityEvalMode>(velocity_mode), x[tid], y[tid],
            z[tid], &logk);
        logk_out[tid] = logk;
    }
}

size_t idx(const par2::GridDesc<double>& grid, int i, int j, int k) {
    return static_cast<size_t>(i) + static_cast<size_t>(grid.nx) *
                                        (static_cast<size_t>(j) + static_cast<size_t>(grid.ny) * k);
}

double x_center(const par2::GridDesc<double>& grid, int i) {
    return grid.px + (static_cast<double>(i) + 0.5) * grid.dx;
}

double y_center(const par2::GridDesc<double>& grid, int j) {
    return grid.py + (static_cast<double>(j) + 0.5) * grid.dy;
}

double z_center(const par2::GridDesc<double>& grid, int k) {
    return grid.pz + (static_cast<double>(k) + 0.5) * grid.dz;
}

struct DeviceArrays {
    double* K = nullptr;
    double* head = nullptr;
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    double* kout = nullptr;
    double* logk = nullptr;

    ~DeviceArrays() {
        cudaFree(K);
        cudaFree(head);
        cudaFree(x);
        cudaFree(y);
        cudaFree(z);
        cudaFree(vx);
        cudaFree(vy);
        cudaFree(vz);
        cudaFree(kout);
        cudaFree(logk);
    }
};

void reset_sampling_buffers(DeviceArrays& dev) {
    cudaFree(dev.x);
    cudaFree(dev.y);
    cudaFree(dev.z);
    cudaFree(dev.vx);
    cudaFree(dev.vy);
    cudaFree(dev.vz);
    cudaFree(dev.kout);
    cudaFree(dev.logk);
    dev.x = nullptr;
    dev.y = nullptr;
    dev.z = nullptr;
    dev.vx = nullptr;
    dev.vy = nullptr;
    dev.vz = nullptr;
    dev.kout = nullptr;
    dev.logk = nullptr;
}

par2::PotentialFlowView<double> make_view(const par2::GridDesc<double>& grid, DeviceArrays& dev,
                                          par2::ScalarBoundaryType x_type, double h_xlo,
                                          double h_xhi, bool periodic_yz) {
    par2::PotentialFlowView<double> view;
    view.K = dev.K;
    view.head = dev.head;
    view.size = static_cast<size_t>(grid.num_cells());
    view.head_bc.x.lo.type = x_type;
    view.head_bc.x.lo.value = h_xlo;
    view.head_bc.x.hi.type = x_type;
    view.head_bc.x.hi.value = h_xhi;
    view.head_bc.y.lo.type =
        periodic_yz ? par2::ScalarBoundaryType::Periodic : par2::ScalarBoundaryType::Extrapolate;
    view.head_bc.y.hi.type = view.head_bc.y.lo.type;
    view.head_bc.z.lo.type = view.head_bc.y.lo.type;
    view.head_bc.z.hi.type = view.head_bc.y.lo.type;
    return view;
}

void upload_field(DeviceArrays& dev, const std::vector<double>& field, double*& dst) {
    TEST_CUDA_CHECK(cudaMalloc(&dst, field.size() * sizeof(double)));
    TEST_CUDA_CHECK(
        cudaMemcpy(dst, field.data(), field.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void run_sampler(const par2::GridDesc<double>& grid, const par2::PotentialFlowView<double>& view,
                 par2::VelocityEvalMode velocity_mode, DeviceArrays& dev,
                 const std::vector<double>& hx, const std::vector<double>& hy,
                 const std::vector<double>& hz, std::vector<double>& vx, std::vector<double>& vy,
                 std::vector<double>& vz) {
    const int n = static_cast<int>(hx.size());
    reset_sampling_buffers(dev);
    TEST_CUDA_CHECK(cudaMalloc(&dev.x, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.y, hy.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.z, hz.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.vx, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.vy, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.vz, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.x, hx.data(), hx.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.y, hy.data(), hy.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.z, hz.data(), hz.size() * sizeof(double), cudaMemcpyHostToDevice));

    sample_kh_kernel<<<1, 128>>>(grid, view, static_cast<int>(velocity_mode), dev.x, dev.y, dev.z,
                                 dev.vx, dev.vy, dev.vz, n);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    vx.resize(hx.size());
    vy.resize(hx.size());
    vz.resize(hx.size());
    TEST_CUDA_CHECK(
        cudaMemcpy(vx.data(), dev.vx, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(
        cudaMemcpy(vy.data(), dev.vy, hy.size() * sizeof(double), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(
        cudaMemcpy(vz.data(), dev.vz, hz.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

void run_conductivity_sampler(const par2::GridDesc<double>& grid,
                              const par2::PotentialFlowView<double>& view,
                              par2::VelocityEvalMode velocity_mode, DeviceArrays& dev,
                              const std::vector<double>& hx, const std::vector<double>& hy,
                              const std::vector<double>& hz, std::vector<double>& kout,
                              std::vector<double>& logk_out) {
    const int n = static_cast<int>(hx.size());
    reset_sampling_buffers(dev);
    TEST_CUDA_CHECK(cudaMalloc(&dev.x, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.y, hy.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.z, hz.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.kout, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.logk, hx.size() * sizeof(double)));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.x, hx.data(), hx.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.y, hy.data(), hy.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.z, hz.data(), hz.size() * sizeof(double), cudaMemcpyHostToDevice));

    sample_conductivity_kernel<<<1, 128>>>(grid, view, static_cast<int>(velocity_mode), dev.x,
                                           dev.y, dev.z, dev.kout, dev.logk, n);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    kout.resize(hx.size());
    logk_out.resize(hx.size());
    TEST_CUDA_CHECK(
        cudaMemcpy(kout.data(), dev.kout, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(
        cudaMemcpy(logk_out.data(), dev.logk, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

void expect_close(double got, double expected, double tol, const char* label) {
    if (!std::isfinite(got)) {
        std::printf("FAIL %s: got non-finite %.17g\n", label, got);
        throw std::runtime_error("non-finite KH value");
    }
    if (std::abs(got - expected) > tol) {
        std::printf("FAIL %s: got %.17g expected %.17g tol %.3e\n", label, got, expected, tol);
        throw std::runtime_error("KH value mismatch");
    }
}

double manufactured_cubic_head(double x, double y, double z) {
    return 0.11 * x * x * x - 0.07 * y * y * y + 0.05 * z * z * z + 0.09 * x * x * y -
           0.04 * x * y * z + 0.03 * y * z * z + 0.02 * x - 0.01 * y + 0.04 * z + 1.5;
}

void manufactured_cubic_grad(double x, double y, double z, double& dhdx, double& dhdy,
                             double& dhdz) {
    dhdx = 0.33 * x * x + 0.18 * x * y - 0.04 * y * z + 0.02;
    dhdy = -0.21 * y * y + 0.09 * x * x - 0.04 * x * z + 0.03 * z * z - 0.01;
    dhdz = 0.15 * z * z - 0.04 * x * y + 0.06 * y * z + 0.04;
}

double smooth_logk(double x, double y, double z, double lx, double ly, double lz) {
    const double gx = 2.0 * kPi / lx;
    const double gy = 2.0 * kPi / ly;
    const double gz = 2.0 * kPi / lz;
    return 0.20 * std::sin(gx * x) + 0.15 * std::cos(gy * y) - 0.10 * std::sin(gz * z);
}

double smooth_order_head(double x, double y, double z, double lx, double ly, double lz) {
    const double gx = 2.0 * kPi / lx;
    const double gy = 2.0 * kPi / ly;
    const double gz = 2.0 * kPi / lz;
    return std::sin(gx * x) + 0.5 * std::cos(gy * y) - 0.25 * std::sin(gz * z) +
           0.125 * std::sin(gx * x + gy * y) + 0.05 * std::cos(gz * z - gx * x);
}

void smooth_order_grad(double x, double y, double z, double lx, double ly, double lz, double& dhdx,
                       double& dhdy, double& dhdz) {
    const double gx = 2.0 * kPi / lx;
    const double gy = 2.0 * kPi / ly;
    const double gz = 2.0 * kPi / lz;
    dhdx = gx * std::cos(gx * x) + 0.125 * gx * std::cos(gx * x + gy * y) +
           0.05 * gx * std::sin(gz * z - gx * x);
    dhdy = -0.5 * gy * std::sin(gy * y) + 0.125 * gy * std::cos(gx * x + gy * y);
    dhdz = -0.25 * gz * std::cos(gz * z) - 0.05 * gz * std::sin(gz * z - gx * x);
}

OrderErrorRow compute_order_error_row(int n) {
    const double length = 2.0 * kPi;
    const double spacing = length / static_cast<double>(n);
    const auto grid = par2::make_grid<double>(n, n, n, spacing, spacing, spacing);
    const double K0 = 1.7;

    std::vector<double> K(grid.num_cells(), K0);
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                head[idx(grid, i, j, k)] =
                    smooth_order_head(x_center(grid, i), y_center(grid, j), z_center(grid, k),
                                      length, length, length);
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);
    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Extrapolate, 0.0, 0.0, true);

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    const double fractions[4] = {0.21, 0.37, 0.58, 0.79};
    for (double fx : fractions) {
        for (double fy : fractions) {
            for (double fz : fractions) {
                xs.push_back(fx * length);
                ys.push_back(fy * length);
                zs.push_back(fz * length);
            }
        }
    }

    std::vector<double> vx_linear, vy_linear, vz_linear;
    std::vector<double> vx_cubic, vy_cubic, vz_cubic;
    std::vector<double> vx_logk, vy_logk, vz_logk;
    run_sampler(grid, view, par2::VelocityEvalMode::KhLinear, dev, xs, ys, zs, vx_linear, vy_linear,
                vz_linear);
    run_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction, dev, xs, ys, zs,
                vx_cubic, vy_cubic, vz_cubic);
    run_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction, dev, xs, ys,
                zs, vx_logk, vy_logk, vz_logk);

    double linear_num = 0.0;
    double cubic_num = 0.0;
    double logk_num = 0.0;
    double denom = 0.0;
    for (size_t p = 0; p < xs.size(); ++p) {
        double dhdx = 0.0;
        double dhdy = 0.0;
        double dhdz = 0.0;
        smooth_order_grad(xs[p], ys[p], zs[p], length, length, length, dhdx, dhdy, dhdz);
        const double qx = -K0 * dhdx;
        const double qy = -K0 * dhdy;
        const double qz = -K0 * dhdz;
        denom += qx * qx + qy * qy + qz * qz;
        linear_num += (vx_linear[p] - qx) * (vx_linear[p] - qx) +
                      (vy_linear[p] - qy) * (vy_linear[p] - qy) +
                      (vz_linear[p] - qz) * (vz_linear[p] - qz);
        cubic_num += (vx_cubic[p] - qx) * (vx_cubic[p] - qx) +
                     (vy_cubic[p] - qy) * (vy_cubic[p] - qy) +
                     (vz_cubic[p] - qz) * (vz_cubic[p] - qz);
        logk_num += (vx_logk[p] - qx) * (vx_logk[p] - qx) + (vy_logk[p] - qy) * (vy_logk[p] - qy) +
                    (vz_logk[p] - qz) * (vz_logk[p] - qz);
    }

    OrderErrorRow row;
    row.n = n;
    row.linear_rel_l2 = std::sqrt(linear_num / denom);
    row.cubic_rel_l2 = std::sqrt(cubic_num / denom);
    row.logk_rel_l2 = std::sqrt(logk_num / denom);
    return row;
}

void write_order_csv(const std::string& path, const std::vector<OrderErrorRow>& rows) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        throw std::runtime_error("cannot open manufactured order CSV");
    }
    std::fprintf(f, "n,linear_rel_l2,cubic_rel_l2,logk_rel_l2\n");
    for (const auto& row : rows) {
        std::fprintf(f, "%d,%.17g,%.17g,%.17g\n", row.n, row.linear_rel_l2, row.cubic_rel_l2,
                     row.logk_rel_l2);
    }
    std::fclose(f);
}

void test_linear_head_constant_k() {
    const auto grid = par2::make_grid<double>(8, 6, 5, 1.0, 1.0, 1.0);
    const double grad_x = -3.25;
    const double h0 = 12.0;
    const double K0 = 2.0;

    std::vector<double> K(grid.num_cells(), K0);
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                head[idx(grid, i, j, k)] = h0 + grad_x * x_center(grid, i);
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Dirichlet, h0,
                                h0 + grad_x * grid.length_x(), false);
    std::vector<double> vx, vy, vz;
    const par2::VelocityEvalMode modes[] = {
        par2::VelocityEvalMode::KhLinear,
        par2::VelocityEvalMode::KhCubicPotentialReconstruction,
        par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction,
    };
    for (const auto mode : modes) {
        run_sampler(grid, view, mode, dev, {0.2, 2.3, 7.8}, {1.7, 3.1, 4.4}, {0.6, 2.2, 3.8}, vx,
                    vy, vz);
        for (size_t n = 0; n < vx.size(); ++n) {
            expect_close(vx[n], -K0 * grad_x, 1e-10, "linear constant K vx");
            expect_close(vy[n], 0.0, 1e-10, "linear constant K vy");
            expect_close(vz[n], 0.0, 1e-10, "linear constant K vz");
        }
    }
}

void test_linear_head_variable_k() {
    const auto grid = par2::make_grid<double>(10, 8, 6, 1.0, 1.0, 1.0);
    const double grad_x = 1.75;
    const double h0 = -4.0;

    std::vector<double> K(grid.num_cells());
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const double x = x_center(grid, i);
                const double y = y_center(grid, j);
                const double z = z_center(grid, k);
                K[idx(grid, i, j, k)] = 1.0 + 0.03 * x + 0.02 * y + 0.01 * z;
                head[idx(grid, i, j, k)] = h0 + grad_x * x;
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Dirichlet, h0,
                                h0 + grad_x * grid.length_x(), false);
    std::vector<double> vx, vy, vz;
    const std::vector<double> xs{2.25, 4.5, 7.25};
    const std::vector<double> ys{2.0, 5.25, 3.75};
    const std::vector<double> zs{1.75, 2.5, 4.25};
    const par2::VelocityEvalMode modes[] = {
        par2::VelocityEvalMode::KhLinear,
        par2::VelocityEvalMode::KhCubicPotentialReconstruction,
    };
    for (const auto mode : modes) {
        run_sampler(grid, view, mode, dev, xs, ys, zs, vx, vy, vz);
        for (size_t n = 0; n < vx.size(); ++n) {
            const double k_expected = 1.0 + 0.03 * xs[n] + 0.02 * ys[n] + 0.01 * zs[n];
            expect_close(vx[n], -k_expected * grad_x, 1e-6, "linear variable K vx");
            expect_close(vy[n], 0.0, 1e-10, "linear variable K vy");
            expect_close(vz[n], 0.0, 1e-10, "linear variable K vz");
        }
    }
}

void test_linear_head_smooth_logk() {
    const auto grid = par2::make_grid<double>(24, 20, 18, 1.0, 1.0, 1.0);
    const double lx = grid.length_x();
    const double ly = grid.length_y();
    const double lz = grid.length_z();
    const double grad_x = 0.8;
    const double grad_y = -0.35;
    const double grad_z = 0.20;
    const double h0 = 2.0;

    std::vector<double> K(grid.num_cells());
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const double x = x_center(grid, i);
                const double y = y_center(grid, j);
                const double z = z_center(grid, k);
                const double Y = smooth_logk(x, y, z, lx, ly, lz);
                K[idx(grid, i, j, k)] = std::exp(Y);
                head[idx(grid, i, j, k)] = h0 + grad_x * x + grad_y * y + grad_z * z;
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Extrapolate, 0.0, 0.0, false);
    const std::vector<double> xs{3.1, 8.7, 14.2};
    const std::vector<double> ys{3.4, 8.8, 14.1};
    const std::vector<double> zs{2.7, 7.4, 12.2};

    std::vector<double> vx, vy, vz;
    std::vector<double> kcubic, logk_dummy;
    std::vector<double> klogk, ylogk;
    run_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction, dev, xs, ys, zs,
                vx, vy, vz);
    run_conductivity_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction,
                             dev, xs, ys, zs, kcubic, logk_dummy);
    for (size_t n = 0; n < xs.size(); ++n) {
        const double k_expected = std::exp(smooth_logk(xs[n], ys[n], zs[n], lx, ly, lz));
        expect_close(vx[n], -k_expected * grad_x, 2.0e-2, "smooth logK cubic vx");
        expect_close(vy[n], -k_expected * grad_y, 1.5e-2, "smooth logK cubic vy");
        expect_close(vz[n], -k_expected * grad_z, 1.0e-2, "smooth logK cubic vz");
        if (!(std::isfinite(kcubic[n]) && kcubic[n] > 0.0)) {
            throw std::runtime_error("direct cubic produced non-positive K in smooth logK test");
        }
    }

    run_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction, dev, xs, ys,
                zs, vx, vy, vz);
    run_conductivity_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction,
                             dev, xs, ys, zs, klogk, ylogk);
    for (size_t n = 0; n < xs.size(); ++n) {
        const double Y = smooth_logk(xs[n], ys[n], zs[n], lx, ly, lz);
        const double k_expected = std::exp(Y);
        expect_close(vx[n], -k_expected * grad_x, 1.5e-2, "smooth logK logk vx");
        expect_close(vy[n], -k_expected * grad_y, 1.0e-2, "smooth logK logk vy");
        expect_close(vz[n], -k_expected * grad_z, 8.0e-3, "smooth logK logk vz");
        if (!(std::isfinite(klogk[n]) && klogk[n] > 0.0)) {
            throw std::runtime_error("logK cubic produced non-positive K");
        }
        expect_close(std::log(klogk[n]), Y, 2.0e-2, "smooth logK reconstructed Y");
        expect_close(ylogk[n], Y, 2.0e-2, "smooth logK interpolated Y");
    }
}

void test_smooth_periodic_yz() {
    const auto grid = par2::make_grid<double>(48, 48, 48, 1.0, 1.0, 1.0);
    const double Ly = grid.length_y();
    const double Lz = grid.length_z();
    const double gy = 2.0 * kPi / Ly;
    const double gz = 2.0 * kPi / Lz;
    const double grad_x = 0.8;

    std::vector<double> K(grid.num_cells());
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const double x = x_center(grid, i);
                const double y = y_center(grid, j);
                const double z = z_center(grid, k);
                K[idx(grid, i, j, k)] = 1.0 + 0.1 * std::cos(gy * y) + 0.05 * std::sin(gz * z);
                head[idx(grid, i, j, k)] =
                    3.0 + grad_x * x + std::sin(gy * y) + 0.5 * std::cos(gz * z);
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Extrapolate, 0.0, 0.0, true);
    std::vector<double> vx, vy, vz;
    const std::vector<double> xs{12.3, 24.7, 31.5};
    const std::vector<double> ys{0.12, Ly - 0.08, 17.3};
    const std::vector<double> zs{Lz - 0.2, 0.18, 29.6};
    const par2::VelocityEvalMode modes[] = {
        par2::VelocityEvalMode::KhLinear,
        par2::VelocityEvalMode::KhCubicPotentialReconstruction,
        par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction,
    };
    for (const auto mode : modes) {
        run_sampler(grid, view, mode, dev, xs, ys, zs, vx, vy, vz);
        for (size_t n = 0; n < vx.size(); ++n) {
            const double k_expected =
                1.0 + 0.1 * std::cos(gy * ys[n]) + 0.05 * std::sin(gz * zs[n]);
            const double dhdy = gy * std::cos(gy * ys[n]);
            const double dhdz = -0.5 * gz * std::sin(gz * zs[n]);
            expect_close(vx[n], -k_expected * grad_x, 3.0e-2, "smooth periodic vx");
            expect_close(vy[n], -k_expected * dhdy, 3.0e-2, "smooth periodic vy");
            expect_close(vz[n], -k_expected * dhdz, 3.0e-2, "smooth periodic vz");
        }
    }
}

void test_cubic_head_modes() {
    const auto grid = par2::make_grid<double>(14, 12, 10, 1.0, 1.0, 1.0);
    const double K0 = 1.7;

    std::vector<double> K(grid.num_cells(), K0);
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                head[idx(grid, i, j, k)] = manufactured_cubic_head(
                    x_center(grid, i), y_center(grid, j), z_center(grid, k));
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Extrapolate, 0.0, 0.0, false);
    const std::vector<double> xs{3.2, 5.7, 8.4};
    const std::vector<double> ys{2.6, 6.1, 7.3};
    const std::vector<double> zs{2.4, 4.8, 6.2};

    std::vector<double> vx_linear, vy_linear, vz_linear;
    std::vector<double> vx_cubic, vy_cubic, vz_cubic;
    std::vector<double> vx_logk, vy_logk, vz_logk;

    run_sampler(grid, view, par2::VelocityEvalMode::KhLinear, dev, xs, ys, zs, vx_linear, vy_linear,
                vz_linear);
    run_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction, dev, xs, ys, zs,
                vx_cubic, vy_cubic, vz_cubic);
    run_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction, dev, xs, ys,
                zs, vx_logk, vy_logk, vz_logk);

    double linear_err = 0.0;
    double cubic_err = 0.0;
    double logk_err = 0.0;
    for (size_t n = 0; n < xs.size(); ++n) {
        double dhdx = 0.0;
        double dhdy = 0.0;
        double dhdz = 0.0;
        manufactured_cubic_grad(xs[n], ys[n], zs[n], dhdx, dhdy, dhdz);
        const double qx = -K0 * dhdx;
        const double qy = -K0 * dhdy;
        const double qz = -K0 * dhdz;

        const double err_linear =
            std::abs(vx_linear[n] - qx) + std::abs(vy_linear[n] - qy) + std::abs(vz_linear[n] - qz);
        const double err_cubic =
            std::abs(vx_cubic[n] - qx) + std::abs(vy_cubic[n] - qy) + std::abs(vz_cubic[n] - qz);
        const double err_logk =
            std::abs(vx_logk[n] - qx) + std::abs(vy_logk[n] - qy) + std::abs(vz_logk[n] - qz);

        if (err_linear > linear_err)
            linear_err = err_linear;
        if (err_cubic > cubic_err)
            cubic_err = err_cubic;
        if (err_logk > logk_err)
            logk_err = err_logk;
    }

    if (!(cubic_err < 1.0e-9 && logk_err < 1.0e-9 && cubic_err < linear_err &&
          logk_err < linear_err)) {
        std::printf("FAIL cubic manufactured test: linear_err=%.6e cubic_err=%.6e "
                    "logk_err=%.6e\n",
                    linear_err, cubic_err, logk_err);
        throw std::runtime_error("cubic KH modes did not improve manufactured cubic field");
    }
}

void test_k_positivity_modes() {
    const auto grid = par2::make_grid<double>(8, 4, 4, 1.0, 1.0, 1.0);
    const double grad_x = 1.0;

    const double x_profile[8] = {
        2.386093216690147,   1.0760132314343227,  0.048115646257296786, 0.010855929172121817,
        0.11094273944918018, 0.04166165803092431, 1.3651939536141027,   0.01633995681191814,
    };

    std::vector<double> K(grid.num_cells());
    std::vector<double> head(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                K[idx(grid, i, j, k)] = x_profile[i];
                head[idx(grid, i, j, k)] = grad_x * x_center(grid, i);
            }
        }
    }

    DeviceArrays dev;
    upload_field(dev, K, dev.K);
    upload_field(dev, head, dev.head);

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Dirichlet, 0.0,
                                grad_x * grid.length_x(), false);
    const std::vector<double> xs{2.651};
    const std::vector<double> ys{1.5};
    const std::vector<double> zs{1.5};
    std::vector<double> kcubic, logk_dummy;
    std::vector<double> klogk, ylogk;
    std::vector<double> vx, vy, vz;

    run_conductivity_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction,
                             dev, xs, ys, zs, kcubic, logk_dummy);
    run_sampler(grid, view, par2::VelocityEvalMode::KhCubicPotentialReconstruction, dev, xs, ys, zs,
                vx, vy, vz);
    if (!(kcubic[0] < 0.0 && vx[0] > 0.0)) {
        std::printf("FAIL cubic positivity test: kcubic=%.17g vx=%.17g\n", kcubic[0], vx[0]);
        throw std::runtime_error("direct cubic did not expose expected non-positive K overshoot");
    }

    run_conductivity_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction,
                             dev, xs, ys, zs, klogk, ylogk);
    run_sampler(grid, view, par2::VelocityEvalMode::KhLogKCubicPotentialReconstruction, dev, xs, ys,
                zs, vx, vy, vz);
    if (!(std::isfinite(klogk[0]) && klogk[0] > 0.0 && std::isfinite(ylogk[0]) && vx[0] < 0.0)) {
        std::printf("FAIL logK positivity test: klogk=%.17g ylogk=%.17g vx=%.17g\n", klogk[0],
                    ylogk[0], vx[0]);
        throw std::runtime_error("logK cubic did not preserve positive conductivity");
    }
}

void test_order_study(std::vector<OrderErrorRow>& rows) {
    rows.clear();
    for (int n : {16, 32, 64}) {
        rows.push_back(compute_order_error_row(n));
    }

    if (!(rows[0].cubic_rel_l2 < rows[0].linear_rel_l2 &&
          rows[1].cubic_rel_l2 < rows[1].linear_rel_l2 &&
          rows[2].cubic_rel_l2 < rows[2].linear_rel_l2 &&
          rows[0].logk_rel_l2 < rows[0].linear_rel_l2 &&
          rows[1].logk_rel_l2 < rows[1].linear_rel_l2 &&
          rows[2].logk_rel_l2 < rows[2].linear_rel_l2 &&
          rows[2].cubic_rel_l2 < rows[0].cubic_rel_l2 * 0.2 &&
          rows[2].logk_rel_l2 < rows[0].logk_rel_l2 * 0.2)) {
        for (const auto& row : rows) {
            std::printf("order row n=%d linear=%.6e cubic=%.6e logk=%.6e\n", row.n,
                        row.linear_rel_l2, row.cubic_rel_l2, row.logk_rel_l2);
        }
        throw std::runtime_error("manufactured order study did not show expected KH improvement");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string order_csv_path;
        if (argc == 3 && std::string(argv[1]) == "--manufactured-order-csv") {
            order_csv_path = argv[2];
        } else if (argc != 1) {
            throw std::runtime_error(
                "usage: kh_potential_reconstruction_tests [--manufactured-order-csv path]");
        }

        test_linear_head_constant_k();
        test_linear_head_variable_k();
        test_linear_head_smooth_logk();
        test_smooth_periodic_yz();
        test_cubic_head_modes();
        test_k_positivity_modes();

        std::vector<OrderErrorRow> rows;
        test_order_study(rows);
        if (!order_csv_path.empty()) {
            write_order_csv(order_csv_path, rows);
        }

        std::printf("KH potential reconstruction tests passed.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "KH potential reconstruction tests failed: %s\n", e.what());
        return 1;
    }
}
