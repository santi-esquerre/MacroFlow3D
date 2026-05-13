/**
 * @file test_kh_potential_reconstruction.cu
 * @brief Manufactured-field tests for KH potential-flow velocity evaluation.
 */

#include "src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh"

#include <cuda_runtime.h>
#include <par2_core/grid.hpp>
#include <par2_core/views.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
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

__global__ void sample_kh_kernel(const par2::GridDesc<double> grid,
                                 const par2::PotentialFlowView<double> potential,
                                 const double* __restrict__ x, const double* __restrict__ y,
                                 const double* __restrict__ z, double* __restrict__ vx,
                                 double* __restrict__ vy, double* __restrict__ vz, int n) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        par2::internal::sample_velocity_kh_potential(potential, grid, x[tid], y[tid], z[tid],
                                                     vx[tid], vy[tid], vz[tid]);
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

    ~DeviceArrays() {
        cudaFree(K);
        cudaFree(head);
        cudaFree(x);
        cudaFree(y);
        cudaFree(z);
        cudaFree(vx);
        cudaFree(vy);
        cudaFree(vz);
    }
};

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

void run_sampler(const par2::GridDesc<double>& grid, const par2::PotentialFlowView<double>& view,
                 DeviceArrays& dev, const std::vector<double>& hx, const std::vector<double>& hy,
                 const std::vector<double>& hz, std::vector<double>& vx, std::vector<double>& vy,
                 std::vector<double>& vz) {
    const int n = static_cast<int>(hx.size());
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

    sample_kh_kernel<<<1, 128>>>(grid, view, dev.x, dev.y, dev.z, dev.vx, dev.vy, dev.vz, n);
    TEST_CUDA_CHECK(cudaGetLastError());
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    vx.resize(hx.size());
    vy.resize(hx.size());
    vz.resize(hx.size());
    TEST_CUDA_CHECK(
        cudaMemcpy(vx.data(), dev.vx, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(
        cudaMemcpy(vy.data(), dev.vy, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    TEST_CUDA_CHECK(
        cudaMemcpy(vz.data(), dev.vz, hx.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

void expect_close(double got, double expected, double tol, const char* label) {
    if (!std::isfinite(got)) {
        std::printf("FAIL %s: got non-finite %.17g\n", label, got);
        throw std::runtime_error("non-finite KH velocity");
    }
    if (std::abs(got - expected) > tol) {
        std::printf("FAIL %s: got %.17g expected %.17g tol %.3e\n", label, got, expected, tol);
        throw std::runtime_error("KH velocity mismatch");
    }
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
    TEST_CUDA_CHECK(cudaMalloc(&dev.K, K.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.head, head.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMemcpy(dev.K, K.data(), K.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.head, head.data(), head.size() * sizeof(double), cudaMemcpyHostToDevice));

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Dirichlet, h0,
                                h0 + grad_x * grid.length_x(), false);
    std::vector<double> vx, vy, vz;
    run_sampler(grid, view, dev, {0.2, 2.3, 7.8}, {1.7, 3.1, 4.4}, {0.6, 2.2, 3.8}, vx, vy, vz);

    for (size_t n = 0; n < vx.size(); ++n) {
        expect_close(vx[n], -K0 * grad_x, 1e-10, "linear constant K vx");
        expect_close(vy[n], 0.0, 1e-10, "linear constant K vy");
        expect_close(vz[n], 0.0, 1e-10, "linear constant K vz");
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
    TEST_CUDA_CHECK(cudaMalloc(&dev.K, K.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.head, head.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMemcpy(dev.K, K.data(), K.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.head, head.data(), head.size() * sizeof(double), cudaMemcpyHostToDevice));

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Dirichlet, h0,
                                h0 + grad_x * grid.length_x(), false);
    std::vector<double> vx, vy, vz;
    const std::vector<double> xs{2.25, 4.5, 7.25};
    const std::vector<double> ys{2.0, 5.25, 3.75};
    const std::vector<double> zs{1.75, 2.5, 4.25};
    run_sampler(grid, view, dev, xs, ys, zs, vx, vy, vz);

    for (size_t n = 0; n < vx.size(); ++n) {
        const double k_expected = 1.0 + 0.03 * xs[n] + 0.02 * ys[n] + 0.01 * zs[n];
        expect_close(vx[n], -k_expected * grad_x, 1e-10, "linear variable K vx");
        expect_close(vy[n], 0.0, 1e-10, "linear variable K vy");
        expect_close(vz[n], 0.0, 1e-10, "linear variable K vz");
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
    TEST_CUDA_CHECK(cudaMalloc(&dev.K, K.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMalloc(&dev.head, head.size() * sizeof(double)));
    TEST_CUDA_CHECK(cudaMemcpy(dev.K, K.data(), K.size() * sizeof(double), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(
        cudaMemcpy(dev.head, head.data(), head.size() * sizeof(double), cudaMemcpyHostToDevice));

    const auto view = make_view(grid, dev, par2::ScalarBoundaryType::Extrapolate, 0.0, 0.0, true);
    std::vector<double> vx, vy, vz;
    const std::vector<double> xs{12.3, 24.7, 31.5};
    const std::vector<double> ys{0.12, Ly - 0.08, 17.3};
    const std::vector<double> zs{Lz - 0.2, 0.18, 29.6};
    run_sampler(grid, view, dev, xs, ys, zs, vx, vy, vz);

    for (size_t n = 0; n < vx.size(); ++n) {
        const double k_expected = 1.0 + 0.1 * std::cos(gy * ys[n]) + 0.05 * std::sin(gz * zs[n]);
        const double dhdy = gy * std::cos(gy * ys[n]);
        const double dhdz = -0.5 * gz * std::sin(gz * zs[n]);
        expect_close(vx[n], -k_expected * grad_x, 3.0e-2, "smooth periodic vx");
        expect_close(vy[n], -k_expected * dhdy, 3.0e-2, "smooth periodic vy");
        expect_close(vz[n], -k_expected * dhdz, 3.0e-2, "smooth periodic vz");
    }
}

} // namespace

int main() {
    try {
        test_linear_head_constant_k();
        test_linear_head_variable_k();
        test_smooth_periodic_yz();
        std::printf("KH potential reconstruction tests passed.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "KH potential reconstruction tests failed: %s\n", e.what());
        return 1;
    }
}
