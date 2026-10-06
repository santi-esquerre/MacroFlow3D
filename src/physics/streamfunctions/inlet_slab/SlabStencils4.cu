/**
 * @file SlabStencils4.cu
 * @brief SF-33 N0: host Fornberg weights and the device x1 stencil table of the inlet slab (see
 * SlabStencils4.cuh).
 */

#include "SlabStencils4.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

std::vector<std::vector<real>> fd_weights(const std::vector<real>& z, real x0, int m) {
    // Line-by-line port of metrics.fd_weights (Fornberg 1988), same operation order.
    const int n = static_cast<int>(z.size());
    if (n < 1 || m < 0)
        throw std::invalid_argument("fd_weights: need at least one node and m >= 0");
    std::vector<std::vector<real>> c(static_cast<std::size_t>(m + 1),
                                     std::vector<real>(z.size(), 0.0));
    real c1 = 1.0;
    real c4 = z[0] - x0;
    c[0][0] = 1.0;
    for (int i = 1; i < n; ++i) {
        const int mn = std::min(i, m);
        real c2 = 1.0;
        const real c5 = c4;
        c4 = z[i] - x0;
        for (int j = 0; j < i; ++j) {
            const real c3 = z[i] - z[j];
            c2 = c2 * c3;
            if (j == i - 1) {
                for (int k = mn; k > 0; --k) {
                    c[k][i] = c1 * (k * c[k - 1][i - 1] - c5 * c[k][i - 1]) / c2;
                }
                c[0][i] = -c1 * c5 * c[0][i - 1] / c2;
            }
            for (int k = mn; k > 0; --k) {
                c[k][j] = (c4 * c[k][j] - k * c[k - 1][j]) / c3;
            }
            c[0][j] = c4 * c[0][j] / c3;
        }
        c1 = c2;
    }
    return c;
}

namespace {

X1Stencil fornberg_row(int first, int npts, int plane, int deriv) {
    // metrics.x1_stencils4: nodes 0..npts-1 relative to `first`, evaluation point plane - first.
    std::vector<real> z(static_cast<std::size_t>(npts));
    for (int k = 0; k < npts; ++k)
        z[static_cast<std::size_t>(k)] = static_cast<real>(k);
    const auto c = fd_weights(z, static_cast<real>(plane - first), deriv);
    X1Stencil s;
    s.first = first;
    s.npts = npts;
    for (int k = 0; k < npts; ++k)
        s.w[k] = c[static_cast<std::size_t>(deriv)][static_cast<std::size_t>(k)];
    return s;
}

X1Stencil centered_row(int plane, int deriv) {
    // metrics.C4_D1 / C4_D2: np.array([...]) / 12.0
    static const real d1[5] = {1.0, -8.0, 0.0, 8.0, -1.0};
    static const real d2[5] = {-1.0, 16.0, -30.0, 16.0, -1.0};
    X1Stencil s;
    s.first = plane - 2;
    s.npts = 5;
    for (int k = 0; k < 5; ++k)
        s.w[k] = (deriv == 1 ? d1[k] : d2[k]) / 12.0;
    return s;
}

} // namespace

X1StencilTablesHost build_x1_stencils4(int n) {
    if (n < 8 || (n % 2) != 0) {
        throw std::invalid_argument("build_x1_stencils4: N must be even and >= 8 (got " +
                                    std::to_string(n) + ")");
    }
    X1StencilTablesHost t;
    t.d1.resize(static_cast<std::size_t>(n + 1));
    t.d11.resize(static_cast<std::size_t>(n + 1));
    for (int deriv = 1; deriv <= 2; ++deriv) {
        auto& tab = (deriv == 1) ? t.d1 : t.d11;
        const int npts = (deriv == 1) ? 5 : 6;
        for (int p = 0; p <= n; ++p) {
            if (p == 0 || p == 1) {
                tab[static_cast<std::size_t>(p)] = fornberg_row(0, npts, p, deriv);
            } else if (p == n - 1 || p == n) {
                tab[static_cast<std::size_t>(p)] = fornberg_row(n - npts + 1, npts, p, deriv);
            } else {
                tab[static_cast<std::size_t>(p)] = centered_row(p, deriv);
            }
        }
    }
    return t;
}

void SlabStencilTable::build(const InletSlabGrid& g) {
    require_valid_grid(g, "SlabStencilTable::build");
    host_ = build_x1_stencils4(g.n);
    const std::size_t np = static_cast<std::size_t>(g.n + 1);
    dev_.resize(2 * np);
    std::vector<X1Stencil> packed;
    packed.reserve(2 * np);
    packed.insert(packed.end(), host_.d1.begin(), host_.d1.end());
    packed.insert(packed.end(), host_.d11.begin(), host_.d11.end());
    // Preparation step: synchronous upload (not part of any hot loop).
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dev_.data(), packed.data(), packed.size() * sizeof(X1Stencil),
                                      cudaMemcpyHostToDevice));
    n_ = g.n;
}

SlabStencilView SlabStencilTable::device_view() const {
    if (n_ <= 0)
        throw std::logic_error("SlabStencilTable: not built");
    SlabStencilView v;
    v.d1 = dev_.data();
    v.d11 = dev_.data() + (n_ + 1);
    return v;
}

SlabStencilView SlabStencilTable::host_view() const {
    if (n_ <= 0)
        throw std::logic_error("SlabStencilTable: not built");
    SlabStencilView v;
    v.d1 = host_.d1.data();
    v.d11 = host_.d11.data();
    return v;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
