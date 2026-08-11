#include "AndersonAccelerator.cuh"

#include "../../numerics/blas/axpy.cuh"
#include "../../numerics/blas/copy.cuh"
#include "../../numerics/blas/dot.cuh"
#include "../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace macroflow3d {
namespace streamfunctions {
namespace {

// Total dot-product entries the Gram/rhs assembly ever issues for a given
// held column count m: the coupled (component-1 + component-2) Gram upper
// triangle (m*(m+1)/2 pairs * 2 entries) plus the coupled rhs (m * 2
// entries) = m^2 + 3*m; bounded by depth <= kMaxAndersonDepth.
std::size_t max_dot_scratch_entries(int depth) {
    return static_cast<std::size_t>(depth) * (static_cast<std::size_t>(depth) + 3);
}

/**
 * Pivoted Householder QR solve of the small m x m SYMMETRIC POSITIVE
 * SEMIDEFINITE Gram system `A gamma = b` (A modified in place; b modified in
 * place to hold Q^T b). Returns false, leaving `gamma` untouched, if the
 * R-diagonal condition estimate (max|R_ii| / min|R_ii|) is non-finite or
 * exceeds `condition_limit`; `condition_estimate` always receives the
 * computed value. See AndersonAccelerator.cuh for the normal-equations
 * conditioning caveat this routine accepts by design.
 */
bool solve_small_ls_pivoted_qr(int m, real a[kMaxAndersonDepth][kMaxAndersonDepth],
                                real b[kMaxAndersonDepth], real condition_limit,
                                real gamma[kMaxAndersonDepth], real& condition_estimate) {
    int perm[kMaxAndersonDepth];
    for (int j = 0; j < m; ++j) {
        perm[j] = j;
    }
    real diag[kMaxAndersonDepth] = {};

    for (int k = 0; k < m; ++k) {
        // Column pivoting: select the remaining column with the largest
        // trailing sub-column norm^2.
        int pivot = k;
        real best = real{-1};
        for (int j = k; j < m; ++j) {
            real s = real{0};
            for (int r = k; r < m; ++r) {
                s += a[r][j] * a[r][j];
            }
            if (s > best) {
                best = s;
                pivot = j;
            }
        }
        if (pivot != k) {
            for (int r = 0; r < m; ++r) {
                std::swap(a[r][k], a[r][pivot]);
            }
            std::swap(perm[k], perm[pivot]);
        }

        // Householder reflection for a[k:m-1][k].
        real norm_x = real{0};
        for (int r = k; r < m; ++r) {
            norm_x += a[r][k] * a[r][k];
        }
        norm_x = std::sqrt(norm_x);
        if (norm_x < std::numeric_limits<real>::epsilon()) {
            diag[k] = real{0};
            continue; // Rank-deficient direction; caught by the condition check below.
        }
        const real alpha = (a[k][k] >= real{0}) ? -norm_x : norm_x;
        real v[kMaxAndersonDepth] = {};
        for (int r = k; r < m; ++r) {
            v[r] = a[r][k];
        }
        v[k] -= alpha;
        real v_norm = real{0};
        for (int r = k; r < m; ++r) {
            v_norm += v[r] * v[r];
        }
        v_norm = std::sqrt(v_norm);
        if (v_norm < std::numeric_limits<real>::epsilon()) {
            diag[k] = a[k][k];
            continue;
        }
        for (int r = k; r < m; ++r) {
            v[r] /= v_norm;
        }

        for (int c = k; c < m; ++c) {
            real dot = real{0};
            for (int r = k; r < m; ++r) {
                dot += v[r] * a[r][c];
            }
            for (int r = k; r < m; ++r) {
                a[r][c] -= real{2} * dot * v[r];
            }
        }
        real dot_b = real{0};
        for (int r = k; r < m; ++r) {
            dot_b += v[r] * b[r];
        }
        for (int r = k; r < m; ++r) {
            b[r] -= real{2} * dot_b * v[r];
        }

        diag[k] = a[k][k];
    }

    real max_abs = real{0};
    real min_abs = std::numeric_limits<real>::infinity();
    for (int k = 0; k < m; ++k) {
        const real value = std::fabs(diag[k]);
        max_abs = std::max(max_abs, value);
        min_abs = std::min(min_abs, value);
    }
    condition_estimate =
        (min_abs > real{0}) ? max_abs / min_abs : std::numeric_limits<real>::infinity();
    if (!std::isfinite(condition_estimate) || condition_estimate > condition_limit) {
        return false;
    }

    // Back substitution: R z = b (R is the upper-triangular part of `a`).
    real z[kMaxAndersonDepth] = {};
    for (int k = m - 1; k >= 0; --k) {
        real sum = b[k];
        for (int c = k + 1; c < m; ++c) {
            sum -= a[k][c] * z[c];
        }
        z[k] = sum / diag[k];
    }
    for (int j = 0; j < m; ++j) {
        gamma[perm[j]] = z[j];
    }
    return true;
}

} // namespace

AndersonMemoryEstimate estimate_anderson_device_bytes(std::size_t n, int depth) {
    AndersonMemoryEstimate estimate;
    estimate.history_bytes = 4 * static_cast<std::size_t>(depth) * n * sizeof(real);
    estimate.staging_bytes = 4 * n * sizeof(real);
    // Mirrors blas::ReductionWorkspace's default-constructed footprint
    // (empty `temp`, one-element `d_scalar`) WITHOUT instantiating one here,
    // so this pure host prediction never allocates device memory (see
    // StreamfunctionWorkspace.cuh's estimator contract).
    estimate.scratch_bytes = max_dot_scratch_entries(depth) * sizeof(real) + sizeof(real);
    return estimate;
}

void AndersonAccelerator::prepare(std::size_t n, int depth) {
    delta_x1_.resize(static_cast<std::size_t>(depth) * n);
    delta_x2_.resize(static_cast<std::size_t>(depth) * n);
    delta_f1_.resize(static_cast<std::size_t>(depth) * n);
    delta_f2_.resize(static_cast<std::size_t>(depth) * n);

    prev_x1_.resize(n);
    prev_x2_.resize(n);
    prev_f1_.resize(n);
    prev_f2_.resize(n);

    dot_scratch_.resize(max_dot_scratch_entries(depth));
    reduction_ws_.ensure_scalar();

    n_ = n;
    depth_ = depth;
    clear();
}

bool AndersonAccelerator::prepared_for(std::size_t n, int depth) const noexcept {
    return n_ == n && depth_ == depth && delta_x1_.size() == static_cast<std::size_t>(depth) * n &&
           delta_x2_.size() == static_cast<std::size_t>(depth) * n &&
           delta_f1_.size() == static_cast<std::size_t>(depth) * n &&
           delta_f2_.size() == static_cast<std::size_t>(depth) * n && prev_x1_.size() == n &&
           prev_x2_.size() == n && prev_f1_.size() == n && prev_f2_.size() == n &&
           dot_scratch_.size() == max_dot_scratch_entries(depth);
}

void AndersonAccelerator::clear() noexcept {
    num_columns_ = 0;
    total_pushes_ = 0;
    has_previous_ = false;
}

DeviceSpan<real> AndersonAccelerator::column(DeviceBuffer<real>& buffer, int slot) noexcept {
    return DeviceSpan<real>(buffer.data() + static_cast<std::size_t>(slot) * n_, n_);
}
DeviceSpan<const real> AndersonAccelerator::column(const DeviceBuffer<real>& buffer,
                                                   int slot) const noexcept {
    return DeviceSpan<const real>(buffer.data() + static_cast<std::size_t>(slot) * n_, n_);
}

void AndersonAccelerator::update_history(CudaContext& context, DeviceSpan<const real> x1,
                                         DeviceSpan<const real> x2, DeviceSpan<const real> u_hat1,
                                         DeviceSpan<const real> u_hat2) {
    if (has_previous_) {
        const int slot = static_cast<int>(total_pushes_ % static_cast<long long>(depth_));
        DeviceSpan<real> col_x1 = column(delta_x1_, slot);
        DeviceSpan<real> col_x2 = column(delta_x2_, slot);
        DeviceSpan<real> col_f1 = column(delta_f1_, slot);
        DeviceSpan<real> col_f2 = column(delta_f2_, slot);

        // DeltaX = x - x_prev (x_prev read here, overwritten further below).
        blas::copy(context, x1, col_x1);
        blas::axpy(context, real{-1}, DeviceSpan<const real>(prev_x1_.span()), col_x1);
        blas::copy(context, x2, col_x2);
        blas::axpy(context, real{-1}, DeviceSpan<const real>(prev_x2_.span()), col_x2);

        // DeltaF = (u_hat - x) - f_prev, formed without a separate "new f"
        // scratch buffer: col <- u_hat; col -= x  (col == new f);
        // col -= f_prev (col == DeltaF, f_prev still holds the OLD value);
        // f_prev += col (f_prev == new f, restaged for the next call).
        blas::copy(context, u_hat1, col_f1);
        blas::axpy(context, real{-1}, x1, col_f1);
        blas::axpy(context, real{-1}, DeviceSpan<const real>(prev_f1_.span()), col_f1);
        blas::axpy(context, real{1}, DeviceSpan<const real>(col_f1), prev_f1_.span());

        blas::copy(context, u_hat2, col_f2);
        blas::axpy(context, real{-1}, x2, col_f2);
        blas::axpy(context, real{-1}, DeviceSpan<const real>(prev_f2_.span()), col_f2);
        blas::axpy(context, real{1}, DeviceSpan<const real>(col_f2), prev_f2_.span());

        // x_prev <- x (independent plain overwrite).
        blas::copy(context, x1, prev_x1_.span());
        blas::copy(context, x2, prev_x2_.span());

        ++total_pushes_;
        num_columns_ = static_cast<int>(std::min<long long>(total_pushes_, depth_));
    } else {
        // Bootstrap: no previous pair yet, so no delta column is formed;
        // just stage (x, f = u_hat - x) for the NEXT call.
        blas::copy(context, x1, prev_x1_.span());
        blas::copy(context, x2, prev_x2_.span());
        blas::copy(context, u_hat1, prev_f1_.span());
        blas::axpy(context, real{-1}, x1, prev_f1_.span());
        blas::copy(context, u_hat2, prev_f2_.span());
        blas::axpy(context, real{-1}, x2, prev_f2_.span());
        has_previous_ = true;
    }
}

bool AndersonAccelerator::form_candidate(CudaContext& context, DeviceSpan<const real> u_hat1,
                                         DeviceSpan<const real> u_hat2, real condition_limit,
                                         DeviceSpan<real> out1, DeviceSpan<real> out2,
                                         real& condition_estimate) {
    condition_estimate = std::numeric_limits<real>::infinity();
    const int m = num_columns_;
    if (m <= 0) {
        return false;
    }

    // Fixed-shape, deterministic GPU dot products (cuBLAS Ddot on a fixed
    // problem size is a deterministic tree reduction; no atomics, no
    // run-to-run reduction-order nondeterminism): fill the coupled Gram
    // upper triangle and the coupled rhs into distinct slots of one small
    // device buffer with NO intermediate host synchronization, then transfer
    // the whole buffer once.
    int offset = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            blas::dot_device(context, column(delta_f1_, i), column(delta_f1_, j),
                             DeviceSpan<real>(dot_scratch_.data() + offset, 1), reduction_ws_);
            ++offset;
            blas::dot_device(context, column(delta_f2_, i), column(delta_f2_, j),
                             DeviceSpan<real>(dot_scratch_.data() + offset, 1), reduction_ws_);
            ++offset;
        }
    }
    for (int i = 0; i < m; ++i) {
        blas::dot_device(context, column(delta_f1_, i), DeviceSpan<const real>(prev_f1_.span()),
                         DeviceSpan<real>(dot_scratch_.data() + offset, 1), reduction_ws_);
        ++offset;
        blas::dot_device(context, column(delta_f2_, i), DeviceSpan<const real>(prev_f2_.span()),
                         DeviceSpan<real>(dot_scratch_.data() + offset, 1), reduction_ws_);
        ++offset;
    }

    real host_buf[kMaxAndersonDepth * (kMaxAndersonDepth + 3)];
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_buf, dot_scratch_.data(),
                                           static_cast<std::size_t>(offset) * sizeof(real),
                                           cudaMemcpyDeviceToHost, context.cuda_stream()));
    context.synchronize();

    real gram[kMaxAndersonDepth][kMaxAndersonDepth] = {};
    real rhs[kMaxAndersonDepth] = {};
    int idx = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            const real value = host_buf[idx] + host_buf[idx + 1];
            idx += 2;
            gram[i][j] = value;
            gram[j][i] = value;
        }
    }
    for (int i = 0; i < m; ++i) {
        rhs[i] = host_buf[idx] + host_buf[idx + 1];
        idx += 2;
    }

    real gamma[kMaxAndersonDepth] = {};
    if (!solve_small_ls_pivoted_qr(m, gram, rhs, condition_limit, gamma, condition_estimate)) {
        return false;
    }

    // x_acc = u_hat - sum_j gamma_j * (DeltaX_j + DeltaF_j); see the file
    // header for the algebraic equivalence to x_k + f_k - (X_k+F_k)*gamma.
    blas::copy(context, u_hat1, out1);
    blas::copy(context, u_hat2, out2);
    for (int j = 0; j < m; ++j) {
        blas::axpy(context, -gamma[j], column(delta_x1_, j), out1);
        blas::axpy(context, -gamma[j], column(delta_f1_, j), out1);
        blas::axpy(context, -gamma[j], column(delta_x2_, j), out2);
        blas::axpy(context, -gamma[j], column(delta_f2_, j), out2);
    }
    return true;
}

std::size_t AndersonAccelerator::history_bytes() const noexcept {
    return (delta_x1_.capacity() + delta_x2_.capacity() + delta_f1_.capacity() +
            delta_f2_.capacity()) *
           sizeof(real);
}
std::size_t AndersonAccelerator::staging_bytes() const noexcept {
    return (prev_x1_.capacity() + prev_x2_.capacity() + prev_f1_.capacity() +
            prev_f2_.capacity()) *
           sizeof(real);
}
std::size_t AndersonAccelerator::scratch_bytes() const noexcept {
    return dot_scratch_.capacity() * sizeof(real) +
           blas::reduction_workspace_allocated_bytes(reduction_ws_);
}

} // namespace streamfunctions
} // namespace macroflow3d
