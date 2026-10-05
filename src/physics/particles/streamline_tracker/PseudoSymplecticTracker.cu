/**
 * @file PseudoSymplecticTracker.cu
 * @brief SF-31 pseudo-symplectic streamline tracker - GPU engine.
 *
 * See PseudoSymplecticTracker.cuh for the scheme, the time semantics, the
 * failure policy and the status codes. The kernels instantiate the very same
 * __host__ __device__ integrator core on SplineLabelPair.
 */

#include "../../../runtime/cuda_check.cuh"
#include "PseudoSymplecticTracker.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

namespace {

constexpr int kBlockSize = 128;

int grid_size(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

/// Sample the label targets at the current positions; zero clocks and counters.
__global__ void kernel_prepare(SplineLabelPair labels, const real* __restrict__ px,
                               const real* __restrict__ py, const real* __restrict__ pz,
                               const int32_t* __restrict__ wx, const int32_t* __restrict__ wy,
                               const int32_t* __restrict__ wz, int n, real* __restrict__ psi1_0,
                               real* __restrict__ psi2_0, real* __restrict__ clock,
                               uint32_t* __restrict__ fail, uint32_t* __restrict__ clamp,
                               uint32_t* __restrict__ nmax) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= n)
        return;
    const real xi[3] = {px[i], py[i], pz[i]};
    const int32_t w[3] = {wx[i], wy[i], wz[i]};
    LabelSample s;
    labels(xi, w, s);
    psi1_0[i] = s.psi1;
    psi2_0[i] = s.psi2;
    clock[i] = static_cast<real>(0.0);
    fail[i] = 0u;
    clamp[i] = 0u;
    nmax[i] = 0u;
}

/// Per-call particle update. kArclength: one panel of length `a`;
/// otherwise advance_to_time with t_target = `a`.
template <bool kArclength>
__global__ void kernel_advance(SplineLabelPair labels, PseudoSymplecticParams prm,
                               real* __restrict__ px, real* __restrict__ py,
                               real* __restrict__ pz, int32_t* __restrict__ wx,
                               int32_t* __restrict__ wy, int32_t* __restrict__ wz,
                               uint8_t* __restrict__ status, int n,
                               const real* __restrict__ psi1_0, const real* __restrict__ psi2_0,
                               real* __restrict__ clock, uint32_t* __restrict__ fail,
                               uint32_t* __restrict__ clamp, uint32_t* __restrict__ nmax, real a,
                               real ds_max, int max_panels) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= n)
        return;
    if (status[i] != kStatusActive)
        return;

    PanelState st;
    st.xi[0] = px[i];
    st.xi[1] = py[i];
    st.xi[2] = pz[i];
    st.w[0] = wx[i];
    st.w[1] = wy[i];
    st.w[2] = wz[i];
    st.t = clock[i];
    const real t1 = psi1_0[i];
    const real t2 = psi2_0[i];

    ProjectionCounters cnt{0u, 0u};
    uint8_t code = evaluate_label_state(labels, st.xi, st.w, prm, st.at);
    if (code == kStatusActive) {
        if (kArclength) {
            code = advance_panel(labels, st, t1, t2, a, prm, cnt);
        } else {
            code = advance_to_time(labels, st, t1, t2, a, ds_max, max_panels, prm, cnt);
        }
    }

    // Last accepted state (a failed panel leaves st untouched).
    px[i] = st.xi[0];
    py[i] = st.xi[1];
    pz[i] = st.xi[2];
    wx[i] = st.w[0];
    wy[i] = st.w[1];
    wz[i] = st.w[2];
    clock[i] = st.t;
    if (code != kStatusActive) {
        status[i] = code;
        fail[i] += 1u;
    }
    if (cnt.newton_iter_max > nmax[i])
        nmax[i] = cnt.newton_iter_max;
    clamp[i] += cnt.clamp_count;
}

void require_wraps(const ParticlesSoA<real>& p, const char* who) {
    if (p.wrapX == nullptr || p.wrapY == nullptr || p.wrapZ == nullptr) {
        throw std::invalid_argument(std::string(who) +
                                    ": a wrap-counter pointer (wrapX, wrapY or wrapZ) is null "
                                    "(wrap arrays are required: triply periodic tracker)");
    }
}

} // namespace

// ============================================================================
// Construction and configuration
// ============================================================================

PseudoSymplecticTracker::PseudoSymplecticTracker(cudaStream_t stream, uint64_t inject_seed)
    : stream_(stream), inject_seed_(inject_seed) {}

void PseudoSymplecticTracker::configure(const PseudoSymplecticConfig& cfg) {
    if (!std::isfinite(cfg.ds_max) || !(cfg.ds_max > 0.0)) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: ds_max must be finite and > 0");
    }
    if (!std::isfinite(cfg.tol_psi) || !(cfg.tol_psi > 0.0)) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: tol_psi must be finite and > 0");
    }
    if (!std::isfinite(cfg.trust_factor) || !(cfg.trust_factor > 0.0)) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: trust_factor must be finite and > 0");
    }
    if (cfg.max_newton_iter < 1) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: max_newton_iter must be >= 1");
    }
    if (!std::isfinite(cfg.min_cross_norm) || cfg.min_cross_norm < 0.0) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: min_cross_norm must be finite and >= 0");
    }
    if (!std::isfinite(cfg.min_cross_sin2) || cfg.min_cross_sin2 < 0.0) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: min_cross_sin2 must be finite and >= 0");
    }
    if (cfg.max_panels_per_step < 1) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::configure: max_panels_per_step must be >= 1");
    }
    cfg_ = cfg;
    configured_ = true;
}

void PseudoSymplecticTracker::bind_labels(const SplineLabelPair& labels) {
    if (labels.s1.coeff == nullptr || labels.s2.coeff == nullptr) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::bind_labels: a spline coefficient pointer is null");
    }
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(labels.L[d]) || !(labels.L[d] > 0.0)) {
            throw std::invalid_argument(
                "PseudoSymplecticTracker::bind_labels: a period L[d] is not finite or not > 0");
        }
    }
    labels_ = labels;
    labels_bound_ = true;
    prepared_ = false;
}

void PseudoSymplecticTracker::bind_particles(ParticlesSoA<real>& p) {
    if (p.x == nullptr || p.y == nullptr || p.z == nullptr) {
        throw std::invalid_argument("PseudoSymplecticTracker::bind_particles: a position "
                                    "pointer (x, y or z) is null");
    }
    if (p.status == nullptr) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::bind_particles: status pointer is null");
    }
    if (p.n < 0) {
        throw std::invalid_argument("PseudoSymplecticTracker::bind_particles: p.n < 0");
    }
    p_ = p;
    particles_bound_ = true;
    prepared_ = false;
}

void PseudoSymplecticTracker::inject_box(real x0, real y0, real z0, real x1, real y1, real z1,
                                         int first, int count) {
    if (!particles_bound_) {
        throw std::logic_error(
            "PseudoSymplecticTracker::inject_box: called before bind_particles");
    }
    streamline_tracker::inject_box(stream_, p_, x0, y0, z0, x1, y1, z1, first, count,
                                   inject_seed_);
}

void PseudoSymplecticTracker::ensure_tracking() {
    if (!particles_bound_) {
        throw std::logic_error(
            "PseudoSymplecticTracker::ensure_tracking: called before bind_particles");
    }
    require_wraps(p_, "PseudoSymplecticTracker::ensure_tracking");
}

// ============================================================================
// prepare
// ============================================================================

void PseudoSymplecticTracker::prepare() {
    if (!configured_) {
        throw std::logic_error("PseudoSymplecticTracker::prepare: called before configure");
    }
    if (!labels_bound_) {
        throw std::logic_error("PseudoSymplecticTracker::prepare: called before bind_labels");
    }
    if (!particles_bound_) {
        throw std::logic_error(
            "PseudoSymplecticTracker::prepare: called before bind_particles");
    }
    require_wraps(p_, "PseudoSymplecticTracker::prepare");

    const size_t n = static_cast<size_t>(p_.n);
    psi1_0_.resize(n);
    psi2_0_.resize(n);
    clock_.resize(n);
    fail_count_.resize(n);
    clamp_count_.resize(n);
    newton_iter_max_.resize(n);

    if (p_.n > 0) {
        kernel_prepare<<<grid_size(p_.n), kBlockSize, 0, stream_>>>(
            labels_, p_.x, p_.y, p_.z, p_.wrapX, p_.wrapY, p_.wrapZ, p_.n, psi1_0_.data(),
            psi2_0_.data(), clock_.data(), fail_count_.data(), clamp_count_.data(),
            newton_iter_max_.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    t_target_ = 0.0;
    prepared_n_ = p_.n;
    prepared_ = true;
}

void PseudoSymplecticTracker::require_prepared(const char* who) const {
    if (!prepared_) {
        throw std::logic_error(std::string(who) + ": called before prepare");
    }
}

PseudoSymplecticParams PseudoSymplecticTracker::params() const {
    PseudoSymplecticParams prm{};
    prm.tol_psi = cfg_.tol_psi;
    prm.max_newton_iter = cfg_.max_newton_iter;
    prm.trust_factor = cfg_.trust_factor;
    prm.min_cross_norm = cfg_.min_cross_norm;
    prm.min_cross_sin2 = cfg_.min_cross_sin2;
    return prm;
}

// ============================================================================
// Hot path: kernel launches only (no allocation, no synchronization)
// ============================================================================

void PseudoSymplecticTracker::step(real dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::step: dt must be finite and >= 0");
    }
    require_prepared("PseudoSymplecticTracker::step");
    t_target_ += dt;
    if (prepared_n_ == 0)
        return;
    kernel_advance<false><<<grid_size(prepared_n_), kBlockSize, 0, stream_>>>(
        labels_, params(), p_.x, p_.y, p_.z, p_.wrapX, p_.wrapY, p_.wrapZ, p_.status,
        prepared_n_, psi1_0_.data(), psi2_0_.data(), clock_.data(), fail_count_.data(),
        clamp_count_.data(), newton_iter_max_.data(), t_target_, cfg_.ds_max,
        cfg_.max_panels_per_step);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void PseudoSymplecticTracker::step_arclength(real ds) {
    if (!std::isfinite(ds) || !(ds > 0.0)) {
        throw std::invalid_argument(
            "PseudoSymplecticTracker::step_arclength: ds must be finite and > 0");
    }
    require_prepared("PseudoSymplecticTracker::step_arclength");
    if (prepared_n_ == 0)
        return;
    kernel_advance<true><<<grid_size(prepared_n_), kBlockSize, 0, stream_>>>(
        labels_, params(), p_.x, p_.y, p_.z, p_.wrapX, p_.wrapY, p_.wrapZ, p_.status,
        prepared_n_, psi1_0_.data(), psi2_0_.data(), clock_.data(), fail_count_.data(),
        clamp_count_.data(), newton_iter_max_.data(), ds, cfg_.ds_max,
        cfg_.max_panels_per_step);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Contract accessors
// ============================================================================

void PseudoSymplecticTracker::synchronize() {
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ConstParticlesSoA<real> PseudoSymplecticTracker::particles() const {
    ConstParticlesSoA<real> c;
    c.x = p_.x;
    c.y = p_.y;
    c.z = p_.z;
    c.n = p_.n;
    c.status = p_.status;
    c.wrapX = p_.wrapX;
    c.wrapY = p_.wrapY;
    c.wrapZ = p_.wrapZ;
    return c;
}

void PseudoSymplecticTracker::compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream) {
    if (!labels_bound_) {
        throw std::logic_error(
            "PseudoSymplecticTracker::compute_unwrapped: called before bind_labels");
    }
    if (!particles_bound_) {
        throw std::logic_error(
            "PseudoSymplecticTracker::compute_unwrapped: called before bind_particles");
    }
    streamline_tracker::compute_unwrapped(stream, particles(), labels_.L, uw);
}

// ============================================================================
// Stats (host; synchronizes; not for the hot loop)
// ============================================================================

PseudoSymplecticStats PseudoSymplecticTracker::compute_stats() {
    require_prepared("PseudoSymplecticTracker::compute_stats");
    const int n = prepared_n_;
    PseudoSymplecticStats out{};
    out.n_particles = n;
    if (n == 0) {
        return out;
    }
    const size_t nn = static_cast<size_t>(n);
    std::vector<uint8_t> status(nn);
    std::vector<uint32_t> fail(nn), clamp(nn), nmax(nn);
    std::vector<real> clock(nn);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(status.data(), p_.status, nn * sizeof(uint8_t),
                                           cudaMemcpyDeviceToHost, stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(fail.data(), fail_count_.data(),
                                           nn * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                           stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(clamp.data(), clamp_count_.data(),
                                           nn * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                           stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(nmax.data(), newton_iter_max_.data(),
                                           nn * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                           stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(clock.data(), clock_.data(), nn * sizeof(real),
                                           cudaMemcpyDeviceToHost, stream_));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));

    out.min_clock = clock[0];
    out.max_clock = clock[0];
    for (size_t i = 0; i < nn; ++i) {
        switch (status[i]) {
        case kStatusActive:
            ++out.n_active;
            break;
        case kStatusNewtonFailed:
            ++out.n_newton_failed;
            break;
        case kStatusDegenerate:
            ++out.n_degenerate;
            break;
        case kStatusSubstepLimit:
            ++out.n_substep_limit;
            break;
        case kStatusNonFinite:
            ++out.n_nonfinite;
            break;
        default:
            ++out.n_other;
            break;
        }
        out.total_fail += fail[i];
        if (fail[i] > out.max_fail)
            out.max_fail = fail[i];
        if (nmax[i] > out.max_newton_iter)
            out.max_newton_iter = nmax[i];
        out.total_clamps += clamp[i];
        if (clock[i] < out.min_clock)
            out.min_clock = clock[i];
        if (clock[i] > out.max_clock)
            out.max_clock = clock[i];
    }
    return out;
}

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
