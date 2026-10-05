/**
 * @file ReferenceRkTracker.cu
 * @brief SF-31 adaptive RK reference tracker - GPU engine implementation.
 *
 * See ReferenceRkTracker.cuh for the full specification (ODE, Dormand-Prince
 * 5(4) pair and controller, exact landing, status codes, splined-label caveat).
 */

#include "../../../runtime/cuda_check.cuh"
#include "ReferenceRkTracker.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

namespace {

constexpr int kBlockSize = 256;

int grid_size(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

/// One thread per particle: advance every active particle to t_target.
__global__ void kernel_rk_advance(real* __restrict__ px, real* __restrict__ py,
                                  real* __restrict__ pz, uint8_t* __restrict__ status,
                                  int32_t* __restrict__ wx, int32_t* __restrict__ wy,
                                  int32_t* __restrict__ wz, real* __restrict__ clock,
                                  real* __restrict__ hprop, uint32_t* __restrict__ acc,
                                  uint32_t* __restrict__ rej, int n, real t_target,
                                  const SplineLabelPair labels, const ReferenceRkParams prm) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= n)
        return;
    if (status[i] != kStatusActive)
        return;

    RkState st;
    st.xi[0] = px[i];
    st.xi[1] = py[i];
    st.xi[2] = pz[i];
    st.w[0] = wx[i];
    st.w[1] = wy[i];
    st.w[2] = wz[i];
    st.t = clock[i];
    st.h = hprop[i];

    RkCounters cnt{0u, 0u};
    const LabelVelocity<SplineLabelPair> vel{labels};
    const uint8_t code = rk_advance_to_time(vel, st, labels.L, t_target, prm, cnt);
    if (code != kStatusActive) {
        status[i] = code;
    }

    px[i] = st.xi[0];
    py[i] = st.xi[1];
    pz[i] = st.xi[2];
    wx[i] = st.w[0];
    wy[i] = st.w[1];
    wz[i] = st.w[2];
    clock[i] = st.t;
    hprop[i] = st.h;
    acc[i] += cnt.accepted;
    rej[i] += cnt.rejected;
}

__global__ void kernel_fill_real(real* __restrict__ a, int n, real value) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= n)
        return;
    a[i] = value;
}

} // namespace

ReferenceRkTracker::ReferenceRkTracker(cudaStream_t stream, uint64_t inject_seed)
    : stream_(stream), inject_seed_(inject_seed) {}

void ReferenceRkTracker::configure(const ReferenceRkConfig& cfg) {
    if (!std::isfinite(cfg.tol) || !(cfg.tol > 0.0)) {
        throw std::invalid_argument("ReferenceRkTracker::configure: tol must be finite and > 0");
    }
    if (!std::isfinite(cfg.dt_max) || !(cfg.dt_max > 0.0)) {
        throw std::invalid_argument(
            "ReferenceRkTracker::configure: dt_max must be finite and > 0");
    }
    if (!std::isfinite(cfg.min_step) || cfg.min_step < 0.0) {
        throw std::invalid_argument(
            "ReferenceRkTracker::configure: min_step must be finite and >= 0");
    }
    if (cfg.max_steps_per_call < 1) {
        throw std::invalid_argument(
            "ReferenceRkTracker::configure: max_steps_per_call must be >= 1");
    }
    cfg_ = cfg;
    configured_ = true;
    tracking_ready_ = false;
    prepared_ = false;
}

void ReferenceRkTracker::bind_labels(const SplineLabelPair& labels) {
    labels_ = labels;
    labels_bound_ = true;
    tracking_ready_ = false;
    prepared_ = false;
}

void ReferenceRkTracker::bind_particles(ParticlesSoA<real>& p) {
    if (p.n < 0) {
        throw std::invalid_argument("ReferenceRkTracker::bind_particles: p.n < 0");
    }
    if (p.x == nullptr || p.y == nullptr || p.z == nullptr) {
        throw std::invalid_argument(
            "ReferenceRkTracker::bind_particles: a position pointer (x, y or z) is null");
    }
    if (p.status == nullptr) {
        throw std::invalid_argument("ReferenceRkTracker::bind_particles: status pointer is null");
    }
    if (p.wrapX == nullptr || p.wrapY == nullptr || p.wrapZ == nullptr) {
        throw std::invalid_argument(
            "ReferenceRkTracker::bind_particles: a wrap-counter pointer (wrapX, wrapY or "
            "wrapZ) is null (wrap arrays are required)");
    }
    parts_ = p;
    particles_bound_ = true;
    tracking_ready_ = false;
    prepared_ = false;
}

void ReferenceRkTracker::inject_box(real x0, real y0, real z0, real x1, real y1, real z1,
                                    int first, int count) {
    if (!particles_bound_) {
        throw std::logic_error(
            "ReferenceRkTracker::inject_box called before bind_particles");
    }
    streamline_tracker::inject_box(stream_, parts_, x0, y0, z0, x1, y1, z1, first, count,
                                   inject_seed_);
}

void ReferenceRkTracker::ensure_tracking() {
    if (!configured_) {
        throw std::logic_error("ReferenceRkTracker::ensure_tracking called before configure");
    }
    if (!labels_bound_) {
        throw std::logic_error("ReferenceRkTracker::ensure_tracking called before bind_labels");
    }
    if (!particles_bound_) {
        throw std::logic_error(
            "ReferenceRkTracker::ensure_tracking called before bind_particles");
    }
    tracking_ready_ = true;
}

void ReferenceRkTracker::prepare() {
    if (!tracking_ready_) {
        throw std::logic_error("ReferenceRkTracker::prepare called before ensure_tracking");
    }
    const int n = parts_.n;
    const size_t ns = static_cast<size_t>(n);
    clock_.resize(ns);
    h_.resize(ns);
    accepted_.resize(ns);
    rejected_.resize(ns);
    if (n > 0) {
        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(clock_.data(), 0, ns * sizeof(real), stream_));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(accepted_.data(), 0, ns * sizeof(uint32_t), stream_));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(rejected_.data(), 0, ns * sizeof(uint32_t), stream_));
        kernel_fill_real<<<grid_size(n), kBlockSize, 0, stream_>>>(
            h_.data(), n, static_cast<real>(0.1) * cfg_.dt_max);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    t_target_ = 0.0;
    prepared_ = true;
}

void ReferenceRkTracker::step(real dt) {
    // Hot path: validation, one kernel launch, no allocation, no synchronization.
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("ReferenceRkTracker::step: dt must be finite and >= 0");
    }
    if (!prepared_) {
        throw std::logic_error("ReferenceRkTracker::step called before prepare");
    }
    t_target_ += dt;
    const int n = parts_.n;
    if (n == 0)
        return;
    const ReferenceRkParams prm{cfg_.tol, cfg_.dt_max, cfg_.min_step, cfg_.max_steps_per_call};
    kernel_rk_advance<<<grid_size(n), kBlockSize, 0, stream_>>>(
        parts_.x, parts_.y, parts_.z, parts_.status, parts_.wrapX, parts_.wrapY, parts_.wrapZ,
        clock_.data(), h_.data(), accepted_.data(), rejected_.data(), n, t_target_, labels_, prm);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void ReferenceRkTracker::synchronize() {
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ConstParticlesSoA<real> ReferenceRkTracker::particles() const {
    ConstParticlesSoA<real> c;
    c.x = parts_.x;
    c.y = parts_.y;
    c.z = parts_.z;
    c.n = parts_.n;
    c.status = parts_.status;
    c.wrapX = parts_.wrapX;
    c.wrapY = parts_.wrapY;
    c.wrapZ = parts_.wrapZ;
    return c;
}

void ReferenceRkTracker::compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream) {
    if (!particles_bound_) {
        throw std::logic_error(
            "ReferenceRkTracker::compute_unwrapped called before bind_particles");
    }
    if (!labels_bound_) {
        throw std::logic_error("ReferenceRkTracker::compute_unwrapped called before bind_labels");
    }
    streamline_tracker::compute_unwrapped(stream, particles(), labels_.L, uw);
}

ReferenceRkStats ReferenceRkTracker::compute_stats() {
    if (!prepared_) {
        throw std::logic_error("ReferenceRkTracker::compute_stats called before prepare");
    }
    synchronize();
    ReferenceRkStats s{};
    const int n = parts_.n;
    s.n_particles = n;
    if (n == 0)
        return s;
    const size_t ns = static_cast<size_t>(n);
    std::vector<uint8_t> status(ns);
    std::vector<real> clock(ns), h(ns);
    std::vector<uint32_t> acc(ns), rej(ns);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(status.data(), parts_.status, ns * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(clock.data(), clock_.data(), ns * sizeof(real), cudaMemcpyDeviceToHost));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), h_.data(), ns * sizeof(real), cudaMemcpyDeviceToHost));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(acc.data(), accepted_.data(), ns * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(rej.data(), rejected_.data(), ns * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    s.min_clock = clock[0];
    s.max_clock = clock[0];
    s.min_step_proposal = h[0];
    s.max_step_proposal = h[0];
    for (size_t i = 0; i < ns; ++i) {
        switch (status[i]) {
        case kStatusActive: ++s.n_active; break;
        case kStatusStepUnderflow: ++s.n_step_underflow; break;
        case kStatusSubstepLimit: ++s.n_substep_limit; break;
        case kStatusNonFinite: ++s.n_nonfinite; break;
        default: ++s.n_other; break;
        }
        s.total_accepted += acc[i];
        s.total_rejected += rej[i];
        if (acc[i] > s.max_accepted)
            s.max_accepted = acc[i];
        if (rej[i] > s.max_rejected)
            s.max_rejected = rej[i];
        if (clock[i] < s.min_clock)
            s.min_clock = clock[i];
        if (clock[i] > s.max_clock)
            s.max_clock = clock[i];
        if (h[i] < s.min_step_proposal)
            s.min_step_proposal = h[i];
        if (h[i] > s.max_step_proposal)
            s.max_step_proposal = h[i];
    }
    return s;
}

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
