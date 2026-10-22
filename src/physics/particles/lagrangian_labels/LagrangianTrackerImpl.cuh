#pragma once

/**
 * @file LagrangianTrackerImpl.cuh
 * @brief SF-36 N4: header-only launcher `LagrangianTrackerT<Field>` for any device field
 *        implementing the eval / eval_ext contract of LagrangianTrackerDevice.cuh (the cube spline
 *        field and the analytic test fields share it).
 *
 * Include only from translation units compiled with --fmad=false (see LagrangianTrackerDevice.cuh).
 * prepare() is the only allocating call; track() performs two kernel launches (plane pass, time
 * pass), device-to-host copies into pre-sized host vectors and the host reductions
 * (finalize_track_result, index order).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/DeviceSpan.cuh"
#include "../../../runtime/CudaContext.cuh"
#include "../../../runtime/cuda_check.cuh"
#include "../../streamfunctions/inlet_slab/InletLabels.cuh"
#include "LagrangianTrackerDevice.cuh"
#include "LagrangianTrackerTypes.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

template <class Field> class LagrangianTrackerT {
  public:
    /// `labels` (nullable, must be device-prepared) supplies the D-1 labels at the seeds when
    /// inj.X0 == 0; otherwise the labels are NaN.
    void prepare(CudaContext& ctx, const Field& field, const InjectionSpec& inj,
                 const TrackOptions& opt, const streamfunctions::inlet_slab::InletLabels* labels) {
        validate_track_setup(inj, opt);
        field_ = field;
        block_ = opt.block;
        covectors_ = opt.covectors;
        single_launch_ = opt.single_launch;

        LagrangianTrackResult& r = result_;
        r = LagrangianTrackResult{};
        r.covectors = opt.covectors;
        r.single_launch = opt.single_launch;
        r.tol = opt.tol;
        r.h_max = opt.h_max;
        r.min_step = opt.min_step;
        r.max_arclength =
            opt.max_arclength > 0.0 ? opt.max_arclength : default_max_arclength(inj, opt);
        r.injection = inj;
        r.planes = opt.planes;
        r.times = opt.times;
        const std::size_t Np = static_cast<std::size_t>(inj.M) * static_cast<std::size_t>(inj.M);
        const std::size_t P = opt.planes.size(), T = opt.times.size();
        r.Np = static_cast<int>(Np);
        r.P = static_cast<int>(P);
        r.T = static_cast<int>(T);
        r.seeds = make_seed_lattice(inj);

        // ---- device buffers (the only allocations)
        d_seeds_ = DeviceBuffer<double>(3 * Np);
        d_planes_ = DeviceBuffer<double>(std::max<std::size_t>(P, 1));
        d_times_ = DeviceBuffer<double>(std::max<std::size_t>(T, 1));
        d_stats_plane_ = DeviceBuffer<PassStats>(Np);
        d_stats_time_ = DeviceBuffer<PassStats>(Np);
        d_planes_x23_ = DeviceBuffer<double>(std::max<std::size_t>(Np * P * 2, 1));
        d_planes_tau_ = DeviceBuffer<double>(std::max<std::size_t>(Np * P, 1));
        d_planes_I_ = DeviceBuffer<double>(std::max<std::size_t>(Np * P, 1));
        d_times_x_ = DeviceBuffer<double>(std::max<std::size_t>(Np * T * 3, 1));
        d_times_I_ = DeviceBuffer<double>(std::max<std::size_t>(Np * T, 1));
        if (opt.covectors) {
            d_planes_cov_ = DeviceBuffer<double>(std::max<std::size_t>(Np * P * 6, 1));
            d_times_cov_ = DeviceBuffer<double>(std::max<std::size_t>(Np * T * 6, 1));
        } else {
            d_planes_cov_ = DeviceBuffer<double>();
            d_times_cov_ = DeviceBuffer<double>();
        }

        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_seeds_.data(), r.seeds.data(),
                                          3 * Np * sizeof(double), cudaMemcpyHostToDevice));
        if (P > 0)
            MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_planes_.data(), opt.planes.data(),
                                              P * sizeof(double), cudaMemcpyHostToDevice));
        if (T > 0)
            MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_times_.data(), opt.times.data(),
                                              T * sizeof(double), cudaMemcpyHostToDevice));
        // SF-33 C2: legacy-stream copies must land before ctx-stream work.
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

        // ---- injection weights: flux k g1 at the seeds
        {
            DeviceBuffer<double> d_flux(Np);
            const int blk = 64;
            const unsigned grid = static_cast<unsigned>((Np + blk - 1) / blk);
            ldev::flux_kernel<Field><<<grid, blk, 0, ctx.cuda_stream()>>>(
                field_, static_cast<int>(Np), d_seeds_.data(), d_flux.data());
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            ctx.synchronize();
            std::vector<double> flux(Np);
            MACROFLOW3D_CUDA_CHECK(cudaMemcpy(flux.data(), d_flux.data(), Np * sizeof(double),
                                              cudaMemcpyDeviceToHost));
            normalize_flux_weights(flux, r.weights, r.seed_backflow);
        }

        // ---- D-1 labels at the seeds
        const double nan = std::numeric_limits<double>::quiet_NaN();
        r.labels.assign(2 * Np, nan);
        if (labels != nullptr && inj.X0 == 0.0) {
            if (!labels->device_prepared()) {
                throw std::logic_error("LagrangianTracker::prepare: inlet labels not device-prepared");
            }
            std::vector<double> hy(Np), hz(Np);
            for (std::size_t p = 0; p < Np; ++p) {
                hy[p] = r.seeds[3 * p + 1];
                hz[p] = r.seeds[3 * p + 2];
            }
            DeviceBuffer<double> dy(Np), dz(Np), p1(Np), p2(Np);
            MACROFLOW3D_CUDA_CHECK(
                cudaMemcpy(dy.data(), hy.data(), Np * sizeof(double), cudaMemcpyHostToDevice));
            MACROFLOW3D_CUDA_CHECK(
                cudaMemcpy(dz.data(), hz.data(), Np * sizeof(double), cudaMemcpyHostToDevice));
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
            labels->evaluate_labels(ctx, DeviceSpan<const double>(dy.data(), Np),
                                    DeviceSpan<const double>(dz.data(), Np),
                                    DeviceSpan<double>(p1.data(), Np),
                                    DeviceSpan<double>(p2.data(), Np));
            ctx.synchronize();
            std::vector<double> h1(Np), h2(Np);
            MACROFLOW3D_CUDA_CHECK(
                cudaMemcpy(h1.data(), p1.data(), Np * sizeof(double), cudaMemcpyDeviceToHost));
            MACROFLOW3D_CUDA_CHECK(
                cudaMemcpy(h2.data(), p2.data(), Np * sizeof(double), cudaMemcpyDeviceToHost));
            for (std::size_t p = 0; p < Np; ++p) {
                r.labels[2 * p] = h1[p];
                r.labels[2 * p + 1] = h2[p];
            }
        }

        // ---- host result storage (sized once; track() does not allocate on the device)
        r.stats_plane.assign(Np, PassStats{});
        r.stats_time.assign(Np, PassStats{});
        r.planes_x23.assign(Np * P * 2, nan);
        r.planes_tau.assign(Np * P, nan);
        r.planes_I.assign(Np * P, nan);
        r.times_x.assign(Np * T * 3, nan);
        r.times_I.assign(Np * T, nan);
        if (opt.covectors) {
            r.planes_cov.assign(Np * P * 6, nan);
            r.times_cov.assign(Np * T * 6, nan);
        }
        prepared_ = true;
    }

    const LagrangianTrackResult& track(CudaContext& ctx, int block = 0) {
        if (!prepared_) {
            throw std::logic_error("LagrangianTracker::track: prepare() not called");
        }
        const int blk = block > 0 ? block : block_;
        if (blk < 1 || blk > 1024) {
            throw std::invalid_argument("LagrangianTracker::track: block must be in [1, 1024]");
        }
        LagrangianTrackResult& r = result_;
        const std::size_t Np = static_cast<std::size_t>(r.Np);
        const std::size_t P = static_cast<std::size_t>(r.P), T = static_cast<std::size_t>(r.T);

        ldev::KernelArgs a;
        a.Np = r.Np;
        a.seeds = d_seeds_.data();
        a.planes = d_planes_.data();
        a.P = r.P;
        a.times = d_times_.data();
        a.T = r.T;
        a.opt.tol = r.tol;
        a.opt.h_max = r.h_max;
        a.opt.min_step = r.min_step;
        a.opt.max_arclength = r.max_arclength;
        a.stats_plane = d_stats_plane_.data();
        a.stats_time = d_stats_time_.data();
        a.planes_x23 = d_planes_x23_.data();
        a.planes_tau = d_planes_tau_.data();
        a.planes_I = d_planes_I_.data();
        a.planes_cov = covectors_ ? d_planes_cov_.data() : nullptr;
        a.times_x = d_times_x_.data();
        a.times_I = d_times_I_.data();
        a.times_cov = covectors_ ? d_times_cov_.data() : nullptr;
        const unsigned grid = static_cast<unsigned>((Np + static_cast<std::size_t>(blk) - 1) /
                                                    static_cast<std::size_t>(blk));
        using clock = std::chrono::steady_clock;

        auto t0 = clock::now();
        if (single_launch_) {
            // SF-40 L6: one launch, plane pass then time pass per thread (own step sequences).
            if (covectors_)
                ldev::single_kernel<11, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
            else
                ldev::single_kernel<5, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            ctx.synchronize();
            r.seconds_single_kernel = std::chrono::duration<double>(clock::now() - t0).count();
            r.seconds_plane_kernel = r.seconds_time_kernel =
                std::numeric_limits<double>::quiet_NaN();
        } else {
            if (P > 0) {
                if (covectors_)
                    ldev::plane_kernel<11, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
                else
                    ldev::plane_kernel<5, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
                MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                ctx.synchronize();
            }
            r.seconds_plane_kernel = std::chrono::duration<double>(clock::now() - t0).count();

            t0 = clock::now();
            if (T > 0) {
                if (covectors_)
                    ldev::time_kernel<11, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
                else
                    ldev::time_kernel<5, Field><<<grid, blk, 0, ctx.cuda_stream()>>>(field_, a);
                MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                ctx.synchronize();
            }
            r.seconds_time_kernel = std::chrono::duration<double>(clock::now() - t0).count();
        }

        t0 = clock::now();
        auto down = [](void* dst, const void* src, std::size_t bytes) {
            if (bytes > 0)
                MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        };
        if (P > 0) {
            down(r.stats_plane.data(), d_stats_plane_.data(), Np * sizeof(PassStats));
            down(r.planes_x23.data(), d_planes_x23_.data(), Np * P * 2 * sizeof(double));
            down(r.planes_tau.data(), d_planes_tau_.data(), Np * P * sizeof(double));
            down(r.planes_I.data(), d_planes_I_.data(), Np * P * sizeof(double));
            if (covectors_)
                down(r.planes_cov.data(), d_planes_cov_.data(), Np * P * 6 * sizeof(double));
        }
        if (T > 0) {
            down(r.stats_time.data(), d_stats_time_.data(), Np * sizeof(PassStats));
            down(r.times_x.data(), d_times_x_.data(), Np * T * 3 * sizeof(double));
            down(r.times_I.data(), d_times_I_.data(), Np * T * sizeof(double));
            if (covectors_)
                down(r.times_cov.data(), d_times_cov_.data(), Np * T * 6 * sizeof(double));
        }
        r.seconds_download = std::chrono::duration<double>(clock::now() - t0).count();

        t0 = clock::now();
        finalize_track_result(r);
        r.seconds_moments = std::chrono::duration<double>(clock::now() - t0).count();
        return r;
    }

    bool prepared() const { return prepared_; }

    std::vector<const void*> device_storage_pointers() const {
        return {d_seeds_.data(),       d_planes_.data(),     d_times_.data(),
                d_stats_plane_.data(), d_stats_time_.data(), d_planes_x23_.data(),
                d_planes_tau_.data(),  d_planes_I_.data(),   d_times_x_.data(),
                d_times_I_.data(),     d_planes_cov_.data(), d_times_cov_.data()};
    }

  private:
    bool prepared_ = false;
    bool covectors_ = false;
    bool single_launch_ = false;
    int block_ = 64;
    Field field_{};
    LagrangianTrackResult result_;
    DeviceBuffer<double> d_seeds_, d_planes_, d_times_;
    DeviceBuffer<PassStats> d_stats_plane_, d_stats_time_;
    DeviceBuffer<double> d_planes_x23_, d_planes_tau_, d_planes_I_, d_times_x_, d_times_I_;
    DeviceBuffer<double> d_planes_cov_, d_times_cov_;
};

} // namespace lagrangian_labels
} // namespace macroflow3d
