#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"

namespace rwpt {

class GpuTimer {
public:
    GpuTimer() {
        RWPT_CUDA_CHECK(cudaEventCreate(&start_));
        RWPT_CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
    }

    // Delete copy semantics
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // Move semantics
    GpuTimer(GpuTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    GpuTimer& operator=(GpuTimer&& other) noexcept {
        if (this != &other) {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
            
            start_ = other.start_;
            stop_ = other.stop_;
            
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    void start(cudaStream_t stream) {
        RWPT_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop(cudaStream_t stream) {
        RWPT_CUDA_CHECK(cudaEventRecord(stop_, stream));
        RWPT_CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float elapsed_ms = 0.0f;
        RWPT_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace rwpt
