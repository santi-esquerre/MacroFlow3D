#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"

namespace macroflow3d {

class GpuTimer {
public:
    GpuTimer() {
        MACROFLOW3D_CUDA_CHECK(cudaEventCreate(&start_));
        MACROFLOW3D_CUDA_CHECK(cudaEventCreate(&stop_));
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
        MACROFLOW3D_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    float stop(cudaStream_t stream) {
        MACROFLOW3D_CUDA_CHECK(cudaEventRecord(stop_, stream));
        MACROFLOW3D_CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float elapsed_ms = 0.0f;
        MACROFLOW3D_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace macroflow3d
