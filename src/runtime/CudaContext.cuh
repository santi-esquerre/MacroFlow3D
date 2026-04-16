#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace macroflow3d {

class CudaContext {
  public:
    explicit CudaContext(int device_id = 0);
    ~CudaContext();

    // Move semantics
    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;

    // Delete copy semantics
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    // Getters
    int device() const noexcept { return device_id_; }
    cudaStream_t cuda_stream() const noexcept { return stream_; }
    cublasHandle_t cublas_handle() const noexcept { return cublas_; }

    // Explicit synchronization
    void synchronize() const;

  private:
    int device_id_;
    cudaStream_t stream_;
    cublasHandle_t cublas_;

    void release();
};

} // namespace macroflow3d
