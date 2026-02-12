#include "CudaContext.cuh"
#include "cuda_check.cuh"

namespace macroflow3d {

CudaContext::CudaContext(int device_id)
    : device_id_(device_id), stream_(nullptr), cublas_(nullptr) {
    
    MACROFLOW3D_CUDA_CHECK(cudaSetDevice(device_id_));
    MACROFLOW3D_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    MACROFLOW3D_CUBLAS_CHECK(cublasCreate(&cublas_));
    MACROFLOW3D_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
}

CudaContext::~CudaContext() {
    release();
}

CudaContext::CudaContext(CudaContext&& other) noexcept
    : device_id_(other.device_id_),
      stream_(other.stream_),
      cublas_(other.cublas_) {
    
    other.stream_ = nullptr;
    other.cublas_ = nullptr;
}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        release();
        
        device_id_ = other.device_id_;
        stream_ = other.stream_;
        cublas_ = other.cublas_;
        
        other.stream_ = nullptr;
        other.cublas_ = nullptr;
    }
    return *this;
}

void CudaContext::synchronize() const {
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaContext::release() {
    if (cublas_ != nullptr) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

} // namespace macroflow3d
