#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    try {
        // Create CUDA context for device 0
        rwpt::CudaContext ctx(0);
        
        // Get device properties
        cudaDeviceProp prop;
        RWPT_CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx.device()));
        
        // Print device information
        std::cout << "GPU Device Information:\n";
        std::cout << "  Name: " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  SM Count: " << prop.multiProcessorCount << "\n";
        
        // Convert global memory to GB
        double memory_gb = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  Total Global Memory: " << memory_gb << " GB\n";
        
        // Verify stream and cublas handle are created
        std::cout << "\nCUDA Context:\n";
        std::cout << "  Stream created: " << (ctx.cuda_stream() != nullptr ? "yes" : "no") << "\n";
        std::cout << "  cuBLAS handle created: " << (ctx.cublas_handle() != nullptr ? "yes" : "no") << "\n";
        
        std::cout << "\nSmoke test passed!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
