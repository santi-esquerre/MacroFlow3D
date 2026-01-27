#include "../src/core/Scalar.hpp"
#include "../src/core/Grid3D.hpp"
#include "../src/core/BCSpec.hpp"
#include "../src/core/DeviceBuffer.cuh"
#include "../src/runtime/CudaContext.cuh"
#include "../src/runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    try {
        std::cout << "Testing core contracts...\n\n";
        
        // Create CUDA context
        rwpt::CudaContext ctx(0);
        
        // Test Scalar type
        std::cout << "1. Testing rwpt::real type:\n";
        rwpt::real test_value = 3.14159;
        std::cout << "   rwpt::real value: " << test_value << "\n\n";
        
        // Test Grid3D
        std::cout << "2. Testing rwpt::Grid3D:\n";
        rwpt::Grid3D grid(16, 8, 4, 1.0, 1.0, 1.0);
        std::cout << "   Dimensions: " << grid.nx << " x " << grid.ny << " x " << grid.nz << "\n";
        std::cout << "   Spacing: dx=" << grid.dx << ", dy=" << grid.dy << ", dz=" << grid.dz << "\n";
        std::cout << "   Total cells: " << grid.num_cells() << "\n";
        std::cout << "   Linear index at (1,2,3): " << grid.idx(1, 2, 3) << "\n\n";
        
        // Test BCSpec
        std::cout << "3. Testing rwpt::BCSpec:\n";
        rwpt::BCSpec bc;
        std::cout << "   Default BC created (all Dirichlet with value 0)\n";
        std::cout << "   xmin: type=" << static_cast<int>(bc.xmin.type) 
                  << ", value=" << bc.xmin.value << "\n\n";
        
        // Test DeviceBuffer
        std::cout << "4. Testing rwpt::DeviceBuffer:\n";
        const size_t buffer_size = 128;
        rwpt::DeviceBuffer<rwpt::real> buf(buffer_size);
        std::cout << "   Allocated buffer of size: " << buf.size() << "\n";
        std::cout << "   Device pointer: " << buf.data() << "\n";
        
        // Optional: Fill with cudaMemset
        RWPT_CUDA_CHECK(cudaMemsetAsync(buf.data(), 0, buffer_size * sizeof(rwpt::real), 
                                         ctx.cuda_stream()));
        ctx.synchronize();
        std::cout << "   Buffer zeroed successfully\n\n";
        
        // Test DeviceSpan
        std::cout << "5. Testing rwpt::DeviceSpan:\n";
        auto span = buf.span();
        std::cout << "   Span size: " << span.size() << "\n";
        std::cout << "   Span data pointer: " << span.data() << "\n\n";
        
        std::cout << "All core contract tests passed!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
