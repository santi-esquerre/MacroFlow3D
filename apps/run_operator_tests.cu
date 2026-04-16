/**
 * @file run_operator_tests.cu
 * @brief CTest-friendly runner for PSPTA operator verification.
 *
 * Exercises the 8 OperatorTestHarness checks on a 16³ uniform-flow grid.
 * Returns 0 on success, 1 on failure — suitable for ctest.
 */

#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/common/fields.cuh"
#include "src/physics/particles/pspta/invariants/OperatorTestHarness.cuh"
#include "src/runtime/CudaContext.cuh"

#include <cstdio>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;
using namespace macroflow3d::physics::particles::pspta;

int main() {
    CudaContext ctx(0);

    constexpr int N = 16;
    constexpr real dx = 1.0 / N;
    Grid3D grid(N, N, N, dx, dx, dx);

    VelocityField vel(grid);

    // Uniform flow: vx = 1, vy = vz = 0
    {
        std::vector<real> hU(vel.size_U(), 1.0);
        std::vector<real> hV(vel.size_V(), 0.0);
        std::vector<real> hW(vel.size_W(), 0.0);
        cudaMemcpy(vel.U.data(), hU.data(), hU.size() * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(vel.V.data(), hV.data(), hV.size() * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(vel.W.data(), hW.data(), hW.size() * sizeof(real), cudaMemcpyHostToDevice);
    }

    bool ok = run_operator_tests(grid, &vel, ctx, 1.0e-6);

    std::printf("\n=== Operator tests: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
