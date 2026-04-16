/**
 * @file OperatorTestHarness.cuh
 * @brief Test harness for PSPTA operators (D, D†, L, A) and
 * PsptaInvariantField.
 *
 * Provides runnable verification routines for:
 *   1. D(constant) ≈ 0
 *   2. Adjoint exactness: <Dx, y> ≈ <x, D†y>
 *   3. L(constant) ≈ 0
 *   4. L(linear) ≈ 0 (interior)
 *   5. L symmetry: <Lx, y> ≈ <x, Ly>
 *   6. A symmetry: <Ax, y> ≈ <x, Ay>
 *   7. PsptaInvariantField construction, metadata, quality computation
 *   8. PsptaEngine::bind_invariants() smoke test
 *
 * ## Usage
 *
 * The harness is designed to be called from a main() or test framework:
 *
 * @code
 * CudaContext ctx;
 * Grid3D grid(32, 32, 32, 0.1, 0.1, 0.1);
 * VelocityField vel;
 * // ... initialize vel ...
 *
 * OperatorTestHarness harness(grid, &vel, ctx);
 * OperatorTestReport report = harness.run_all();
 *
 * if (report.all_passed()) {
 *     printf("All tests passed!\n");
 * } else {
 *     printf("FAILURES:\n%s", report.failures_summary().c_str());
 * }
 * @endcode
 *
 * @ingroup physics_particles_pspta
 */

#pragma once

#include "../../../../core/Grid3D.hpp"
#include "../../../../runtime/CudaContext.cuh"
#include "../../../common/fields.cuh"
#include "PsptaInvariantField.cuh"
#include "TransportOperator3D.cuh"
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Test result structures
// ============================================================================

/**
 * @brief Result of a single test.
 */
struct TestResult {
    std::string name;
    bool passed = false;
    double value = 0.0;
    double tolerance = 0.0;
    std::string message;
};

/**
 * @brief Full report from test harness.
 */
struct OperatorTestReport {
    std::vector<TestResult> results;
    int n_passed = 0;
    int n_failed = 0;

    bool all_passed() const { return n_failed == 0; }

    std::string summary() const {
        std::ostringstream oss;
        oss << "=== PSPTA Operator Test Report ===\n";
        oss << "Total: " << results.size() << ", Passed: " << n_passed << ", Failed: " << n_failed
            << "\n\n";

        for (const auto& r : results) {
            oss << (r.passed ? "[PASS]" : "[FAIL]") << " " << r.name << ": " << r.value;
            if (!r.passed) {
                oss << " (tolerance: " << r.tolerance << ")";
            }
            if (!r.message.empty()) {
                oss << " - " << r.message;
            }
            oss << "\n";
        }
        return oss.str();
    }

    std::string failures_summary() const {
        std::ostringstream oss;
        for (const auto& r : results) {
            if (!r.passed) {
                oss << "[FAIL] " << r.name << ": " << r.value << " (tolerance: " << r.tolerance
                    << ")";
                if (!r.message.empty()) {
                    oss << " - " << r.message;
                }
                oss << "\n";
            }
        }
        return oss.str();
    }
};

// ============================================================================
// OperatorTestHarness
// ============================================================================

/**
 * @brief Test harness for PSPTA operators.
 */
class OperatorTestHarness {
  public:
    /**
     * @param grid Grid for testing
     * @param vel  Velocity field (optional, will use synthetic if null)
     * @param ctx  CUDA context
     * @param mu   Regularization parameter for operator A
     */
    OperatorTestHarness(const Grid3D& grid, const VelocityField* vel, CudaContext& ctx,
                        double mu = 1e-6);

    /**
     * @brief Run all tests and return report.
     */
    OperatorTestReport run_all();

    /**
     * @brief Run individual tests (for selective execution).
     */
    TestResult test_D_constant();
    TestResult test_D_adjoint();
    TestResult test_L_constant();
    TestResult test_L_linear();
    TestResult test_L_symmetry();
    TestResult test_A_symmetry();
    TestResult test_invariant_field_smoke();
    TestResult test_engine_bind_invariants(); ///< Engine + new field binding

  private:
    Grid3D grid_;
    const VelocityField* vel_;
    CudaContext& ctx_;
    double mu_;

    // Operators (created lazily)
    std::unique_ptr<TransportOperator3D> D_;
    std::unique_ptr<LaplacianOperator3D> L_;
    std::unique_ptr<CombinedOperatorA> A_;

    void ensure_operators();

    // Tolerances
    static constexpr double TOL_D_CONST = 1e-10;
    static constexpr double TOL_ADJOINT = 1e-10;
    static constexpr double TOL_L_CONST = 1e-10;
    static constexpr double TOL_L_LINEAR = 1e-6; // Larger due to boundary effects
    static constexpr double TOL_SYMMETRY = 1e-10;
};

/**
 * @brief Run the test harness and print results to stdout.
 *
 * Convenience function for quick verification.
 *
 * @param grid Grid for testing
 * @param vel  Velocity field
 * @param ctx  CUDA context
 * @param mu   Regularization parameter
 * @return true if all tests passed
 */
bool run_operator_tests(const Grid3D& grid, const VelocityField* vel, CudaContext& ctx,
                        double mu = 1e-6);

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
