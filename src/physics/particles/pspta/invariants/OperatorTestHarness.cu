/**
 * @file OperatorTestHarness.cu
 * @brief Implementation of PSPTA operator test harness.
 */

#include "../../../../runtime/cuda_check.cuh"
#include "../PsptaEngine.hpp" // For engine binding test
#include "OperatorTestHarness.cuh"
#include <cstdio>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// OperatorTestHarness implementation
// ============================================================================

OperatorTestHarness::OperatorTestHarness(const Grid3D& grid, const VelocityField* vel,
                                         CudaContext& ctx, double mu)
    : grid_(grid), vel_(vel), ctx_(ctx), mu_(mu) {}

void OperatorTestHarness::ensure_operators() {
    if (!D_ && vel_ != nullptr) {
        D_ = std::make_unique<TransportOperator3D>(vel_, grid_);
    }
    if (!L_) {
        L_ = std::make_unique<LaplacianOperator3D>(grid_);
    }
    if (!A_ && D_ && L_) {
        A_ = std::make_unique<CombinedOperatorA>(D_.get(), L_.get(), mu_);
    }
}

TestResult OperatorTestHarness::test_D_constant() {
    TestResult r;
    r.name = "D(constant) ≈ 0";
    r.tolerance = TOL_D_CONST;

    if (!vel_) {
        r.passed = false;
        r.message = "No velocity field provided";
        return r;
    }

    ensure_operators();
    r.value = D_->test_constant_in_kernel(ctx_);
    r.passed = (r.value < r.tolerance);
    return r;
}

TestResult OperatorTestHarness::test_D_adjoint() {
    TestResult r;
    r.name = "D adjoint: <Dx,y> ≈ <x,D†y>";
    r.tolerance = TOL_ADJOINT;

    if (!vel_) {
        r.passed = false;
        r.message = "No velocity field provided";
        return r;
    }

    ensure_operators();
    r.value = D_->test_adjoint(ctx_, 42);
    r.passed = (r.value < r.tolerance);
    return r;
}

TestResult OperatorTestHarness::test_L_constant() {
    TestResult r;
    r.name = "L(constant) ≈ 0";
    r.tolerance = TOL_L_CONST;

    ensure_operators();
    r.value = L_->test_constant(ctx_);
    r.passed = (r.value < r.tolerance);
    return r;
}

TestResult OperatorTestHarness::test_L_linear() {
    TestResult r;
    r.name = "L(linear ax+by+cz) ≈ 0";
    r.tolerance = TOL_L_LINEAR;

    ensure_operators();
    // Only test x-direction: y,z are periodic and linear functions
    // are not periodic-compatible (wrap-around introduces O(N/dx²) artifacts).
    r.value = L_->test_linear(ctx_, 1.0, 0.0, 0.0);
    r.passed = (r.value < r.tolerance);

    if (!r.passed) {
        r.message = "Some boundary artifacts expected; tolerance is relaxed";
    }
    return r;
}

TestResult OperatorTestHarness::test_L_symmetry() {
    TestResult r;
    r.name = "L symmetry: <Lx,y> ≈ <x,Ly>";
    r.tolerance = TOL_SYMMETRY;

    ensure_operators();
    r.value = L_->test_symmetry(ctx_, 123);
    r.passed = (r.value < r.tolerance);
    return r;
}

TestResult OperatorTestHarness::test_A_symmetry() {
    TestResult r;
    r.name = "A symmetry: <Ax,y> ≈ <x,Ay>";
    r.tolerance = TOL_SYMMETRY;

    if (!vel_) {
        r.passed = false;
        r.message = "No velocity field provided (needed for D component)";
        return r;
    }

    ensure_operators();
    r.value = A_->test_symmetry(ctx_, 456);
    r.passed = (r.value < r.tolerance);
    return r;
}

TestResult OperatorTestHarness::test_invariant_field_smoke() {
    TestResult r;
    r.name = "PsptaInvariantField smoke test";
    r.tolerance = 0.0; // N/A for smoke test

    try {
        // Create and resize
        PsptaInvariantField inv;
        inv.resize(grid_);

        // Check basic properties
        bool ok = true;
        ok = ok && (inv.nx() == grid_.nx);
        ok = ok && (inv.ny() == grid_.ny);
        ok = ok && (inv.nz() == grid_.nz);
        ok = ok && (inv.is_valid());
        ok = ok && (inv.num_cells() == static_cast<size_t>(grid_.nx) * grid_.ny * grid_.nz);
        ok = ok && (inv.psi1_ptr() != nullptr);
        ok = ok && (inv.psi2_ptr() != nullptr);

        // Check metadata can be set
        InvariantConstructionInfo info;
        info.method = InvariantConstructionMethod::LegacyMarching;
        info.mu = mu_;
        info.construction_time_ms = 123.456;
        inv.set_construction_info(info);
        ok = ok && (inv.method() == InvariantConstructionMethod::LegacyMarching);
        ok = ok && (inv.construction_info().mu == mu_);

        // Check quality (will be zero for uninitialized field)
        if (vel_) {
            auto q = inv.compute_quality(*vel_, ctx_.cuda_stream());
            ok = ok && q.valid;
            // Values will be garbage but computation should succeed
        }

        // Check clear
        inv.clear();
        ok = ok && (!inv.is_valid());
        ok = ok && (inv.nx() == 0);

        r.passed = ok;
        r.value = ok ? 1.0 : 0.0;
    } catch (const std::exception& e) {
        r.passed = false;
        r.message = std::string("Exception: ") + e.what();
    }

    return r;
}

TestResult OperatorTestHarness::test_engine_bind_invariants() {
    TestResult r;
    r.name = "PsptaEngine::bind_invariants() smoke test";
    r.tolerance = 0.0; // N/A for smoke test

    try {
        // Create and resize invariant field
        PsptaInvariantField inv;
        inv.resize(grid_);

        // Create engine
        PsptaEngine engine(grid_, ctx_.cuda_stream(), 12345ULL);

        // Bind velocity (required for engine)
        if (vel_) {
            engine.bind_velocity(vel_);
        }

        // Bind invariant field via new interface
        engine.bind_invariants(&inv);

        // Verify no crash — success
        r.passed = true;
        r.value = 1.0;
        r.message = "Engine accepts PsptaInvariantField binding";
    } catch (const std::exception& e) {
        r.passed = false;
        r.message = std::string("Exception: ") + e.what();
    }

    return r;
}

OperatorTestReport OperatorTestHarness::run_all() {
    OperatorTestReport report;

    // Run all tests
    report.results.push_back(test_L_constant());
    report.results.push_back(test_L_linear());
    report.results.push_back(test_L_symmetry());
    report.results.push_back(test_invariant_field_smoke());
    report.results.push_back(test_engine_bind_invariants());

    // Tests requiring velocity field
    if (vel_) {
        report.results.push_back(test_D_constant());
        report.results.push_back(test_D_adjoint());
        report.results.push_back(test_A_symmetry());
    }

    // Count results
    for (const auto& r : report.results) {
        if (r.passed)
            report.n_passed++;
        else
            report.n_failed++;
    }

    return report;
}

// ============================================================================
// Convenience function
// ============================================================================

bool run_operator_tests(const Grid3D& grid, const VelocityField* vel, CudaContext& ctx, double mu) {
    OperatorTestHarness harness(grid, vel, ctx, mu);
    OperatorTestReport report = harness.run_all();

    printf("%s", report.summary().c_str());

    return report.all_passed();
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
