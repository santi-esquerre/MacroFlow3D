#pragma once

#include <map>
#include <string>

namespace macroflow3d::streamfunctions::test {

struct CaseResult {
    bool pass{};
    std::string name;
    std::string kind;
    std::string grid;
    double coarse_norm{};
    double fine_norm{};
    std::string expected_order;
    std::string observed_order{"n/a"};
    std::string threshold;
};

using CaseFunction = CaseResult (*)();
using CaseRegistry = std::map<std::string, CaseFunction>;

// GPU production-operator cases. The CPU-only runner owns the combined CLI.
[[nodiscard]] CaseRegistry gpu_case_registry();

// GPU mean-zero projector cases.  Kept separate from the SF-02 operator
// cases so the projector's workspace and stream contract remains explicit.
[[nodiscard]] CaseRegistry mean_zero_projector_case_registry();

// GPU projected-PCG manufactured controls.  Their RHS and residual oracle are
// explicit CPU long-double stencils, kept separate from the GPU operator.
[[nodiscard]] CaseRegistry projected_pcg_case_registry();

// SF-05 quantitative acceptance controls for the reused, projected positive
// multigrid hierarchy.  Kept separate so their CPU long-double oracle remains
// visibly independent from the production kernels.
[[nodiscard]] CaseRegistry multigrid_reuse_case_registry();

// SF-06 affine-periodic RHS acceptance controls.  Kept separate so their
// independent long-double oracle cannot accidentally become a production API.
[[nodiscard]] CaseRegistry affine_periodic_rhs_case_registry();

// SF-07 positive total-gradient acceptance controls.  These retain an
// independent long-double CPU stencil and deliberately unequal spacing.
[[nodiscard]] CaseRegistry streamfunction_gradient_case_registry();

// SF-08 positive Hessian-vector/B controls.  They deliberately run the SF-07
// total-gradient kernel immediately before the register-only HVP/B kernel.
[[nodiscard]] CaseRegistry hessian_vector_b_case_registry();

}  // namespace macroflow3d::streamfunctions::test
