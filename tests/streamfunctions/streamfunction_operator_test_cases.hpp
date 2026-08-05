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

}  // namespace macroflow3d::streamfunctions::test
