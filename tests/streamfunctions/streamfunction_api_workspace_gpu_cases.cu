#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/multigrid/coefficient_hierarchy.cuh"
#include "src/multigrid/mg_types.hpp"
#include "src/numerics/blas/sum.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/numerics/solvers/pcg.cuh"
#include "src/numerics/solvers/projected_positive_mg_preconditioner.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-12 T02: GPU acceptance test cases for the public streamfunction API,
// ownership workspace, and exact memory estimator
// (src/physics/streamfunctions/{StreamfunctionTypes.hpp,
// StreamfunctionWorkspace.cuh/.cu}). Follows the registry/CaseResult
// convention established by the SF-02..SF-11 GPU case files (see, e.g.,
// physical_diagnostics_gpu_cases.cu). `solve_streamfunctions` is
// declaration-only in SF-12 (SF-13's deliverable); nothing in this file
// references it.

namespace macroflow3d::streamfunctions::test {
namespace {

// ---------------------------------------------------------------------------
// Small local helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
}

[[nodiscard]] BCSpec triply_periodic() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    return bc;
}

// Nonowning, non-null-content problem view sufficient for
// validate_streamfunction_problem, which is host-size-only and never
// dereferences buffer contents (documented in StreamfunctionTypes.hpp).
// Using nullptr-backed DeviceSpans with the correct *size* is the same
// pattern already used by the SF-11 error-contract cases.
[[nodiscard]] StreamfunctionProblemView valid_problem_view(const Grid3D& grid) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(nullptr, grid.num_cells());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{
        DeviceSpan<const real>(nullptr, compact_mac_u_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_v_size(grid)),
        DeviceSpan<const real>(nullptr, compact_mac_w_size(grid))};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

[[nodiscard]] StreamfunctionSolverConfig valid_config() {
    StreamfunctionSolverConfig config;
    // Every sub-config keeps its production default, which the increment
    // spec/T01 documentation already asserts is self-consistent (mg
    // pre==post>=1, coarse_solve_iters>0; linear max_iter>=0,
    // check_every>0, rtol>=0; histogram 0<c_min_rel<c_max_rel; diagnostics
    // angle_exclusion_rel/low_speed_rel>0, num_degeneracy_thresholds=0;
    // eta=1, epsilon=1e-2).
    return config;
}

template <typename Callable>
[[nodiscard]] bool rejects_with_invalid_argument(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "api_workspace_contract name=" << name
                  << " exception=none expected=std::invalid_argument\n";
        return false;
    } catch (const std::invalid_argument& error) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=std::invalid_argument message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::invalid_argument\n";
        return false;
    } catch (...) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=non-standard expected=std::invalid_argument\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool rejects_with_logic_error(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "api_workspace_contract name=" << name
                  << " exception=none expected=std::logic_error\n";
        return false;
    } catch (const std::logic_error& error) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=std::logic_error message=" << error.what() << '\n';
        return true;
    } catch (const std::exception& error) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=std::exception message=" << error.what()
                  << " expected=std::logic_error\n";
        return false;
    } catch (...) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=non-standard expected=std::logic_error\n";
        return false;
    }
}

template <typename Callable>
[[nodiscard]] bool accepts_without_throw(const char* name, Callable&& callable) {
    try {
        callable();
        std::cout << "api_workspace_contract name=" << name << " exception=none expected=none\n";
        return true;
    } catch (const std::exception& error) {
        std::cout << "api_workspace_contract name=" << name << " exception=" << error.what()
                  << " expected=none\n";
        return false;
    } catch (...) {
        std::cout << "api_workspace_contract name=" << name
                  << " exception=non-standard expected=none\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Case 1: exhaustive error contract for validate_streamfunction_problem,
// StreamfunctionWorkspace::prepare, estimate_streamfunction_memory, and
// pre-prepare accessor use.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_error_contract() {
    const Grid3D base_grid = isotropic_grid(16);
    const StreamfunctionProblemView base_problem = valid_problem_view(base_grid);
    const StreamfunctionSolverConfig base_config = valid_config();

    std::size_t checks = 0;
    bool pass = true;
    const auto require_invalid = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_invalid_argument(name, callable) && pass;
    };
    const auto require_logic = [&](const char* name, const auto& callable) {
        ++checks;
        pass = rejects_with_logic_error(name, callable) && pass;
    };
    const auto require_ok = [&](const char* name, const auto& callable) {
        ++checks;
        pass = accepts_without_throw(name, callable) && pass;
    };

    // --- validate_streamfunction_problem: conductivity size mismatch ---
    require_invalid("conductivity_short", [&] {
        auto p = base_problem;
        p.conductivity = DeviceSpan<const real>(nullptr, base_grid.num_cells() - 1);
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("conductivity_long", [&] {
        auto p = base_problem;
        p.conductivity = DeviceSpan<const real>(nullptr, base_grid.num_cells() + 1);
        validate_streamfunction_problem(base_grid, p, base_config);
    });

    // --- validate_streamfunction_problem: each CompactMAC Darcy component wrong size ---
    // NOTE: base_grid is an isotropic cube (nx==ny==nz), so the three
    // CompactMAC face sizes (nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1) are all
    // numerically EQUAL on this fixture; swapping between them would not
    // actually produce a wrong size. grid.num_cells() (nx*ny*nz) is smaller
    // than every face size and is used as the deliberately-wrong size here.
    require_invalid("darcy_u_wrong_size", [&] {
        auto p = base_problem;
        p.darcy_velocity.u = DeviceSpan<const real>(nullptr, base_grid.num_cells());
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("darcy_v_wrong_size", [&] {
        auto p = base_problem;
        p.darcy_velocity.v = DeviceSpan<const real>(nullptr, base_grid.num_cells());
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("darcy_w_wrong_size", [&] {
        auto p = base_problem;
        p.darcy_velocity.w = DeviceSpan<const real>(nullptr, base_grid.num_cells());
        validate_streamfunction_problem(base_grid, p, base_config);
    });

    // --- validate_streamfunction_problem: each nonperiodic BC face direction ---
    require_invalid("bc_x_nonperiodic", [&] {
        auto p = base_problem;
        p.bc.xmin = BCFace(BCType::Dirichlet, real{0});
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("bc_y_nonperiodic", [&] {
        auto p = base_problem;
        p.bc.ymax = BCFace(BCType::Neumann, real{0});
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("bc_z_nonperiodic", [&] {
        auto p = base_problem;
        p.bc.zmin = BCFace(BCType::Dirichlet, real{0});
        validate_streamfunction_problem(base_grid, p, base_config);
    });

    // --- validate_streamfunction_problem: nonfinite/nonpositive spacing per direction ---
    const std::vector<std::pair<real, const char*>> bad_spacings{
        {real{0}, "zero"},
        {real{-1}, "negative"},
        {std::numeric_limits<real>::quiet_NaN(), "nan"},
        {std::numeric_limits<real>::infinity(), "inf"}};
    for (const auto& entry : bad_spacings) {
        const real bad_value = entry.first;
        const char* value_name = entry.second;
        {
            const std::string label = std::string("dx_") + value_name;
            require_invalid(label.c_str(), [&, bad_value] {
                auto g = base_grid;
                g.dx = bad_value;
                validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
            });
        }
        {
            const std::string label = std::string("dy_") + value_name;
            require_invalid(label.c_str(), [&, bad_value] {
                auto g = base_grid;
                g.dy = bad_value;
                validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
            });
        }
        {
            const std::string label = std::string("dz_") + value_name;
            require_invalid(label.c_str(), [&, bad_value] {
                auto g = base_grid;
                g.dz = bad_value;
                validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
            });
        }
    }

    // --- validate_streamfunction_problem: anisotropic spacing ---
    require_invalid("anisotropic_dx_ne_dy", [&] {
        auto g = base_grid;
        g.dy = g.dx * real{2};
        validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
    });
    require_invalid("anisotropic_dx_ne_dz", [&] {
        auto g = base_grid;
        g.dz = g.dx * real{3};
        validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
    });

    // --- validate_streamfunction_problem (and prepare()/estimate()): each extent < 2 ---
    require_invalid("extent_nx_one", [&] {
        auto g = base_grid;
        g.nx = 1;
        validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
    });
    require_invalid("extent_ny_one", [&] {
        auto g = base_grid;
        g.ny = 1;
        validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
    });
    require_invalid("extent_nz_one", [&] {
        auto g = base_grid;
        g.nz = 1;
        validate_streamfunction_problem(g, valid_problem_view(base_grid), base_config);
    });
    require_invalid("prepare_extent_one", [&] {
        StreamfunctionWorkspace ws;
        auto g = base_grid;
        g.nx = 1;
        ws.prepare(g, base_config);
    });

    // --- validate_streamfunction_problem: nonfinite affine gauge gradient ---
    require_invalid("gauge_psi1_nan", [&] {
        auto p = base_problem;
        p.gauge.psi1_gradient.x = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, p, base_config);
    });
    require_invalid("gauge_psi2_inf", [&] {
        auto p = base_problem;
        p.gauge.psi2_gradient.z = std::numeric_limits<real>::infinity();
        validate_streamfunction_problem(base_grid, p, base_config);
    });

    // --- grid fails MG coarsenability for config.mg.num_levels ---
    require_invalid("mg_coarsen_odd_intermediate", [&] {
        // 20 -> 10 -> 5 (odd): fails the per-level even-extent requirement.
        const Grid3D g20 = isotropic_grid(20);
        validate_streamfunction_problem(g20, valid_problem_view(g20), base_config);
    });
    require_invalid("mg_num_levels_too_deep", [&] {
        // 16 -> 8 -> 4 -> 2 -> 1(<2, break): only 4 levels are attainable, not 5.
        auto config = base_config;
        config.mg.num_levels = 5;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("prepare_mg_coarsen_odd_intermediate", [&] {
        StreamfunctionWorkspace ws;
        const Grid3D g20 = isotropic_grid(20);
        ws.prepare(g20, base_config);
    });
    require_invalid("estimate_mg_num_levels_too_deep", [&] {
        auto config = base_config;
        config.mg.num_levels = 5;
        (void)estimate_streamfunction_memory(base_grid, config);
    });

    // --- invalid mg (pre!=post, pre=0, coarse_solve_iters=0) ---
    require_invalid("mg_pre_ne_post", [&] {
        auto config = base_config;
        config.mg.pre_smooth = 2;
        config.mg.post_smooth = 3;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("mg_pre_zero", [&] {
        auto config = base_config;
        config.mg.pre_smooth = 0;
        config.mg.post_smooth = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("mg_coarse_solve_iters_zero", [&] {
        auto config = base_config;
        config.mg.coarse_solve_iters = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // --- invalid linear (max_iter<0, check_every=0, rtol=-1, rtol=NaN) ---
    require_invalid("linear_max_iter_negative", [&] {
        auto config = base_config;
        config.linear.max_iter = -1;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("linear_check_every_zero", [&] {
        auto config = base_config;
        config.linear.check_every = 0;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("linear_rtol_negative", [&] {
        auto config = base_config;
        config.linear.rtol = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("linear_rtol_nan", [&] {
        auto config = base_config;
        config.linear.rtol = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("estimate_linear_max_iter_negative", [&] {
        auto config = base_config;
        config.linear.max_iter = -1;
        (void)estimate_streamfunction_memory(base_grid, config);
    });

    // --- invalid histogram (c_min_rel=0, c_max_rel<=c_min_rel, NaN) ---
    require_invalid("histogram_c_min_zero", [&] {
        auto config = base_config;
        config.histogram.c_min_rel = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("histogram_c_max_not_greater", [&] {
        auto config = base_config;
        config.histogram.c_max_rel = config.histogram.c_min_rel;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("histogram_c_min_nan", [&] {
        auto config = base_config;
        config.histogram.c_min_rel = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("histogram_c_max_nan", [&] {
        auto config = base_config;
        config.histogram.c_max_rel = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("estimate_histogram_c_min_zero", [&] {
        auto config = base_config;
        config.histogram.c_min_rel = real{0};
        (void)estimate_streamfunction_memory(base_grid, config);
    });

    // --- invalid diagnostics (angle_exclusion_rel<=0, low_speed_rel<=0, bad
    //     threshold count, non-increasing thresholds, nonpositive threshold) ---
    require_invalid("diagnostics_angle_exclusion_nonpositive", [&] {
        auto config = base_config;
        config.diagnostics.angle_exclusion_rel = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("diagnostics_low_speed_nonpositive", [&] {
        auto config = base_config;
        config.diagnostics.low_speed_rel = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("diagnostics_threshold_count_out_of_range", [&] {
        auto config = base_config;
        config.diagnostics.num_degeneracy_thresholds = kMaxDegeneracyThresholds + 1;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("diagnostics_thresholds_non_increasing", [&] {
        auto config = base_config;
        config.diagnostics.num_degeneracy_thresholds = 2;
        config.diagnostics.degeneracy_thresholds[0] = real{0.5};
        config.diagnostics.degeneracy_thresholds[1] = real{0.3};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("diagnostics_threshold_nonpositive", [&] {
        auto config = base_config;
        config.diagnostics.num_degeneracy_thresholds = 1;
        config.diagnostics.degeneracy_thresholds[0] = real{0};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("estimate_diagnostics_angle_exclusion_nonpositive", [&] {
        auto config = base_config;
        config.diagnostics.angle_exclusion_rel = real{0};
        (void)estimate_streamfunction_memory(base_grid, config);
    });

    // --- invalid eta/epsilon (negative, NaN) ---
    require_invalid("eta_negative", [&] {
        auto config = base_config;
        config.eta = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("eta_nan", [&] {
        auto config = base_config;
        config.eta = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("epsilon_negative", [&] {
        auto config = base_config;
        config.epsilon = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("epsilon_nan", [&] {
        auto config = base_config;
        config.epsilon = std::numeric_limits<real>::quiet_NaN();
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("estimate_eta_negative", [&] {
        auto config = base_config;
        config.eta = real{-1};
        (void)estimate_streamfunction_memory(base_grid, config);
    });

    // --- invalid source-side (top-level) thresholds ---
    require_invalid("source_side_threshold_count_out_of_range", [&] {
        auto config = base_config;
        config.num_degeneracy_thresholds = kMaxDegeneracyThresholds + 1;
        validate_streamfunction_problem(base_grid, base_problem, config);
    });
    require_invalid("source_side_threshold_negative", [&] {
        auto config = base_config;
        config.num_degeneracy_thresholds = 1;
        config.degeneracy_thresholds[0] = real{-1};
        validate_streamfunction_problem(base_grid, base_problem, config);
    });

    // --- pre-prepare accessor use: StreamfunctionFields ---
    require_logic("fields_fluctuations_unprepared", [&] {
        StreamfunctionFields fields;
        (void)fields.fluctuations();
    });

    // --- pre-prepare accessor use: StreamfunctionWorkspace ---
    require_logic("workspace_q_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.q();
    });
    require_logic("workspace_rhs1_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.rhs1();
    });
    require_logic("workspace_rhs2_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.rhs2();
    });
    require_logic("workspace_f1_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.f1();
    });
    require_logic("workspace_f2_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.f2();
    });
    require_logic("workspace_v_psi_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.v_psi();
    });
    require_logic("workspace_residual_workspace_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.residual_workspace();
    });
    require_logic("workspace_diagnostics_workspace_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.diagnostics_workspace();
    });
    require_logic("workspace_affine_rhs_workspace_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.affine_rhs_workspace();
    });
    require_logic("workspace_pcg_workspace_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.pcg_workspace();
    });
    require_logic("workspace_hierarchy_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.hierarchy();
    });
    require_logic("workspace_preconditioner_unprepared", [&] {
        StreamfunctionWorkspace ws;
        (void)ws.preconditioner();
    });

    // --- a fully valid problem/config validates WITHOUT throwing ---
    require_ok("valid_problem_config_no_throw", [&] {
        validate_streamfunction_problem(base_grid, base_problem, base_config);
    });
    require_ok("prepare_valid_no_throw", [&] {
        StreamfunctionWorkspace ws;
        ws.prepare(base_grid, base_config);
    });
    require_ok("estimate_valid_no_throw", [&] {
        (void)estimate_streamfunction_memory(base_grid, base_config);
    });

    std::cout << "api_workspace_error_contract total_checks=" << checks
              << " pass=" << (pass ? "true" : "false") << '\n';

    return {pass,
            "api_workspace_error_contract",
            "contract",
            "16x16x16 + config mutations",
            static_cast<double>(checks),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "validate_streamfunction_problem/prepare()/estimate_streamfunction_memory throw "
            "std::invalid_argument for every documented precondition violation; pre-prepare "
            "accessor use throws std::logic_error; a fully valid problem/config validates "
            "without throwing"};
}

// ---------------------------------------------------------------------------
// Shared closed-form memory-report reconstruction (case 2/3/6). Independently
// recomputes every StreamfunctionMemoryReport field from the documented
// per-primitive layouts in StreamfunctionWorkspace.cu/ResidualEvaluator.cu/
// Diagnostics.cu/affine_periodic_rhs.cu/MeanZeroProjector.cu, calling only
// the independent host-size primitive blas::sum_workspace_bytes(n) for
// backend-dependent CUB temporary-storage sizes. This function is NEVER
// called by the production estimator/accessor code paths.
// ---------------------------------------------------------------------------

[[nodiscard]] std::vector<Grid3D> simulate_mg_hierarchy_levels(const Grid3D& finest,
                                                                int num_levels) {
    std::vector<Grid3D> levels;
    if (num_levels <= 0) {
        return levels;
    }
    Grid3D current = finest;
    for (int l = 0; l < num_levels; ++l) {
        levels.push_back(current);
        if (l < num_levels - 1) {
            current.nx /= 2;
            current.ny /= 2;
            current.nz /= 2;
            current.dx *= real{2};
            current.dy *= real{2};
            current.dz *= real{2};
            if (current.nx < 2 || current.ny < 2 || current.nz < 2) {
                break;
            }
        }
    }
    return levels;
}

// MeanZeroWorkspace::estimate_device_bytes(n) (MeanZeroProjector.cu): the CUB
// sum-reduction temp bytes plus one result scalar.
[[nodiscard]] std::size_t closed_mean_zero_bytes(std::size_t n) {
    return blas::sum_workspace_bytes(n) + sizeof(real);
}

// AffinePeriodicRhsWorkspace::estimate_device_bytes(n) (affine_periodic_rhs.cu):
// mean_zero_.prepare(n) plus raw_means_/projected_means_ (2 elements each).
[[nodiscard]] std::size_t closed_affine_rhs_bytes(std::size_t n) {
    return closed_mean_zero_bytes(n) + 2 * sizeof(real) + 2 * sizeof(real);
}

// ProjectedPCGWorkspace estimate (pcg.cuh::prepare/StreamfunctionWorkspace.cu
// projected_pcg_workspace_estimate_bytes): pcg.ensure(n) (4n reals + 1 scalar)
// + projected_rhs (n reals) + mean_zero.prepare(n).
[[nodiscard]] std::size_t closed_pcg_workspace_bytes(std::size_t n) {
    return 4 * n * sizeof(real) + sizeof(real) + n * sizeof(real) + closed_mean_zero_bytes(n);
}

// StreamfunctionResidualWorkspace::estimate_device_bytes(n) (ResidualEvaluator.cu):
// 21 fields of size n + kNumScalars(=5, line 24) + kNumHistogramSlots
// (=kResidualHistogramBins+2, public constant + underflow/overflow) +
// (2+kMaxDegeneracyThresholds) source counters (public constant) + the
// top-level AffinePeriodicRhsWorkspace + a private MeanZeroWorkspace (for
// projecting the combined G fields) + one default-constructed reduction
// scalar (this module only ever drives its ReductionWorkspace through
// nrm2/dot, never CUB sum, so its capacity is exactly one real).
[[nodiscard]] std::size_t closed_residual_workspace_bytes(std::size_t n) {
    constexpr std::size_t kResidualNumFieldsN = 21;
    constexpr std::size_t kResidualNumScalars = 5; // ResidualEvaluator.cu:24
    const std::size_t histogram_slots =
        static_cast<std::size_t>(kResidualHistogramBins) + 2; // bins + underflow + overflow
    const std::size_t source_counters = 2 + static_cast<std::size_t>(kMaxDegeneracyThresholds);
    std::size_t bytes = kResidualNumFieldsN * n * sizeof(real);
    bytes += kResidualNumScalars * sizeof(real);
    bytes += histogram_slots * sizeof(unsigned long long);
    bytes += source_counters * sizeof(unsigned long long);
    bytes += closed_affine_rhs_bytes(n);
    bytes += closed_mean_zero_bytes(n);
    bytes += sizeof(real);
    return bytes;
}

// StreamfunctionDiagnosticsWorkspace::estimate_device_bytes(grid) (Diagnostics.cu):
// 27 fields of size n + kNumScalars(=41, Diagnostics.cu:54) +
// blas::sum_workspace_bytes(n)+sizeof(real) (blas::prepare_sum_workspace) +
// kNumCounters (=kCounterLeading(2) + 2*kMaxDegeneracyThresholds,
// Diagnostics.cu:56-61; kCounterLeading is independently derivable from the
// public kMaxDegeneracyThresholds constant plus the fixed leading-count of 2).
[[nodiscard]] std::size_t closed_diagnostics_workspace_bytes(const Grid3D& grid) {
    const std::size_t n = grid.num_cells();
    constexpr std::size_t kDiagNumFieldsN = 27;
    constexpr std::size_t kDiagNumScalars = 41; // Diagnostics.cu:54
    const std::size_t kDiagNumCounters =
        2 + 2 * static_cast<std::size_t>(kMaxDegeneracyThresholds); // Diagnostics.cu:56-61
    std::size_t bytes = kDiagNumFieldsN * n * sizeof(real);
    bytes += kDiagNumScalars * sizeof(real);
    bytes += blas::sum_workspace_bytes(n) + sizeof(real);
    bytes += kDiagNumCounters * sizeof(unsigned long long);
    return bytes;
}

[[nodiscard]] std::size_t closed_mg_hierarchy_bytes(const std::vector<Grid3D>& levels) {
    std::size_t bytes = 0;
    for (const auto& level : levels) {
        bytes += 4 * level.num_cells() * sizeof(real);
    }
    return bytes;
}

[[nodiscard]] std::size_t closed_mg_preconditioner_bytes(const std::vector<Grid3D>& levels) {
    std::size_t bytes = 0;
    for (const auto& level : levels) {
        bytes += closed_mean_zero_bytes(level.num_cells());
    }
    return bytes;
}

[[nodiscard]] StreamfunctionMemoryReport closed_form_memory_report(
    const Grid3D& grid, const StreamfunctionSolverConfig& config) {
    const std::size_t n = grid.num_cells();
    const auto levels = simulate_mg_hierarchy_levels(grid, config.mg.num_levels);

    StreamfunctionMemoryReport report{};
    report.fields_bytes = 2 * n * sizeof(real); // StreamfunctionFields: u1 + u2
    report.scratch_fields_bytes =
        5 * n * sizeof(real) + (compact_mac_u_size(grid) + compact_mac_v_size(grid) +
                                compact_mac_w_size(grid)) * sizeof(real);
    report.residual_workspace_bytes = closed_residual_workspace_bytes(n);
    report.affine_rhs_workspace_bytes = closed_affine_rhs_bytes(n);
    report.pcg_workspace_bytes = closed_pcg_workspace_bytes(n);
    report.mg_hierarchy_bytes = closed_mg_hierarchy_bytes(levels);
    report.mg_preconditioner_bytes = closed_mg_preconditioner_bytes(levels);
    report.diagnostics_workspace_bytes = closed_diagnostics_workspace_bytes(grid);

    report.solve_path_bytes = report.scratch_fields_bytes + report.residual_workspace_bytes +
                              report.affine_rhs_workspace_bytes + report.pcg_workspace_bytes +
                              report.mg_hierarchy_bytes + report.mg_preconditioner_bytes;
    report.diagnostics_path_bytes = report.diagnostics_workspace_bytes;
    report.total_bytes =
        report.fields_bytes + report.solve_path_bytes + report.diagnostics_path_bytes;
    report.fine_grid_equivalent_fields =
        n == 0 ? 0.0
               : static_cast<double>(report.total_bytes) /
                     (static_cast<double>(n) * static_cast<double>(sizeof(real)));
    return report;
}

// Exact-equality field-by-field comparison. Returns the name of the first
// mismatched field, or empty on full agreement.
[[nodiscard]] std::string first_mismatch(const StreamfunctionMemoryReport& a,
                                         const StreamfunctionMemoryReport& b) {
    if (a.fields_bytes != b.fields_bytes) return "fields_bytes";
    if (a.solve_path_bytes != b.solve_path_bytes) return "solve_path_bytes";
    if (a.diagnostics_path_bytes != b.diagnostics_path_bytes) return "diagnostics_path_bytes";
    if (a.total_bytes != b.total_bytes) return "total_bytes";
    if (a.scratch_fields_bytes != b.scratch_fields_bytes) return "scratch_fields_bytes";
    if (a.residual_workspace_bytes != b.residual_workspace_bytes) return "residual_workspace_bytes";
    if (a.affine_rhs_workspace_bytes != b.affine_rhs_workspace_bytes)
        return "affine_rhs_workspace_bytes";
    if (a.pcg_workspace_bytes != b.pcg_workspace_bytes) return "pcg_workspace_bytes";
    if (a.mg_hierarchy_bytes != b.mg_hierarchy_bytes) return "mg_hierarchy_bytes";
    if (a.mg_preconditioner_bytes != b.mg_preconditioner_bytes) return "mg_preconditioner_bytes";
    if (a.diagnostics_workspace_bytes != b.diagnostics_workspace_bytes)
        return "diagnostics_workspace_bytes";
    // fine_grid_equivalent_fields: both sides execute the identical
    // static_cast<double>(total_bytes)/(n*sizeof(real)) expression on equal
    // integer inputs, so bitwise equality is expected, not merely close.
    const auto bits_a = *reinterpret_cast<const std::uint64_t*>(&a.fine_grid_equivalent_fields);
    const auto bits_b = *reinterpret_cast<const std::uint64_t*>(&b.fine_grid_equivalent_fields);
    if (bits_a != bits_b) return "fine_grid_equivalent_fields";
    return "";
}

// ---------------------------------------------------------------------------
// Case 2: exact estimator/actual/closed-form equality at 16^3, 32^3, 64^3.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_estimator_equality() {
    const std::vector<int> sizes{16, 32, 64};
    const StreamfunctionSolverConfig config = valid_config();

    bool pass = true;
    std::ostringstream detail;
    double worst_ffe = 0.0;

    for (const int n : sizes) {
        const Grid3D grid = isotropic_grid(n);

        StreamfunctionFields fields;
        StreamfunctionWorkspace workspace;
        fields.prepare(grid);
        workspace.prepare(grid, config);

        const auto estimate = estimate_streamfunction_memory(grid, config);
        const auto actual = make_streamfunction_memory_report(fields, workspace, grid);
        const auto closed = closed_form_memory_report(grid, config);

        const std::string mismatch_est_actual = first_mismatch(estimate, actual);
        const std::string mismatch_est_closed = first_mismatch(estimate, closed);
        const bool this_pass = mismatch_est_actual.empty() && mismatch_est_closed.empty();
        pass = pass && this_pass;
        worst_ffe = std::max(worst_ffe, estimate.fine_grid_equivalent_fields);

        std::cout << std::setprecision(17) << "api_workspace_estimator_equality grid=" << n << "^3"
                  << " total_bytes=" << estimate.total_bytes
                  << " ffe=" << estimate.fine_grid_equivalent_fields
                  << " mismatch_estimate_vs_actual="
                  << (mismatch_est_actual.empty() ? "none" : mismatch_est_actual)
                  << " mismatch_estimate_vs_closed_form="
                  << (mismatch_est_closed.empty() ? "none" : mismatch_est_closed) << '\n';
        detail << n << "^3:" << (this_pass ? "ok" : "FAIL") << ' ';
    }

    return {pass,
            "api_workspace_estimator_equality",
            "gpu-exact-equality",
            detail.str(),
            worst_ffe,
            0.0,
            "exact ==",
            pass ? "all matched" : "mismatch",
            "estimate_streamfunction_memory(), make_streamfunction_memory_report() of a freshly "
            "prepared fields+workspace pair, and an independent closed-form host reconstruction "
            "agree EXACTLY (bitwise for the double field) on every StreamfunctionMemoryReport "
            "field at 16^3, 32^3, and 64^3"};
}

// ---------------------------------------------------------------------------
// Case 3: pure-host estimate-only agreement at 128^3, 256^3 (no device
// allocation: no StreamfunctionFields/StreamfunctionWorkspace prepare()).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_estimate_only_large() {
    const std::vector<int> sizes{128, 256};
    const StreamfunctionSolverConfig config = valid_config();

    bool pass = true;
    std::ostringstream detail;
    std::size_t bytes_256 = 0;
    double ffe_256 = 0.0;

    for (const int n : sizes) {
        const Grid3D grid = isotropic_grid(n);
        const auto estimate = estimate_streamfunction_memory(grid, config);
        const auto closed = closed_form_memory_report(grid, config);
        const std::string mismatch = first_mismatch(estimate, closed);
        const bool this_pass = mismatch.empty();
        pass = pass && this_pass;

        std::cout << std::setprecision(17) << "api_workspace_estimate_only_large grid=" << n
                  << "^3 total_bytes=" << estimate.total_bytes
                  << " ffe=" << estimate.fine_grid_equivalent_fields
                  << " mismatch=" << (mismatch.empty() ? "none" : mismatch) << '\n';
        detail << n << "^3:" << (this_pass ? "ok" : "FAIL") << ' ';

        if (n == 256) {
            bytes_256 = estimate.total_bytes;
            ffe_256 = estimate.fine_grid_equivalent_fields;
        }
    }

    // Recorded, informational-only comparison against the SF-12 T01 audit's
    // documented 256^3 hard numbers. This is NOT a pass/fail gate: CUB
    // temporary-storage sizes are backend/toolkit-dependent, so the gate is
    // the closed-form-reconstruction equality above; this line only records
    // whether the literal audited number still holds in this environment.
    constexpr std::size_t kAuditedBytes256 = 9070736783ULL;
    constexpr double kAuditedFfe256 = 67.58;
    std::cout << std::setprecision(17)
              << "api_workspace_estimate_only_large audited_256_bytes=" << kAuditedBytes256
              << " observed_256_bytes=" << bytes_256
              << " matches_audit=" << (bytes_256 == kAuditedBytes256 ? "true" : "false")
              << " audited_256_ffe=" << kAuditedFfe256 << " observed_256_ffe=" << ffe_256 << '\n';

    return {pass,
            "api_workspace_estimate_only_large",
            "host-only-closed-form",
            detail.str(),
            static_cast<double>(bytes_256),
            ffe_256,
            "exact == (closed-form)",
            pass ? "all matched" : "mismatch",
            "estimate_streamfunction_memory() at 128^3 and 256^3 (pure host, no device "
            "allocation) agrees exactly with the independent closed-form reconstruction; the "
            "256^3 total_bytes/ffe are additionally recorded against the SF-12 T01 audited "
            "values (9070736783 bytes, ffe~=67.58) for information, not as a pass gate"};
}

// ---------------------------------------------------------------------------
// Case 4: allocation freedom across >=3 repeated full-primitive-chain passes.
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_allocation_freedom() {
    constexpr int kAxis = 32;
    constexpr int kIterations = 3;
    const Grid3D grid = isotropic_grid(kAxis);
    const std::size_t n = grid.num_cells();
    const StreamfunctionSolverConfig config = valid_config();

    CudaContext ctx(0);

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    fields.prepare(grid);
    workspace.prepare(grid, config);

    // Zero u1/u2 (spec: u1=u2=0).
    {
        const std::vector<real> zeros(n, real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(fields.u1_span().data(), zeros.data(), n * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(fields.u2_span().data(), zeros.data(), n * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
    // q = 1 (benchmark gauge, uniform coefficient).
    {
        const std::vector<real> ones(n, real{1});
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(workspace.q().data(), ones.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    }

    // Uniform Darcy field consistent with the benchmark gauge g1=(0,1,0),
    // g2=(0,0,1): v = grad(psi1) x grad(psi2) = (1,0,0). Allocated once,
    // outside the ownership boundary being measured (fields+workspace only).
    DeviceBuffer<real> darcy_u(compact_mac_u_size(grid));
    DeviceBuffer<real> darcy_v(compact_mac_v_size(grid));
    DeviceBuffer<real> darcy_w(compact_mac_w_size(grid));
    {
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(),
                                          ones_u.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(),
                                          zeros_v.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(),
                                          zeros_w.size() * sizeof(real), cudaMemcpyHostToDevice));
    }
    const CompactMacVelocityConstView darcy{DeviceSpan<const real>(darcy_u.span()),
                                            DeviceSpan<const real>(darcy_v.span()),
                                            DeviceSpan<const real>(darcy_w.span())};

    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    NonlinearSourceConfig source_config{};
    source_config.epsilon = config.epsilon;
    source_config.v_rms = real{1};
    source_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        source_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }
    PhysicalDiagnosticsConfig diag_config = config.diagnostics;
    constraints::MeanZeroProjector projector;

    const auto run_full_chain = [&] {
        // (a) affine periodic RHS + diagnostics.
        auto rhs_diag =
            assemble_affine_periodic_rhs(ctx, grid, DeviceSpan<const real>(workspace.q()), gauge,
                                         workspace.rhs1(), workspace.rhs2(),
                                         workspace.affine_rhs_workspace());
        (void)synchronize_affine_rhs_diagnostics(ctx, rhs_diag);

        // (b) coupled residual.
        enqueue_streamfunction_residual(ctx, grid, DeviceSpan<const real>(workspace.q()),
                                        fields.fluctuations(), gauge, config.eta, source_config,
                                        config.histogram, workspace.f1(), workspace.f2(),
                                        workspace.residual_workspace());
        (void)synchronize_streamfunction_residual_report(ctx, grid, config.eta, source_config,
                                                          config.histogram,
                                                          workspace.residual_workspace());

        // (c) physical diagnostics.
        enqueue_streamfunction_physical_diagnostics(ctx, grid, fields.fluctuations(), gauge, darcy,
                                                    diag_config, workspace.v_psi(),
                                                    workspace.diagnostics_workspace());
        (void)synchronize_streamfunction_physical_diagnostics_report(
            ctx, grid, diag_config, workspace.diagnostics_workspace());

        // (d) MG coefficient hierarchy + one projected PCG solve, A x = rhs1.
        multigrid::populate_coefficient_hierarchy(ctx, workspace.hierarchy(),
                                                  DeviceSpan<const real>(workspace.q()));
        operators::LesterPositiveDiffusionOperator A(grid, DeviceSpan<const real>(workspace.q()));
        const std::vector<real> zeros_n(n, real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(workspace.f1().data(), zeros_n.data(), n * sizeof(real),
                                          cudaMemcpyHostToDevice));
        const auto pcg_result = solvers::projected_pcg_solve(
            ctx, A, workspace.preconditioner(), DeviceSpan<const real>(workspace.rhs1()),
            workspace.f1(), config.linear, projector, workspace.pcg_workspace());
        ctx.synchronize();
        return pcg_result;
    };

    const auto observed_pointers = [&] {
        std::vector<const void*> pointers{fields.u1_span().data(),
                                          fields.u2_span().data(),
                                          workspace.q().data(),
                                          workspace.rhs1().data(),
                                          workspace.rhs2().data(),
                                          workspace.f1().data(),
                                          workspace.f2().data(),
                                          workspace.v_psi().u.data(),
                                          workspace.v_psi().v.data(),
                                          workspace.v_psi().w.data()};
        return pointers;
    };

    const auto observed_bytes = [&] {
        return fields.allocated_device_bytes() + workspace.allocated_device_bytes();
    };

    const auto free_bytes = [&] {
        std::size_t free_b = 0, total_b = 0;
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        return free_b;
    };

    // Warmup pass.
    const auto warmup_result = run_full_chain();
    const std::size_t warmup_bytes = observed_bytes();
    const auto warmup_pointers = observed_pointers();
    const std::size_t warmup_free = free_bytes();

    std::cout << "api_workspace_allocation_freedom warmup_converged="
              << (warmup_result.converged ? "true" : "false")
              << " warmup_bytes=" << warmup_bytes << " warmup_free_bytes=" << warmup_free << '\n';

    bool pass = warmup_result.status == solvers::ProjectedPCGStatus::converged ||
               warmup_result.status == solvers::ProjectedPCGStatus::max_iterations;

    int iterations_run = 0;
    for (int iter = 0; iter < kIterations; ++iter) {
        const auto result = run_full_chain();
        const std::size_t bytes = observed_bytes();
        const auto pointers = observed_pointers();
        const std::size_t free_now = free_bytes();

        const bool bytes_ok = bytes == warmup_bytes;
        const bool pointers_ok = pointers == warmup_pointers;
        const bool free_ok = free_now == warmup_free;
        ++iterations_run;

        std::cout << "api_workspace_allocation_freedom iter=" << iter
                  << " converged=" << (result.converged ? "true" : "false")
                  << " bytes=" << bytes << " bytes_ok=" << (bytes_ok ? "true" : "false")
                  << " pointers_ok=" << (pointers_ok ? "true" : "false")
                  << " free_bytes=" << free_now << " free_ok=" << (free_ok ? "true" : "false")
                  << '\n';

        pass = pass && bytes_ok && pointers_ok && free_ok;
    }

    std::ostringstream detail;
    detail << kAxis << "^3, " << kIterations << " repeats after warmup";

    return {pass,
            "api_workspace_allocation_freedom",
            "gpu-no-hot-loop-allocation",
            detail.str(),
            static_cast<double>(warmup_bytes),
            static_cast<double>(warmup_free),
            "identical bytes/pointers/free-memory across repeats",
            std::to_string(iterations_run) + " repeats " + (pass ? "stable" : "UNSTABLE"),
            "after one warmup pass exercising assemble_affine_periodic_rhs, "
            "enqueue_streamfunction_residual, enqueue_streamfunction_physical_diagnostics, "
            "populate_coefficient_hierarchy, and projected_pcg_solve through the owned "
            "workspace, the same sequence repeated >=3 times more shows byte-identical "
            "fields+workspace ownership, byte-identical observed pointers, and byte-identical "
            "cudaMemGetInfo free memory (the codebase allocates only via cudaMalloc, so no "
            "unrelated driver/context noise is expected)"};
}

// ---------------------------------------------------------------------------
// Case 5: re-prepare semantics.
//
// IMPORTANT, DOCUMENTED-BEHAVIOR CAVEAT DISCOVERED WHILE WRITING THIS CASE:
// `StreamfunctionWorkspace.cu`'s file-header comment states "re-preparing
// for a smaller grid keeps the larger capacity" in the context of
// `DeviceBuffer::resize()` never shrinking capacity. That is exactly true
// for every `DeviceBuffer`-backed member (the scratch fields, and every
// sub-workspace's own `DeviceBuffer`s: residual/diagnostics/affine-RHS/PCG).
// It is NOT true for the MG hierarchy/preconditioner: those are held behind
// `std::unique_ptr`/`std::optional` and are fully DESTROYED and
// RECONSTRUCTED (not resized) whenever `grid` or `config.mg` changes
// (`StreamfunctionWorkspace::prepare`'s `mg_already_current` check). So the
// AGGREGATE `allocated_device_bytes()` total CAN shrink across a
// smaller-grid re-prepare, because the MG portion tracks the CURRENT grid
// exactly while the non-MG portion retains its historical high-water-mark
// capacity. This case verifies the precise, narrower claim that actually
// holds (non-MG buffer capacity/pointers never shrink/move) rather than the
// broader aggregate-bytes claim, and records this as a finding, not a defect
// (a stale-grid-sized MG hierarchy could not be used for a correct solve).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_reprepare_semantics() {
    const Grid3D grid32 = isotropic_grid(32);
    const Grid3D grid16 = isotropic_grid(16);
    const StreamfunctionSolverConfig config = valid_config();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    fields.prepare(grid32);
    workspace.prepare(grid32, config);

    const auto non_mg_bytes = [](const StreamfunctionMemoryReport& r) {
        return r.fields_bytes + r.scratch_fields_bytes + r.residual_workspace_bytes +
               r.affine_rhs_workspace_bytes + r.pcg_workspace_bytes + r.diagnostics_workspace_bytes;
    };

    const std::size_t bytes_32_first =
        fields.allocated_device_bytes() + workspace.allocated_device_bytes();
    const std::size_t non_mg_bytes_32_first =
        non_mg_bytes(make_streamfunction_memory_report(fields, workspace, grid32));
    const void* q_ptr_first = workspace.q().data();
    const void* rhs1_ptr_first = workspace.rhs1().data();
    const void* u1_ptr_first = fields.u1_span().data();

    fields.prepare(grid16);
    workspace.prepare(grid16, config);

    const std::size_t bytes_16 =
        fields.allocated_device_bytes() + workspace.allocated_device_bytes();
    const std::size_t non_mg_bytes_16 =
        non_mg_bytes(make_streamfunction_memory_report(fields, workspace, grid16));
    const bool non_mg_bytes_unchanged_16 = non_mg_bytes_16 == non_mg_bytes_32_first;
    const bool pointers_unchanged_16 = workspace.q().data() == q_ptr_first &&
                                       workspace.rhs1().data() == rhs1_ptr_first &&
                                       fields.u1_span().data() == u1_ptr_first;
    const bool prepared_for_16 = workspace.prepared_for(grid16, config) && fields.prepared_for(grid16);

    const auto estimate_16 = estimate_streamfunction_memory(grid16, config);
    const bool estimate_less_than_actual = estimate_16.total_bytes < bytes_16;

    fields.prepare(grid32);
    workspace.prepare(grid32, config);
    const std::size_t bytes_32_second =
        fields.allocated_device_bytes() + workspace.allocated_device_bytes();
    const bool bytes_unchanged_32 = bytes_32_second == bytes_32_first;
    const bool pointers_unchanged_32 =
        workspace.q().data() == q_ptr_first && workspace.rhs1().data() == rhs1_ptr_first &&
        fields.u1_span().data() == u1_ptr_first;

    // Changed config.mg.num_levels triggers hierarchy/preconditioner
    // reconstruction without moving non-MG buffers.
    StreamfunctionSolverConfig config_fewer_levels = config;
    config_fewer_levels.mg.num_levels = 3; // 32 -> 16 -> 8, still valid.
    const void* q_ptr_before_mg_change = workspace.q().data();
    const void* rhs1_ptr_before_mg_change = workspace.rhs1().data();
    workspace.prepare(grid32, config_fewer_levels);
    const bool non_mg_pointers_stable = workspace.q().data() == q_ptr_before_mg_change &&
                                        workspace.rhs1().data() == rhs1_ptr_before_mg_change;
    const bool hierarchy_reconstructed = workspace.hierarchy().num_levels() == 3;

    // Workspace remains usable: one trivial PCG solve.
    CudaContext ctx(0);
    const std::size_t n32 = grid32.num_cells();
    {
        const std::vector<real> ones(n32, real{1});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(workspace.q().data(), ones.data(), n32 * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
    multigrid::populate_coefficient_hierarchy(ctx, workspace.hierarchy(),
                                              DeviceSpan<const real>(workspace.q()));
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    auto rhs_diag = assemble_affine_periodic_rhs(ctx, grid32, DeviceSpan<const real>(workspace.q()),
                                                 gauge, workspace.rhs1(), workspace.rhs2(),
                                                 workspace.affine_rhs_workspace());
    (void)synchronize_affine_rhs_diagnostics(ctx, rhs_diag);
    operators::LesterPositiveDiffusionOperator A(grid32, DeviceSpan<const real>(workspace.q()));
    constraints::MeanZeroProjector projector;
    const std::vector<real> zeros_n(n32, real{0});
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(workspace.f1().data(), zeros_n.data(), n32 * sizeof(real), cudaMemcpyHostToDevice));
    const auto pcg_result = solvers::projected_pcg_solve(
        ctx, A, workspace.preconditioner(), DeviceSpan<const real>(workspace.rhs1()),
        workspace.f1(), config_fewer_levels.linear, projector, workspace.pcg_workspace());
    ctx.synchronize();
    const bool solve_usable = pcg_result.status == solvers::ProjectedPCGStatus::converged ||
                              pcg_result.status == solvers::ProjectedPCGStatus::max_iterations;

    std::cout << std::setprecision(17)
              << "api_workspace_reprepare_semantics bytes_32_first=" << bytes_32_first
              << " non_mg_bytes_32_first=" << non_mg_bytes_32_first << " bytes_16=" << bytes_16
              << " non_mg_bytes_16=" << non_mg_bytes_16 << " bytes_32_second=" << bytes_32_second
              << " estimate_16=" << estimate_16.total_bytes
              << " hierarchy_levels_after_change=" << workspace.hierarchy().num_levels()
              << " solve_status=" << static_cast<int>(pcg_result.status) << '\n';

    const bool pass = non_mg_bytes_unchanged_16 && pointers_unchanged_16 && prepared_for_16 &&
                      estimate_less_than_actual && bytes_unchanged_32 && pointers_unchanged_32 &&
                      non_mg_pointers_stable && hierarchy_reconstructed && solve_usable;

    std::ostringstream observed;
    observed << "bytes:" << bytes_32_first << "->" << bytes_16 << "->" << bytes_32_second
             << " non_mg_bytes:" << non_mg_bytes_32_first << "->" << non_mg_bytes_16;

    return {pass,
            "api_workspace_reprepare_semantics",
            "gpu-non-mg-capacity-never-shrinks",
            "32^3 -> 16^3 -> 32^3, then mg.num_levels 4->3",
            static_cast<double>(bytes_32_first),
            static_cast<double>(bytes_16),
            "non-MG capacity constant; pointers stable; estimate(16^3) < actual",
            observed.str(),
            "re-preparing for a smaller grid keeps every non-MG DeviceBuffer's allocated "
            "capacity and pointer (fields, scratch fields, residual/diagnostics/affine-RHS/PCG "
            "sub-workspaces); prepared_for() reports true for the smaller grid; the smaller "
            "grid's estimate is strictly less than the still-larger actual allocation. The MG "
            "hierarchy/preconditioner, held behind unique_ptr/optional, are fully reconstructed "
            "(not resized) whenever grid or config.mg changes, so the AGGREGATE byte total can "
            "shrink across a smaller-grid re-prepare -- a documented finding distinct from the "
            "narrower, verified non-MG-capacity-never-shrinks claim (see the case's top "
            "comment). Re-preparing back at the identical original (grid, config) restores the "
            "aggregate total exactly. Changing config.mg.num_levels reconstructs only the MG "
            "hierarchy/preconditioner (non-MG buffer pointers stay fixed) and the workspace "
            "remains usable for a real PCG solve"};
}

// ---------------------------------------------------------------------------
// Case 6: 256^3 budget report (informational + consistency; documents the
// expected over-budget rollback-policy outcome).
// ---------------------------------------------------------------------------

[[nodiscard]] CaseResult case_api_workspace_budget_report() {
    constexpr double kPlanBudgetFields = 24.6;
    const Grid3D grid = isotropic_grid(256);
    const StreamfunctionSolverConfig config = valid_config();
    const std::size_t n = grid.num_cells();
    const double n_field_bytes = static_cast<double>(n) * static_cast<double>(sizeof(real));

    const auto estimate = estimate_streamfunction_memory(grid, config);
    const auto closed = closed_form_memory_report(grid, config);
    const std::string mismatch = first_mismatch(estimate, closed);
    const bool consistent = mismatch.empty();

    const double fields_ffe = static_cast<double>(estimate.fields_bytes) / n_field_bytes;
    const double scratch_ffe = static_cast<double>(estimate.scratch_fields_bytes) / n_field_bytes;
    const double residual_ffe = static_cast<double>(estimate.residual_workspace_bytes) / n_field_bytes;
    const double affine_rhs_ffe =
        static_cast<double>(estimate.affine_rhs_workspace_bytes) / n_field_bytes;
    const double pcg_ffe = static_cast<double>(estimate.pcg_workspace_bytes) / n_field_bytes;
    const double mg_hierarchy_ffe = static_cast<double>(estimate.mg_hierarchy_bytes) / n_field_bytes;
    const double mg_precond_ffe =
        static_cast<double>(estimate.mg_preconditioner_bytes) / n_field_bytes;
    const double diagnostics_ffe =
        static_cast<double>(estimate.diagnostics_workspace_bytes) / n_field_bytes;
    const double solve_path_ffe = static_cast<double>(estimate.solve_path_bytes) / n_field_bytes;
    const double total_ffe = estimate.fine_grid_equivalent_fields;

    std::cout << std::setprecision(10)
              << "api_workspace_budget_report grid=256^3"
              << " fields_ffe=" << fields_ffe << " scratch_ffe=" << scratch_ffe
              << " residual_ffe=" << residual_ffe << " affine_rhs_ffe=" << affine_rhs_ffe
              << " pcg_ffe=" << pcg_ffe << " mg_hierarchy_ffe=" << mg_hierarchy_ffe
              << " mg_precond_ffe=" << mg_precond_ffe << " diagnostics_ffe=" << diagnostics_ffe
              << " solve_path_ffe(excl. diagnostics)=" << solve_path_ffe
              << " total_ffe=" << total_ffe << " plan_budget_ffe=" << kPlanBudgetFields
              << " over_budget=" << (total_ffe > kPlanBudgetFields ? "true" : "false") << '\n';

    const bool over_budget = total_ffe > kPlanBudgetFields;
    const bool pass = consistent && over_budget;

    std::ostringstream detail;
    detail << "fields=" << fields_ffe << " scratch=" << scratch_ffe << " residual=" << residual_ffe
           << " affine_rhs=" << affine_rhs_ffe << " pcg=" << pcg_ffe
           << " mg_hierarchy=" << mg_hierarchy_ffe << " mg_precond=" << mg_precond_ffe
           << " diagnostics=" << diagnostics_ffe;

    return {pass,
            "api_workspace_budget_report",
            "informational-and-recorded-over-budget",
            detail.str(),
            total_ffe,
            kPlanBudgetFields,
            "> plan budget (24.6)",
            std::to_string(total_ffe),
            "the categorized 256^3 fine-grid-equivalent field budget is computed and consistent "
            "with the closed-form reconstruction, and the total (~67.6) is explicitly recorded "
            "as EXCEEDING the plan's ~24.6-field budget -- the spec's documented rollback-policy "
            "outcome (record each extra field and redesign ownership before SF-13), not a "
            "target this case pretends is met"};
}

} // namespace

CaseRegistry api_workspace_case_registry() {
    return {{"api_workspace_error_contract", case_api_workspace_error_contract},
            {"api_workspace_estimator_equality", case_api_workspace_estimator_equality},
            {"api_workspace_estimate_only_large", case_api_workspace_estimate_only_large},
            {"api_workspace_allocation_freedom", case_api_workspace_allocation_freedom},
            {"api_workspace_reprepare_semantics", case_api_workspace_reprepare_semantics},
            {"api_workspace_budget_report", case_api_workspace_budget_report}};
}

} // namespace macroflow3d::streamfunctions::test
