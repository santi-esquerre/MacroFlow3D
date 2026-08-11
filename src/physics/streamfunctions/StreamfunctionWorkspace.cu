#include "StreamfunctionWorkspace.cuh"

#include "../../numerics/blas/sum.cuh"
#include "../../multigrid/cycle/projected_positive_v_cycle.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace {

std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

bool grids_equal(const Grid3D& a, const Grid3D& b) noexcept {
    return a.nx == b.nx && a.ny == b.ny && a.nz == b.nz && a.dx == b.dx && a.dy == b.dy &&
           a.dz == b.dz;
}

bool mg_config_equal(const multigrid::MGConfig& a, const multigrid::MGConfig& b) noexcept {
    return a.num_levels == b.num_levels && a.pre_smooth == b.pre_smooth &&
           a.post_smooth == b.post_smooth && a.coarse_solve_iters == b.coarse_solve_iters &&
           a.check_convergence_every == b.check_convergence_every && a.omega == b.omega &&
           a.verbose == b.verbose;
}

// Replicates multigrid::MGHierarchy(finest, num_levels)'s coarsening/break
// rule exactly (see src/multigrid/mg_types.hpp), without allocating any
// device buffer, so validation and the memory estimator cannot drift from
// the actual hierarchy that prepare() constructs.
std::vector<Grid3D> simulate_mg_hierarchy_levels(const Grid3D& finest, int num_levels) {
    std::vector<Grid3D> levels;
    if (num_levels <= 0) {
        return levels;
    }
    levels.reserve(static_cast<std::size_t>(num_levels));
    Grid3D current = finest;
    for (int l = 0; l < num_levels; ++l) {
        levels.push_back(current);
        if (l < num_levels - 1) {
            current.nx = current.nx / 2;
            current.ny = current.ny / 2;
            current.nz = current.nz / 2;
            current.dx = current.dx * real{2};
            current.dy = current.dy * real{2};
            current.dz = current.dz * real{2};
            if (current.nx < 2 || current.ny < 2 || current.nz < 2) {
                break;
            }
        }
    }
    return levels;
}

// True only if `finest` supports exactly `mg_config.num_levels` levels under
// the exact simulate_mg_hierarchy_levels rule AND every resulting level
// satisfies multigrid::validate_projected_positive_hierarchy's per-level
// even-extent requirement (finite positive isotropic spacing is checked
// separately by the caller before this is invoked).
bool grid_supports_mg_levels(const Grid3D& finest, const multigrid::MGConfig& mg_config) {
    if (mg_config.num_levels <= 0) {
        return false;
    }
    const auto levels = simulate_mg_hierarchy_levels(finest, mg_config.num_levels);
    if (static_cast<int>(levels.size()) != mg_config.num_levels) {
        return false;
    }
    for (const auto& level : levels) {
        if (level.nx < 2 || level.ny < 2 || level.nz < 2) {
            return false;
        }
        if (level.nx % 2 != 0 || level.ny % 2 != 0 || level.nz % 2 != 0) {
            return false;
        }
    }
    return true;
}

// --- Byte accounting for public-member accepted types (PCGWorkspace,
// ProjectedPCGWorkspace, MGLevel, MGHierarchy). These are computed here,
// colocated with StreamfunctionWorkspace::prepare(), rather than as new
// accessors on those types, per the increment spec.

std::size_t pcg_workspace_allocated_bytes(const solvers::PCGWorkspace& ws) noexcept {
    return (ws.r.capacity() + ws.z.capacity() + ws.p.capacity() + ws.Ap.capacity()) *
               sizeof(real) +
           blas::reduction_workspace_allocated_bytes(ws.red);
}

std::size_t pcg_workspace_estimate_bytes(std::size_t n) {
    // Mirrors PCGWorkspace::ensure(n): r, z, p, Ap resized to exactly n.
    // `red` is only ever driven through blas::nrm2_host/dot_host (cuBLAS),
    // which never touches `red.temp`; its default-constructed d_scalar
    // (one element) is the only owned storage.
    return 4 * n * sizeof(real) + sizeof(real);
}

std::size_t projected_pcg_workspace_allocated_bytes(
    const solvers::ProjectedPCGWorkspace& ws) noexcept {
    return pcg_workspace_allocated_bytes(ws.pcg) + ws.projected_rhs.capacity() * sizeof(real) +
           ws.mean_zero.allocated_device_bytes();
}

std::size_t projected_pcg_workspace_estimate_bytes(std::size_t n) {
    // Mirrors ProjectedPCGWorkspace::prepare(n): pcg.ensure(n),
    // projected_rhs resized to n, mean_zero.prepare(n).
    return pcg_workspace_estimate_bytes(n) + n * sizeof(real) +
           constraints::MeanZeroWorkspace::estimate_device_bytes(n);
}

std::size_t mg_level_allocated_bytes(const multigrid::MGLevel& level) noexcept {
    return (level.x.capacity() + level.b.capacity() + level.r.capacity() +
            level.coefficient.capacity()) *
           sizeof(real);
}

std::size_t mg_hierarchy_allocated_bytes(const multigrid::MGHierarchy& hierarchy) noexcept {
    std::size_t bytes = 0;
    for (const auto& level : hierarchy.levels) {
        bytes += mg_level_allocated_bytes(level);
    }
    return bytes;
}

std::size_t mg_hierarchy_estimate_bytes(const std::vector<Grid3D>& levels) {
    // Mirrors MGLevel::MGLevel/ensure: x, b, r, coefficient each resized to
    // exactly level.num_cells().
    std::size_t bytes = 0;
    for (const auto& level : levels) {
        bytes += 4 * level.num_cells() * sizeof(real);
    }
    return bytes;
}

std::size_t mg_preconditioner_estimate_bytes(const std::vector<Grid3D>& levels) {
    // Mirrors ProjectedPositiveMGPreconditioner's constructor: one
    // MeanZeroWorkspace prepared per hierarchy level.
    std::size_t bytes = 0;
    for (const auto& level : levels) {
        bytes += constraints::MeanZeroWorkspace::estimate_device_bytes(level.num_cells());
    }
    return bytes;
}

// --- Shared config validation (mirrors the accepted sub-modules' own
// require_valid_config rules exactly; see StreamfunctionTypes.hpp).

void require_valid_grid_geometry(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2) {
        throw std::invalid_argument(
            "Streamfunction problem requires every grid dimension to be at least 2");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires finite, strictly positive grid spacing");
    }
    if (grid.dx != grid.dy || grid.dx != grid.dz) {
        throw std::invalid_argument(
            "Streamfunction problem currently requires isotropic grid spacing (dx == dy == dz), "
            "inherited from the accepted SF-02/SF-06/SF-10 evaluator chain; SF-11 diagnostics "
            "alone remain anisotropy-capable but are not exercised through this restriction here");
    }
}

void require_valid_mg_config(const multigrid::MGConfig& mg, const Grid3D& grid) {
    if (mg.pre_smooth < 1 || mg.post_smooth < 1 || mg.pre_smooth != mg.post_smooth ||
        mg.coarse_solve_iters <= 0) {
        throw std::invalid_argument(
            "Streamfunction problem requires equal positive MG pre/post sweeps and a positive "
            "coarse solve iteration count");
    }
    if (!grid_supports_mg_levels(grid, mg)) {
        throw std::invalid_argument(
            "Streamfunction problem grid does not support config.mg.num_levels under the exact "
            "multigrid::MGHierarchy coarsening/break rule and "
            "multigrid::validate_projected_positive_hierarchy's per-level even-extent "
            "requirements");
    }
}

void require_valid_linear_config(const solvers::ProjectedPCGConfig& linear) {
    if (linear.max_iter < 0 || linear.check_every <= 0 || linear.rtol < real{0} ||
        !std::isfinite(linear.rtol)) {
        throw std::invalid_argument(
            "Streamfunction problem requires a non-negative linear.max_iter, a strictly "
            "positive linear.check_every, and a finite non-negative linear.rtol");
    }
}

void require_valid_histogram_config(const ResidualHistogramConfig& histogram) {
    if (!std::isfinite(histogram.c_min_rel) || !std::isfinite(histogram.c_max_rel) ||
        histogram.c_min_rel <= real{0} || histogram.c_max_rel <= histogram.c_min_rel) {
        throw std::invalid_argument(
            "Streamfunction problem histogram config requires finite 0 < c_min_rel < c_max_rel");
    }
}

void require_valid_diagnostics_config(const PhysicalDiagnosticsConfig& diagnostics) {
    if (!std::isfinite(diagnostics.angle_exclusion_rel) ||
        diagnostics.angle_exclusion_rel <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite, strictly positive "
            "diagnostics.angle_exclusion_rel");
    }
    if (!std::isfinite(diagnostics.low_speed_rel) || diagnostics.low_speed_rel <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite, strictly positive "
            "diagnostics.low_speed_rel");
    }
    if (diagnostics.num_degeneracy_thresholds < 0 ||
        diagnostics.num_degeneracy_thresholds > kMaxDegeneracyThresholds) {
        throw std::invalid_argument(
            "Streamfunction problem requires diagnostics.num_degeneracy_thresholds within "
            "bounds");
    }
    real previous = real{0};
    for (int t = 0; t < diagnostics.num_degeneracy_thresholds; ++t) {
        const real tau = diagnostics.degeneracy_thresholds[t];
        if (!std::isfinite(tau) || tau <= real{0}) {
            throw std::invalid_argument(
                "Streamfunction problem requires finite, strictly positive "
                "diagnostics.degeneracy_thresholds");
        }
        if (t > 0 && tau <= previous) {
            throw std::invalid_argument(
                "Streamfunction problem requires strictly increasing "
                "diagnostics.degeneracy_thresholds");
        }
        previous = tau;
    }
}

void require_valid_picard_config(const FixedPicardConfig& picard) {
    if (picard.max_iter < 0) {
        throw std::invalid_argument(
            "Streamfunction problem requires a non-negative picard.max_iter");
    }
    if (!std::isfinite(picard.tolerance) || picard.tolerance <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite, strictly positive picard.tolerance");
    }
    if (!std::isfinite(picard.omega) || picard.omega <= real{0} || picard.omega > real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite picard.omega in (0, 1]");
    }
}

// SF-15: validated unconditionally (regardless of adaptive.enabled) -- an
// invalid adaptive config is invalid even when the globalization is off.
void require_valid_adaptive_config(const AdaptivePicardConfig& adaptive,
                                   const FixedPicardConfig& picard) {
    if (!std::isfinite(adaptive.omega_min) || adaptive.omega_min <= real{0} ||
        adaptive.omega_min > picard.omega) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.omega_min in (0, picard.omega]");
    }
    if (!std::isfinite(adaptive.backtrack_factor) || adaptive.backtrack_factor <= real{0} ||
        adaptive.backtrack_factor >= real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.backtrack_factor in (0, 1)");
    }
    if (!std::isfinite(adaptive.growth_factor) || adaptive.growth_factor < real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.growth_factor >= 1");
    }
    if (!std::isfinite(adaptive.omega_max) || adaptive.omega_max < picard.omega ||
        adaptive.omega_max > real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.omega_max in [picard.omega, 1]");
    }
    if (adaptive.easy_streak < 1) {
        throw std::invalid_argument(
            "Streamfunction problem requires a strictly positive adaptive.easy_streak");
    }
    if (!std::isfinite(adaptive.armijo_c) || adaptive.armijo_c < real{0} ||
        adaptive.armijo_c >= real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.armijo_c in [0, 1)");
    }
    if (adaptive.stagnation_window < 1) {
        throw std::invalid_argument(
            "Streamfunction problem requires a strictly positive adaptive.stagnation_window");
    }
    if (!std::isfinite(adaptive.stagnation_min_reduction) ||
        adaptive.stagnation_min_reduction <= real{0} ||
        adaptive.stagnation_min_reduction >= real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.stagnation_min_reduction in "
            "(0, 1)");
    }
    if (!std::isfinite(adaptive.max_unexplained_fraction) ||
        adaptive.max_unexplained_fraction <= real{0} ||
        adaptive.max_unexplained_fraction > real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.max_unexplained_fraction in "
            "(0, 1]");
    }
    if (!std::isfinite(adaptive.unexplained_growth_factor) ||
        adaptive.unexplained_growth_factor < real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.unexplained_growth_factor >= 1");
    }
    if (!std::isfinite(adaptive.unexplained_growth_offset) ||
        adaptive.unexplained_growth_offset < real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite, non-negative "
            "adaptive.unexplained_growth_offset");
    }
    if (!std::isfinite(adaptive.percentile_collapse_factor) ||
        adaptive.percentile_collapse_factor <= real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite adaptive.percentile_collapse_factor > 1");
    }
}

// SF-20: validated unconditionally (regardless of anderson.enabled), mirroring
// require_valid_adaptive_config's convention -- an invalid Anderson config is
// invalid even when acceleration is off.
void require_valid_anderson_config(const AndersonConfig& anderson) {
    if (anderson.depth < 3 || anderson.depth > 8) {
        throw std::invalid_argument(
            "Streamfunction problem requires anderson.depth in [3, 8]");
    }
    if (anderson.start_iteration < 1) {
        throw std::invalid_argument(
            "Streamfunction problem requires a strictly positive anderson.start_iteration");
    }
    if (!std::isfinite(anderson.condition_limit) || anderson.condition_limit <= real{1}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite anderson.condition_limit > 1");
    }
}

void require_valid_source_side_config(const StreamfunctionSolverConfig& config) {
    if (!std::isfinite(config.eta) || config.eta < real{0}) {
        throw std::invalid_argument("Streamfunction problem requires a finite, non-negative eta");
    }
    if (!std::isfinite(config.epsilon) || config.epsilon < real{0}) {
        throw std::invalid_argument(
            "Streamfunction problem requires a finite, non-negative epsilon");
    }
    if (config.num_degeneracy_thresholds < 0 ||
        config.num_degeneracy_thresholds > kMaxDegeneracyThresholds) {
        throw std::invalid_argument(
            "Streamfunction problem requires num_degeneracy_thresholds within bounds");
    }
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        const real tau = config.degeneracy_thresholds[t];
        if (!std::isfinite(tau) || tau < real{0}) {
            throw std::invalid_argument(
                "Streamfunction problem requires finite, non-negative degeneracy_thresholds");
        }
    }
}

} // namespace

void validate_streamfunction_problem(const Grid3D& grid, const StreamfunctionProblemView& problem,
                                      const StreamfunctionSolverConfig& config) {
    require_valid_grid_geometry(grid);
    if (!grids_equal(problem.grid, grid)) {
        throw std::invalid_argument(
            "Streamfunction problem view's embedded grid must exactly match the grid argument "
            "(extents and spacings)");
    }

    const std::size_t n = grid.num_cells();
    if (problem.conductivity.size() != n) {
        throw std::invalid_argument(
            "Streamfunction problem conductivity span must have exactly grid.num_cells() "
            "elements");
    }
    if (problem.darcy_velocity.u.size() != compact_mac_u_size(grid) ||
        problem.darcy_velocity.v.size() != compact_mac_v_size(grid) ||
        problem.darcy_velocity.w.size() != compact_mac_w_size(grid)) {
        throw std::invalid_argument(
            "Streamfunction problem reference Darcy velocity must have exact CompactMAC face "
            "sizes ((nx+1)*ny*nz, nx*(ny+1)*nz, nx*ny*(nz+1))");
    }
    if (!problem.bc.is_periodic_x() || !problem.bc.is_periodic_y() || !problem.bc.is_periodic_z()) {
        throw std::invalid_argument(
            "Streamfunction problem requires a triply periodic BCSpec; the v1 benchmark supports "
            "no other boundary condition");
    }
    if (!std::isfinite(problem.gauge.psi1_gradient.x) ||
        !std::isfinite(problem.gauge.psi1_gradient.y) ||
        !std::isfinite(problem.gauge.psi1_gradient.z) ||
        !std::isfinite(problem.gauge.psi2_gradient.x) ||
        !std::isfinite(problem.gauge.psi2_gradient.y) ||
        !std::isfinite(problem.gauge.psi2_gradient.z)) {
        throw std::invalid_argument(
            "Streamfunction problem requires finite affine gauge gradient components");
    }

    require_valid_mg_config(config.mg, grid);
    require_valid_linear_config(config.linear);
    require_valid_histogram_config(config.histogram);
    require_valid_diagnostics_config(config.diagnostics);
    require_valid_picard_config(config.picard);
    require_valid_adaptive_config(config.adaptive, config.picard);
    require_valid_anderson_config(config.anderson);
    require_valid_source_side_config(config);
}

void StreamfunctionFields::prepare(const Grid3D& grid) {
    const std::size_t n = grid.num_cells();
    u1_.resize(n);
    u2_.resize(n);
    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
}

bool StreamfunctionFields::prepared_for(const Grid3D& grid) const noexcept {
    const std::size_t n = grid.num_cells();
    return nx_ == grid.nx && ny_ == grid.ny && nz_ == grid.nz && u1_.size() == n &&
           u2_.size() == n;
}

PeriodicStreamfunctionFluctuations StreamfunctionFields::fluctuations() {
    if (u1_.size() == 0 || u2_.size() == 0) {
        throw std::logic_error("StreamfunctionFields::fluctuations() requires a preceding prepare()");
    }
    return PeriodicStreamfunctionFluctuations{u1_.span(), u2_.span()};
}

std::size_t StreamfunctionFields::allocated_device_bytes() const noexcept {
    return (u1_.capacity() + u2_.capacity()) * sizeof(real);
}

void StreamfunctionWorkspace::ensure_prepared() const {
    if (!prepared_) {
        throw std::logic_error("StreamfunctionWorkspace requires a preceding prepare() call");
    }
}

void StreamfunctionWorkspace::prepare(const Grid3D& grid, const StreamfunctionSolverConfig& config) {
    require_valid_grid_geometry(grid);
    require_valid_mg_config(config.mg, grid);

    const std::size_t n = grid.num_cells();
    q_.resize(n);
    rhs1_.resize(n);
    rhs2_.resize(n);
    f1_.resize(n);
    f2_.resize(n);

    u_trial1_.resize(n);
    u_trial2_.resize(n);

    v_psi_u_.resize(compact_mac_u_size(grid));
    v_psi_v_.resize(compact_mac_v_size(grid));
    v_psi_w_.resize(compact_mac_w_size(grid));

    residual_workspace_.prepare(n);
    diagnostics_workspace_.prepare(grid);
    affine_rhs_workspace_.prepare(n);
    pcg_workspace_.prepare(n);

    const bool mg_already_current = mg_hierarchy_ != nullptr && mg_preconditioner_.has_value() &&
                                    grids_equal(prepared_grid_, grid) &&
                                    mg_config_equal(prepared_mg_config_, config.mg);
    if (!mg_already_current) {
        // MGHierarchy/ProjectedPositiveMGPreconditioner cannot be resized in
        // place; reconstruct only when the grid or config.mg actually
        // changed. The preconditioner's own constructor re-validates the
        // hierarchy via multigrid::validate_projected_positive_hierarchy.
        mg_preconditioner_.reset();
        mg_hierarchy_ = std::make_unique<multigrid::MGHierarchy>(grid, config.mg.num_levels);
        mg_preconditioner_.emplace(*mg_hierarchy_, config.mg);
    }

    // SF-20: optional Anderson accelerator, allocated ONLY when enabled (see
    // StreamfunctionWorkspace.cuh and AndersonAccelerator.cuh). Disabling
    // deallocates it, matching the "allocated only when enabled" contract;
    // an already-prepared (n, depth) pair performs no device allocation.
    if (config.anderson.enabled) {
        if (!anderson_.has_value()) {
            anderson_.emplace();
        }
        if (!anderson_->prepared_for(n, config.anderson.depth)) {
            anderson_->prepare(n, config.anderson.depth);
        }
    } else {
        anderson_.reset();
    }

    prepared_grid_ = grid;
    prepared_mg_config_ = config.mg;
    prepared_ = true;
}

bool StreamfunctionWorkspace::prepared_for(const Grid3D& grid,
                                           const StreamfunctionSolverConfig& config) const noexcept {
    if (!prepared_) {
        return false;
    }
    const std::size_t n = grid.num_cells();
    const bool anderson_ok = config.anderson.enabled
                                 ? (anderson_.has_value() &&
                                    anderson_->prepared_for(n, config.anderson.depth))
                                 : !anderson_.has_value();
    return q_.size() == n && rhs1_.size() == n && rhs2_.size() == n && f1_.size() == n &&
           f2_.size() == n && u_trial1_.size() == n && u_trial2_.size() == n &&
           v_psi_u_.size() == compact_mac_u_size(grid) &&
           v_psi_v_.size() == compact_mac_v_size(grid) &&
           v_psi_w_.size() == compact_mac_w_size(grid) && residual_workspace_.prepared_for(n) &&
           diagnostics_workspace_.prepared_for(grid) && affine_rhs_workspace_.prepared_for(n) &&
           pcg_workspace_.prepared_for(n) && mg_hierarchy_ != nullptr &&
           mg_preconditioner_.has_value() && grids_equal(prepared_grid_, grid) &&
           mg_config_equal(prepared_mg_config_, config.mg) && anderson_ok;
}

DeviceSpan<real> StreamfunctionWorkspace::q() {
    ensure_prepared();
    return q_.span();
}
DeviceSpan<const real> StreamfunctionWorkspace::q() const {
    ensure_prepared();
    return q_.span();
}
DeviceSpan<real> StreamfunctionWorkspace::rhs1() {
    ensure_prepared();
    return rhs1_.span();
}
DeviceSpan<real> StreamfunctionWorkspace::rhs2() {
    ensure_prepared();
    return rhs2_.span();
}
DeviceSpan<real> StreamfunctionWorkspace::f1() {
    ensure_prepared();
    return f1_.span();
}
DeviceSpan<real> StreamfunctionWorkspace::f2() {
    ensure_prepared();
    return f2_.span();
}

DeviceSpan<real> StreamfunctionWorkspace::u_trial1() {
    ensure_prepared();
    return u_trial1_.span();
}
DeviceSpan<real> StreamfunctionWorkspace::u_trial2() {
    ensure_prepared();
    return u_trial2_.span();
}
DeviceSpan<const real> StreamfunctionWorkspace::u_trial1() const {
    ensure_prepared();
    return u_trial1_.span();
}
DeviceSpan<const real> StreamfunctionWorkspace::u_trial2() const {
    ensure_prepared();
    return u_trial2_.span();
}

CompactMacVelocityView StreamfunctionWorkspace::v_psi() {
    ensure_prepared();
    return CompactMacVelocityView{v_psi_u_.span(), v_psi_v_.span(), v_psi_w_.span()};
}
CompactMacVelocityConstView StreamfunctionWorkspace::v_psi() const {
    ensure_prepared();
    return CompactMacVelocityConstView{v_psi_u_.span(), v_psi_v_.span(), v_psi_w_.span()};
}

StreamfunctionResidualWorkspace& StreamfunctionWorkspace::residual_workspace() {
    ensure_prepared();
    return residual_workspace_;
}
StreamfunctionDiagnosticsWorkspace& StreamfunctionWorkspace::diagnostics_workspace() {
    ensure_prepared();
    return diagnostics_workspace_;
}
AffinePeriodicRhsWorkspace& StreamfunctionWorkspace::affine_rhs_workspace() {
    ensure_prepared();
    return affine_rhs_workspace_;
}
solvers::ProjectedPCGWorkspace& StreamfunctionWorkspace::pcg_workspace() {
    ensure_prepared();
    return pcg_workspace_;
}

multigrid::MGHierarchy& StreamfunctionWorkspace::hierarchy() {
    ensure_prepared();
    return *mg_hierarchy_;
}
const multigrid::MGHierarchy& StreamfunctionWorkspace::hierarchy() const {
    ensure_prepared();
    return *mg_hierarchy_;
}
solvers::ProjectedPositiveMGPreconditioner& StreamfunctionWorkspace::preconditioner() {
    ensure_prepared();
    return *mg_preconditioner_;
}
const solvers::ProjectedPositiveMGPreconditioner& StreamfunctionWorkspace::preconditioner() const {
    ensure_prepared();
    return *mg_preconditioner_;
}

AndersonAccelerator& StreamfunctionWorkspace::anderson() {
    ensure_prepared();
    if (!anderson_.has_value()) {
        throw std::logic_error(
            "StreamfunctionWorkspace::anderson() requires config.anderson.enabled at the last "
            "prepare() call");
    }
    return *anderson_;
}
bool StreamfunctionWorkspace::anderson_enabled() const noexcept {
    return anderson_.has_value();
}

std::size_t StreamfunctionWorkspace::allocated_device_bytes() const noexcept {
    std::size_t bytes =
        (q_.capacity() + rhs1_.capacity() + rhs2_.capacity() + f1_.capacity() + f2_.capacity() +
         u_trial1_.capacity() + u_trial2_.capacity() + v_psi_u_.capacity() + v_psi_v_.capacity() +
         v_psi_w_.capacity()) *
        sizeof(real);
    bytes += residual_workspace_.allocated_device_bytes();
    bytes += diagnostics_workspace_.allocated_device_bytes();
    bytes += affine_rhs_workspace_.allocated_device_bytes();
    bytes += projected_pcg_workspace_allocated_bytes(pcg_workspace_);
    if (mg_hierarchy_) {
        bytes += mg_hierarchy_allocated_bytes(*mg_hierarchy_);
    }
    if (mg_preconditioner_.has_value()) {
        bytes += mg_preconditioner_->allocated_device_bytes();
    }
    if (anderson_.has_value()) {
        bytes += anderson_->history_bytes() + anderson_->staging_bytes() +
                 anderson_->scratch_bytes();
    }
    return bytes;
}

StreamfunctionMemoryReport StreamfunctionWorkspace::memory_report(const Grid3D& grid) const {
    StreamfunctionMemoryReport report;

    report.scratch_fields_bytes =
        (q_.capacity() + rhs1_.capacity() + rhs2_.capacity() + f1_.capacity() + f2_.capacity() +
         u_trial1_.capacity() + u_trial2_.capacity() + v_psi_u_.capacity() + v_psi_v_.capacity() +
         v_psi_w_.capacity()) *
        sizeof(real);
    report.residual_workspace_bytes = residual_workspace_.allocated_device_bytes();
    report.affine_rhs_workspace_bytes = affine_rhs_workspace_.allocated_device_bytes();
    report.pcg_workspace_bytes = projected_pcg_workspace_allocated_bytes(pcg_workspace_);
    report.mg_hierarchy_bytes = mg_hierarchy_ ? mg_hierarchy_allocated_bytes(*mg_hierarchy_) : 0;
    report.mg_preconditioner_bytes =
        mg_preconditioner_.has_value() ? mg_preconditioner_->allocated_device_bytes() : 0;
    report.diagnostics_workspace_bytes = diagnostics_workspace_.allocated_device_bytes();

    // SF-20: additive, exactly zero when disabled (see StreamfunctionTypes.hpp).
    if (anderson_.has_value()) {
        report.anderson_history_bytes = anderson_->history_bytes();
        report.anderson_staging_bytes = anderson_->staging_bytes();
        report.anderson_scratch_bytes = anderson_->scratch_bytes();
    }

    report.fields_bytes = 0;
    report.solve_path_bytes = report.scratch_fields_bytes + report.residual_workspace_bytes +
                              report.affine_rhs_workspace_bytes + report.pcg_workspace_bytes +
                              report.mg_hierarchy_bytes + report.mg_preconditioner_bytes +
                              report.anderson_history_bytes + report.anderson_staging_bytes +
                              report.anderson_scratch_bytes;
    report.diagnostics_path_bytes = report.diagnostics_workspace_bytes;
    report.total_bytes = report.fields_bytes + report.solve_path_bytes + report.diagnostics_path_bytes;

    const std::size_t n = grid.num_cells();
    report.fine_grid_equivalent_fields =
        n == 0 ? 0.0
               : static_cast<double>(report.total_bytes) /
                     (static_cast<double>(n) * static_cast<double>(sizeof(real)));

    return report;
}

StreamfunctionMemoryReport make_streamfunction_memory_report(
    const StreamfunctionFields& fields, const StreamfunctionWorkspace& workspace,
    const Grid3D& grid) {
    StreamfunctionMemoryReport report = workspace.memory_report(grid);
    report.fields_bytes = fields.allocated_device_bytes();
    report.total_bytes += report.fields_bytes;

    const std::size_t n = grid.num_cells();
    report.fine_grid_equivalent_fields =
        n == 0 ? 0.0
               : static_cast<double>(report.total_bytes) /
                     (static_cast<double>(n) * static_cast<double>(sizeof(real)));
    return report;
}

StreamfunctionMemoryReport estimate_streamfunction_memory(const Grid3D& grid,
                                                           const StreamfunctionSolverConfig& config) {
    require_valid_grid_geometry(grid);
    require_valid_mg_config(config.mg, grid);
    require_valid_linear_config(config.linear);
    require_valid_histogram_config(config.histogram);
    require_valid_diagnostics_config(config.diagnostics);
    require_valid_picard_config(config.picard);
    require_valid_adaptive_config(config.adaptive, config.picard);
    require_valid_anderson_config(config.anderson);
    require_valid_source_side_config(config);

    const std::size_t n = grid.num_cells();
    const auto mg_levels = simulate_mg_hierarchy_levels(grid, config.mg.num_levels);

    StreamfunctionMemoryReport report;
    report.fields_bytes = 2 * n * sizeof(real); // StreamfunctionFields: u1 + u2

    report.scratch_fields_bytes =
        7 * n * sizeof(real) + // q, rhs1, rhs2, f1, f2, u_trial1, u_trial2
        (compact_mac_u_size(grid) + compact_mac_v_size(grid) + compact_mac_w_size(grid)) *
            sizeof(real);
    report.residual_workspace_bytes = StreamfunctionResidualWorkspace::estimate_device_bytes(n);
    report.affine_rhs_workspace_bytes = AffinePeriodicRhsWorkspace::estimate_device_bytes(n);
    report.pcg_workspace_bytes = projected_pcg_workspace_estimate_bytes(n);
    report.mg_hierarchy_bytes = mg_hierarchy_estimate_bytes(mg_levels);
    report.mg_preconditioner_bytes = mg_preconditioner_estimate_bytes(mg_levels);
    report.diagnostics_workspace_bytes =
        StreamfunctionDiagnosticsWorkspace::estimate_device_bytes(grid);

    // SF-20: additive, exactly zero when disabled (see StreamfunctionTypes.hpp).
    if (config.anderson.enabled) {
        const AndersonMemoryEstimate anderson_estimate =
            estimate_anderson_device_bytes(n, config.anderson.depth);
        report.anderson_history_bytes = anderson_estimate.history_bytes;
        report.anderson_staging_bytes = anderson_estimate.staging_bytes;
        report.anderson_scratch_bytes = anderson_estimate.scratch_bytes;
    }

    report.solve_path_bytes = report.scratch_fields_bytes + report.residual_workspace_bytes +
                              report.affine_rhs_workspace_bytes + report.pcg_workspace_bytes +
                              report.mg_hierarchy_bytes + report.mg_preconditioner_bytes +
                              report.anderson_history_bytes + report.anderson_staging_bytes +
                              report.anderson_scratch_bytes;
    report.diagnostics_path_bytes = report.diagnostics_workspace_bytes;
    report.total_bytes = report.fields_bytes + report.solve_path_bytes + report.diagnostics_path_bytes;
    report.fine_grid_equivalent_fields =
        n == 0 ? 0.0
               : static_cast<double>(report.total_bytes) /
                     (static_cast<double>(n) * static_cast<double>(sizeof(real)));

    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
