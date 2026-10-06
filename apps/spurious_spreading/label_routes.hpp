#pragma once

/**
 * @file label_routes.hpp
 * @brief SF-32 N2a: label pairs of the `spurious_spreading` instrument --
 *        building (two routes), saving, loading and splining.
 *
 * Convention (understanding.md 3.1): psi1 = x2 + u1(x), psi2 = x3 + u2(x),
 * gbar1 = (0, 1, 0), gbar2 = (0, 0, 1) (vbar = 1), u_i cell-centred on an
 * N^3 grid of the unit cube (h = 1/N, cell centres (i + 1/2) h), splined by
 * the SF-28 periodic tricubic B-spline (knots at the cell centres). The
 * surrogate flow of SF-32 is DEFINED as c = grad psi1 x grad psi2 of the
 * splined pair; whether the pair solves eq. (14) only matters for the route
 * record.
 *
 * A labels PREFIX denotes three files:
 *
 *   <prefix>.json     metadata (route, field/pair, eps|amplitude, n, h, vbar,
 *                     gbar1, gbar2, r_F, e_v, e_psi1, e_psi2, min_abs_c_grid,
 *                     exit_reason, iterations, wall_seconds, files, config)
 *   <prefix>_u1.bin   raw little-endian IEEE double, N^3 values, layout
 *   <prefix>_u2.bin   i + N (j + N k)  (i fastest along x1)
 *
 * Routes (understanding.md 3.5):
 *   - "stack": the FROZEN periodic streamfunction stack driven exactly as
 *     `apps/closure_gate/ev_ladder_main.cu` does (public API only; one direct
 *     solve at lambda = eta = 1). Whatever state the solver leaves is saved
 *     as the pair and its exit reason, r_F, e_v, e_psi, |c| are recorded.
 *   - "analytic": a closed-form pair of analytic_pair_g.hpp sampled at the
 *     cell centres (solver-free).
 *
 * Writing is verified: the two .bin files are read back and compared
 * bytewise with the arrays that were written (round-trip check).
 *
 * min_abs_c_grid: min over the N^3 cell centres of |grad psi1 x grad psi2| of
 * the SPLINE pair (one GPU kernel + a host reduction; not a hot path). The
 * return map refuses to run when it is <= 0.5 (pre-registered usability
 * guard of understanding.md 3.5; a guard, not a regularization).
 *
 * Grid restriction: N must be a power of two in [4, 1024], so h = 1/N and the
 * period N h = 1 are exact binary numbers (the return-map target x1 = 1 is
 * then exactly one period).
 */

#include "apps/spurious_spreading/json_writer.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/CudaContext.cuh"

#include <stdexcept>
#include <string>
#include <vector>

namespace spurious_spreading {

using macroflow3d::real;

/// True iff n is a power of two in [4, 1024].
bool valid_label_grid_n(long long n);

/// File names of a labels prefix.
struct LabelFiles {
    std::string json, u1, u2;
};
LabelFiles label_files(const std::string& prefix);

/// Label pair as read from disk (host).
struct LoadedLabels {
    int n = 0;
    double h = 0.0;
    real gbar1[3] = {0.0, 0.0, 0.0};
    real gbar2[3] = {0.0, 0.0, 0.0};
    JVal meta;                  ///< the full metadata object, as read
    std::vector<double> u1, u2; ///< n^3 each, layout i + n (j + n k)
};

/**
 * Read <prefix>.json (keys used: n, h, gbar1, gbar2; h must equal 1/n
 * exactly) and the two .bin files (sizes must be exactly n^3 doubles).
 * Throws std::runtime_error with a distinct message on any inconsistency.
 */
LoadedLabels read_labels(const std::string& prefix);

/**
 * Create the prefix's parent directory if missing, write the two .bin files,
 * read them back and compare bytewise (throws on a mismatch), then write
 * <prefix>.json (meta.dump(2) + '\n'). The caller fills `meta`; this function
 * adds nothing to it.
 */
void write_labels(const std::string& prefix, int n, const std::vector<double>& u1,
                  const std::vector<double>& u2, const JVal& meta);

/// The "files" metadata member of a prefix (names relative to the prefix directory).
JVal label_files_json(const std::string& prefix);

/// `mkdir -p` (POSIX mkdir per component; existing directories are fine).
/// Throws std::runtime_error if a component cannot be created or `path` is not a directory.
void mkdir_p(const std::string& path);

/**
 * Device-resident SF-28 splines of one label pair plus the SplineLabelPair
 * built on them (make_spline_label_pair). Not copyable (the pair points into
 * the workspaces).
 */
class SplineLabels {
  public:
    SplineLabels() = default;
    SplineLabels(const SplineLabels&) = delete;
    SplineLabels& operator=(const SplineLabels&) = delete;

    /// Upload u1, u2 (n^3 each), prefilter twice, build the pair. Synchronizes.
    void build(const macroflow3d::CudaContext& ctx, int n, const std::vector<double>& u1,
               const std::vector<double>& u2, const real gbar1[3], const real gbar2[3]);

    const macroflow3d::physics::particles::streamline_tracker::SplineLabelPair& pair() const {
        return pair_;
    }
    int n() const { return n_; }
    double h() const { return h_; }

    /// min over the n^3 cell centres of |grad psi1 x grad psi2| of the spline pair.
    double min_abs_c_grid(const macroflow3d::CudaContext& ctx) const;

  private:
    int n_ = 0;
    double h_ = 0.0;
    macroflow3d::interpolation::PeriodicTricubicBSplineWorkspace ws1_, ws2_;
    macroflow3d::physics::particles::streamline_tracker::SplineLabelPair pair_{};
};

/// `spurious_spreading solve-labels ...` (argv[0] is the subcommand name).
/// Exit codes: 0 ok (any solver status), 2 usage, 4 Darcy PCG did not converge.
int run_solve_labels(int argc, char** argv);

/// `spurious_spreading analytic-labels ...`. Exit codes: 0 ok, 2 usage.
int run_analytic_labels(int argc, char** argv);

/// Thrown for command-line errors (mapped to exit code 2).
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Small shared CLI helpers (throw UsageError).
double parse_double_arg(const std::string& name, const std::string& v);
long long parse_int_arg(const std::string& name, const std::string& v);
unsigned long long parse_u64_arg(const std::string& name, const std::string& v);
bool parse_bool01_arg(const std::string& name, const std::string& v);

} // namespace spurious_spreading
