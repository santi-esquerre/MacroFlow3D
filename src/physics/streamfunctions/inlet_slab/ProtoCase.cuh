#pragma once

/**
 * @file ProtoCase.cuh
 * @brief SF-33 N4: loader of the SF-29 prototype inputs exported by
 *        `docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/export_proto.py`
 *        (step 9a, claim (a): the GPU solves the same discrete problem as the prototype).
 *
 * Case directory `<field>_<eps:%g>_<N>/` (all float64 C-order `.npy`, see the exporter README):
 *   case.json                          manifest (field, eps, N, nphi, Q0, stage_amplitudes, ...)
 *   lnk, grad_lnk_1..3, vD_1..3        (N+1, N, N)  = the LOCKED full-array layout of
 *                                                      InletSlabGrid.cuh (m3 fastest)
 *   psi0_1, psi0_2                     (N, N)       FULL inlet labels (affine part included)
 *   vperp_2, vperp_3                   (N, N)       (v2, v3) at the inlet vertices
 *   psi_or_1, psi_or_2                 (N+1, N, N)  FULL oracle labels
 *   v_rms                              ()           metrics.rms_vec(vD) (candidate_i.Ctx.v_rms)
 *   stage/<amp:%g>/                    lnk, grad_lnk_*, psi0_*, vperp_*, v_rms for every amplitude
 *                                      the prototype continuation can visit (incl. eps itself)
 *   ref_metrics.json (optional)        metrics.fd_metrics(psi_or, order=4) at full precision
 *
 * Stage inputs are built exactly as candidate_i.Ctx does: q = 1/exp(lnk) (fill_q_from_lnk),
 * u0_1 = psi0_1 - m2/N, u0_2 = psi0_2 - m3/N (host, double, the prototype's `psi0 - x` with
 * x = np.arange(N) / float(N), i.e. InletSlabGrid::coord), vperp_in, v_rms read as exported.
 *
 * Synchronization / allocation: these are LOAD-time functions (file I/O, device allocation,
 * synchronous host-to-device copies, each followed by cudaDeviceSynchronize so it has landed before
 * any work on the non-blocking ctx.cuda_stream() (SF-33 C2), one explicit ctx.synchronize() per
 * loaded input set). They
 * are not hot-path code and must not be called inside a Newton / Krylov iteration except through
 * ProtoStageProvider, which loads each amplitude once and caches it in memory.
 */

#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "InletSlabGrid.cuh"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// Status string recorded by drivers when a continuation amplitude has no exported stage input.
constexpr const char* kStatusMissingStageInput = "missing_stage_input";

/// Thrown when stage/<amp:%g>/ does not exist for a requested continuation amplitude.
class MissingStageInput : public std::runtime_error {
  public:
    MissingStageInput(real amp, const std::string& msg) : std::runtime_error(msg), amp_(amp) {}
    real amplitude() const noexcept { return amp_; }
    static const char* status() noexcept { return kStatusMissingStageInput; }

  private:
    real amp_;
};

/// Manifest fields of case.json used by the C++ side (the raw text is kept for everything else).
struct ProtoCaseMeta {
    std::string dir;
    std::string field;
    real eps = 0.0;
    int N = 0;
    int nphi = 0;
    real Q0 = 0.0;
    real inlet_vmin = 0.0;
    real max_d1phi = 0.0;
    real roundtrip_max = 0.0;
    real v_rms = 0.0;
    bool cache_hit = false;
    std::vector<std::string> stage_amplitudes; ///< "%g" names, ascending
    std::string json_text;                     ///< raw case.json
};

/// A loaded prototype case: target-amplitude stage inputs, reference data (vD, psi_or) and meta.
struct ProtoCase {
    ProtoCaseMeta meta;
    SlabStageInputs inputs;
    SlabReferenceData ref;
};

/// "%g" of an amplitude (the stage directory name; identical to Python's '%g' % amp).
std::string stage_amplitude_name(real amp);

/// Parse <dir>/case.json (throws std::runtime_error on a missing file / key).
ProtoCaseMeta read_proto_case_meta(const std::string& dir);

/// Host u0: psi0_1 - coord(m2) (which = 0) or psi0_2 - coord(m3) (which = 1); N^2 plane layout.
std::vector<real> u0_from_psi0(const InletSlabGrid& grid, const std::vector<real>& psi0, int which);

/**
 * Load one stage-input directory (lnk, grad_lnk_*, psi0_*, vperp_*, v_rms) into `out`
 * (allocate(grid), uploads, fill_q_from_lnk, u0 on the host, then ctx.synchronize()).
 * Throws NpyError / std::runtime_error on a malformed or wrongly shaped file.
 */
void load_stage_inputs_dir(CudaContext& ctx, const std::string& stage_dir,
                           const InletSlabGrid& grid, const std::string& field, real amp,
                           SlabStageInputs& out);

/**
 * Load a case directory: the target-amplitude inputs from the main files, vD and psi_or into the
 * reference data. Requires grid.n == case.json N (throws otherwise).
 */
ProtoCase load_proto_case(CudaContext& ctx, const std::string& case_dir, const InletSlabGrid& grid);

/**
 * Stage-input callback `(amp) -> const SlabStageInputs&` backed by <case_dir>/stage/<amp:%g>/.
 * Each amplitude is loaded on first use and cached in memory (references stay valid for the
 * provider's lifetime). An amplitude without a directory (or whose "%g" name does not round-trip
 * exactly) throws MissingStageInput (status "missing_stage_input").
 */
class ProtoStageProvider {
  public:
    ProtoStageProvider(CudaContext& ctx, std::string case_dir, const InletSlabGrid& grid,
                       std::string field);
    const SlabStageInputs& operator()(real amp);
    bool available(real amp) const;
    std::size_t cached_count() const { return cache_.size(); }
    std::function<const SlabStageInputs&(real)> callback() {
        return [this](real a) -> const SlabStageInputs& { return (*this)(a); };
    }

  private:
    std::string stage_dir(real amp) const;
    CudaContext* ctx_;
    std::string case_dir_;
    InletSlabGrid grid_;
    std::string field_;
    std::map<std::string, std::unique_ptr<SlabStageInputs>> cache_;
};

/// A saved prototype solution (export_proto.py --solutions): periodic parts on planes 0..N.
struct ProtoSolution {
    std::string dir;
    int N = 0;
    std::vector<real> u1, u2; ///< host, (N+1) N^2 each, full-array layout
    std::string field, cand, status, path;
    real eps = 0.0;
    int its = 0;
    real r_F = 0.0, r_out = 0.0;
    std::string json_text; ///< raw solution.json (hist, metrics_json, ...)
};

ProtoSolution load_solution(const std::string& solution_dir);

/// max |a - b| / max |b| over all entries (sizes must match); returns max |a - b| if max |b| == 0
/// and NaN if any difference or entry of b is NaN (infinite differences give inf).
real max_relative_difference(const std::vector<real>& a, const std::vector<real>& b);

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
