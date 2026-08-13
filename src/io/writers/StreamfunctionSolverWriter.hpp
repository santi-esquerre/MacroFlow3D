#pragma once

/**
 * @file StreamfunctionSolverWriter.hpp
 * @brief Per-realization export writers for the Lester equation (14)
 *        streamfunction solver stage (SF-16 T02).
 *
 * All three exports are independently gated by
 * `io::StreamfunctionExportConfig` (`iteration_history`, `summary`,
 * `fields`) and are written once per realization, directly under
 * `OutputLayout::streamfunction_dir(r, nx)`. Each file is written fresh
 * (not appended): one solve report -> one directory -> one set of files,
 * unlike the append-mode PSPTA diagnostics writers.
 *
 * `iteration_history.csv` and `trial_history.csv` are two separate files
 * (rather than one file with a blank-line-separated second header) because
 * they have structurally different row shapes (one row per accepted Picard
 * iteration vs. one row per backtracking trial, of which there can be many
 * per iteration); keeping them separate avoids an ambiguous single-file
 * schema and lets each be parsed with a plain CSV reader.
 *
 * `write_raw_fields()` performs exactly the one `cudaMemcpy` per component
 * (u1, u2) needed to dump the accepted periodic fluctuations; this is the
 * ONLY device-to-host transfer point in the streamfunction export path and
 * only runs when `exports.fields` is enabled.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../external/nlohmann/json.hpp"
#include "../../physics/streamfunctions/ContinuationController.hpp"
#include "../../physics/streamfunctions/StreamfunctionSolver.cuh"
#include "../../physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "../config/Config.hpp"

#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace io {

struct StreamfunctionSolverWriter {

    // ── Enum -> string (report echo only; never fed back into control flow) ──

    static const char* status_str(streamfunctions::StreamfunctionSolveStatus s) {
        using S = streamfunctions::StreamfunctionSolveStatus;
        switch (s) {
        case S::not_run:
            return "not_run";
        case S::converged:
            return "converged";
        case S::not_converged:
            return "not_converged";
        case S::invalid_problem:
            return "invalid_problem";
        }
        return "unknown";
    }

    static const char* exit_reason_str(streamfunctions::PicardExitReason r) {
        using R = streamfunctions::PicardExitReason;
        switch (r) {
        case R::none:
            return "none";
        case R::converged:
            return "converged";
        case R::budget_exhausted:
            return "budget_exhausted";
        case R::linear_block_failure:
            return "linear_block_failure";
        case R::stagnated:
            return "stagnated";
        case R::omega_floor_rejected:
            return "omega_floor_rejected";
        case R::newton_exhausted:
            return "newton_exhausted";
        case R::newton_budget_exhausted:
            return "newton_budget_exhausted";
        }
        return "unknown";
    }

    static const char* continuation_status_str(streamfunctions::ContinuationStatus s) {
        using S = streamfunctions::ContinuationStatus;
        switch (s) {
        case S::reached_target:
            return "reached_target";
        case S::baseline_failed:
            return "baseline_failed";
        case S::step_floor_exhausted:
            return "step_floor_exhausted";
        case S::invalid_problem:
            return "invalid_problem";
        }
        return "unknown";
    }

    static const char* continuation_axis_str(streamfunctions::ContinuationAxis a) {
        using A = streamfunctions::ContinuationAxis;
        switch (a) {
        case A::eta:
            return "eta";
        case A::epsilon:
            return "epsilon";
        }
        return "unknown";
    }

    static const char* continuation_failure_str(streamfunctions::ContinuationStageFailure f) {
        using F = streamfunctions::ContinuationStageFailure;
        switch (f) {
        case F::none:
            return "none";
        case F::solver_not_converged:
            return "solver_not_converged";
        case F::solver_invalid_problem:
            return "solver_invalid_problem";
        }
        return "unknown";
    }

    // ── SF-21 T03: heterogeneity (lambda) continuation echo helpers ───────

    static const char* heterogeneity_status_str(streamfunctions::HeterogeneityStatus s) {
        using S = streamfunctions::HeterogeneityStatus;
        switch (s) {
        case S::reached_target:
            return "reached_target";
        case S::baseline_failed:
            return "baseline_failed";
        case S::lambda_floor_exhausted:
            return "lambda_floor_exhausted";
        case S::epsilon_floor_exhausted:
            return "epsilon_floor_exhausted";
        case S::invalid_problem:
            return "invalid_problem";
        }
        return "unknown";
    }

    static const char* heterogeneity_axis_str(streamfunctions::HeterogeneityAxis a) {
        using A = streamfunctions::HeterogeneityAxis;
        switch (a) {
        case A::lambda:
            return "lambda";
        case A::eta_rescue:
            return "eta_rescue";
        case A::epsilon:
            return "epsilon";
        }
        return "unknown";
    }

    static const char* trial_outcome_str(streamfunctions::PicardTrialOutcome o) {
        using O = streamfunctions::PicardTrialOutcome;
        switch (o) {
        case O::accepted:
            return "accepted";
        case O::rejected_nonfinite:
            return "rejected_nonfinite";
        case O::rejected_degeneracy:
            return "rejected_degeneracy";
        case O::rejected_percentile:
            return "rejected_percentile";
        case O::rejected_armijo:
            return "rejected_armijo";
        }
        return "unknown";
    }

    // ── iteration_history.csv: one row per Picard iteration record ────────

    static void
    write_iteration_history(const std::string& path,
                            const std::vector<streamfunctions::PicardIterationRecord>& history) {
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);

        f << "k,r_F,r1,r2,psi1_iterations,psi1_rel_residual,psi1_converged,"
             "psi2_iterations,psi2_rel_residual,psi2_converged\n";

        for (std::size_t k = 0; k < history.size(); ++k) {
            const auto& it = history[k];
            f << k << ',' << it.r_F << ',' << it.r1 << ',' << it.r2 << ','
              << it.psi1_result.iterations << ',' << it.psi1_result.relative_projected_residual
              << ',' << (it.psi1_result.converged ? 1 : 0) << ',' << it.psi2_result.iterations
              << ',' << it.psi2_result.relative_projected_residual << ','
              << (it.psi2_result.converged ? 1 : 0) << '\n';
        }
    }

    // ── trial_history.csv: one row per SF-15 backtracking trial ───────────

    static void write_trial_history(const std::string& path,
                                    const std::vector<streamfunctions::PicardTrialRecord>& trials) {
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);

        f << "iteration,omega,r_F_trial,outcome\n";
        for (const auto& t : trials) {
            f << t.iteration << ',' << t.omega << ',' << t.r_F_trial << ','
              << trial_outcome_str(t.outcome) << '\n';
        }
    }

    // ── stage_history.csv: one row per SF-17 ContinuationStageRecord ──────

    static void write_stage_history(
        const std::string& path,
        const std::vector<streamfunctions::ContinuationStageRecord>& history) {
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);

        f << "axis,param_start,param_attempted,step_attempted,accepted,failure,exit_reason,"
             "picard_iterations,final_omega,r_F,r1,r2,unexplained_fraction,c_percentile_p001,"
             "psi1_iterations,psi2_iterations\n";

        for (const auto& s : history) {
            f << continuation_axis_str(s.axis) << ',' << s.param_start << ',' << s.param_attempted
              << ',' << s.step_attempted << ',' << (s.accepted ? 1 : 0) << ','
              << continuation_failure_str(s.failure) << ',' << exit_reason_str(s.exit_reason)
              << ',' << s.picard_iterations << ',' << s.final_omega << ',' << s.r_F << ','
              << s.r1 << ',' << s.r2 << ',' << s.unexplained_fraction << ','
              << s.c_percentile_p001 << ',' << s.psi1_iterations << ',' << s.psi2_iterations
              << '\n';
        }
    }

    // ── stage_history.csv (heterogeneity runs): one row per SF-21
    // HeterogeneityStageRecord. Same file path as the SF-17 overload above
    // (io::OutputLayout::streamfunction_stage_history_csv); only one of the
    // two overloads is ever called for a given realization/grid.

    static void write_stage_history(
        const std::string& path,
        const std::vector<streamfunctions::HeterogeneityStageRecord>& history) {
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);

        f << "axis,lambda,eta,epsilon,base_axis,param_start,param_attempted,step_attempted,"
             "accepted,failure,exit_reason,picard_iterations,final_omega,r_F,r1,r2,"
             "mg_rebuild_count,e_v,invariance_e_psi1,invariance_e_psi2,e_div,c_percentile_p001,"
             "degeneracy_total0,degeneracy_unexplained0,psi1_iterations,psi2_iterations\n";

        for (const auto& s : history) {
            const auto& b = s.base;
            f << heterogeneity_axis_str(s.axis) << ',' << s.lambda_value << ',' << s.eta_value
              << ',' << s.epsilon_value << ',' << continuation_axis_str(b.axis) << ','
              << b.param_start << ',' << b.param_attempted << ',' << b.step_attempted << ','
              << (b.accepted ? 1 : 0) << ',' << continuation_failure_str(b.failure) << ','
              << exit_reason_str(b.exit_reason) << ',' << b.picard_iterations << ','
              << b.final_omega << ',' << b.r_F << ',' << b.r1 << ',' << b.r2 << ','
              << s.mg_rebuild_count << ',' << s.e_v << ',' << s.invariance_e_psi1 << ','
              << s.invariance_e_psi2 << ',' << s.e_div << ',' << s.c_percentile_p001 << ','
              << s.degeneracy_total0 << ',' << s.degeneracy_unexplained0 << ','
              << b.psi1_iterations << ',' << b.psi2_iterations << '\n';
        }
    }

    // ── summary.json: flat, stable-schema single-report summary ───────────
    //
    // The 6-argument overload keeps the SF-16 schema byte-identical (no
    // "continuation" key at all). The 7-argument overload additionally
    // appends a "continuation" object (SF-17 T03) and is only called from
    // the pipeline when `streamfunction_solver.continuation.enabled`.

    static void write_summary_json(const std::string& path, const Grid3D& grid,
                                   const StreamfunctionSolverYamlConfig& cfg_yaml, real vbar_used,
                                   const streamfunctions::StreamfunctionSolveReport& report,
                                   const streamfunctions::StreamfunctionContinuationReport&
                                       continuation_report) {
        using json = nlohmann::json;
        json j = build_summary_json(grid, cfg_yaml, vbar_used, report);

        json jc;
        jc["status"] = continuation_status_str(continuation_report.status);
        if (continuation_report.status != streamfunctions::ContinuationStatus::reached_target) {
            jc["failed_axis"] = continuation_axis_str(continuation_report.failed_axis);
        }
        jc["final_eta"] = (double)continuation_report.final_eta;
        jc["final_epsilon"] = (double)continuation_report.final_epsilon;
        jc["num_stages"] = continuation_report.stage_history.size();
        jc["snapshot_bytes"] = continuation_report.snapshot_bytes;
        j["continuation"] = jc;

        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);
        f << j.dump(2) << "\n";
    }

    // SF-21 T03: heterogeneity (lambda) continuation runs. Extends
    // build_summary_json (SF-16 schema, computed from
    // heterogeneity_report.final_solve) with a "heterogeneity" object; never
    // called together with either overload above for the same realization.
    static void write_summary_json(
        const std::string& path, const Grid3D& grid, const StreamfunctionSolverYamlConfig& cfg_yaml,
        real vbar_used, const streamfunctions::HeterogeneityContinuationReport& heterogeneity_report) {
        using json = nlohmann::json;
        json j = build_summary_json(grid, cfg_yaml, vbar_used, heterogeneity_report.final_solve);

        json jh;
        jh["status"] = heterogeneity_status_str(heterogeneity_report.status);
        if (heterogeneity_report.status != streamfunctions::HeterogeneityStatus::reached_target) {
            jh["failed_axis"] = heterogeneity_axis_str(heterogeneity_report.failed_axis);
        }
        jh["final_lambda"] = (double)heterogeneity_report.final_lambda;
        jh["final_eta"] = (double)heterogeneity_report.final_eta;
        jh["final_epsilon"] = (double)heterogeneity_report.final_epsilon;
        jh["num_stages"] = heterogeneity_report.stage_history.size();
        jh["total_mg_rebuilds"] = heterogeneity_report.total_mg_rebuilds;
        jh["snapshot_bytes"] = heterogeneity_report.snapshot_bytes;
        jh["driver_owned_bytes"] = heterogeneity_report.driver_owned_bytes;

        const auto& flow = heterogeneity_report.final_flow;
        json k_eff = json::array();
        for (int i = 0; i < 3; ++i) {
            json row = json::array();
            for (int d = 0; d < 3; ++d)
                row.push_back((double)flow.K_eff[i][d]);
            k_eff.push_back(row);
        }
        jh["final_flow"] = {
            {"K_eff", k_eff},
            {"G", {(double)flow.G[0], (double)flow.G[1], (double)flow.G[2]}},
            {"achieved_mean_flux",
             {(double)flow.achieved_mean_flux[0], (double)flow.achieved_mean_flux[1],
              (double)flow.achieved_mean_flux[2]}},
            {"div_max_abs", (double)flow.div_max_abs},
            {"div_rms", (double)flow.div_rms}};
        j["heterogeneity"] = jh;

        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);
        f << j.dump(2) << "\n";
    }

    static void write_summary_json(const std::string& path, const Grid3D& grid,
                                   const StreamfunctionSolverYamlConfig& cfg_yaml, real vbar_used,
                                   const streamfunctions::StreamfunctionSolveReport& report) {
        using json = nlohmann::json;
        json j = build_summary_json(grid, cfg_yaml, vbar_used, report);

        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);
        f << j.dump(2) << "\n";
    }

  private:
    static nlohmann::json
    build_summary_json(const Grid3D& grid, const StreamfunctionSolverYamlConfig& cfg_yaml,
                       real vbar_used, const streamfunctions::StreamfunctionSolveReport& report) {
        using json = nlohmann::json;
        json j;

        j["status"] = status_str(report.status);
        j["exit_reason"] = exit_reason_str(report.exit_reason);
        j["picard_iterations"] = report.picard_iterations;
        j["final_omega"] = (double)report.final_omega;

        const auto& res = report.residual;
        j["residual"] = {{"r_F", (double)res.r_F},
                         {"r1", (double)res.r1},
                         {"r2", (double)res.r2},
                         {"rms_f1", (double)res.rms_f1},
                         {"rms_f2", (double)res.rms_f2},
                         {"linf_f1", (double)res.linf_f1},
                         {"linf_f2", (double)res.linf_f2},
                         {"q_rms", (double)res.q_rms},
                         {"v_rms", (double)res.v_rms},
                         {"L_ref", (double)res.L_ref},
                         {"raw_rhs_mean_psi1", (double)res.raw_rhs_mean_psi1},
                         {"raw_rhs_mean_psi2", (double)res.raw_rhs_mean_psi2},
                         {"projected_rhs_mean_psi1", (double)res.projected_rhs_mean_psi1},
                         {"projected_rhs_mean_psi2", (double)res.projected_rhs_mean_psi2},
                         {"nonfinite_s1", res.nonfinite_s1},
                         {"nonfinite_s2", res.nonfinite_s2}};

        const auto& diag = report.diagnostics;
        j["diagnostics"] = {{"e_v", (double)diag.e_v},
                            {"e_div", (double)diag.e_div},
                            {"invariance_e_psi1", (double)diag.invariance_e_psi1},
                            {"invariance_e_psi2", (double)diag.invariance_e_psi2},
                            {"c_min", (double)diag.c_min},
                            {"c_max", (double)diag.c_max},
                            {"c_mean", (double)diag.c_mean},
                            {"v_d_rms", (double)diag.v_d_rms},
                            {"rms_magnitude", (double)diag.rms_magnitude},
                            {"linf_magnitude", (double)diag.linf_magnitude},
                            {"rms_theta", (double)diag.rms_theta},
                            {"max_theta", (double)diag.max_theta},
                            {"angle_included_count", diag.angle_included_count},
                            {"angle_excluded_count", diag.angle_excluded_count}};

        const auto& mem = report.memory;
        j["memory"] = {{"total_bytes", mem.total_bytes},
                       {"fine_grid_equivalent_fields", mem.fine_grid_equivalent_fields},
                       {"fields_bytes", mem.fields_bytes},
                       {"solve_path_bytes", mem.solve_path_bytes},
                       {"diagnostics_path_bytes", mem.diagnostics_path_bytes}};

        j["config"] = {{"affine_mean_velocity_mode", cfg_yaml.affine_mean_velocity.mode},
                       {"vbar_used", (double)vbar_used},
                       {"epsilon", (double)cfg_yaml.epsilon},
                       {"eta", (double)cfg_yaml.eta},
                       {"picard_max_iter", cfg_yaml.picard.max_iter},
                       {"picard_tolerance", (double)cfg_yaml.picard.tolerance},
                       {"picard_omega", (double)cfg_yaml.picard.omega},
                       {"adaptive_enabled", cfg_yaml.adaptive.enabled},
                       {"linear_rtol", (double)cfg_yaml.linear.rtol},
                       {"linear_max_iter", cfg_yaml.linear.max_iter},
                       {"linear_check_every", cfg_yaml.linear.check_every},
                       {"mg_num_levels", cfg_yaml.mg.num_levels}};

        // SF-24 T02: additive "newton" object, following the same conditional
        // pattern as the "continuation" object in the 7-argument
        // write_summary_json overload above -- only present when the
        // subsystem is enabled, so a default (disabled) config's summary.json
        // stays byte-identical to the pre-SF-24 schema. Anderson counters
        // (report.anderson_accepted/anderson_rejected/
        // anderson_condition_resets) are NOT currently exported anywhere in
        // this writer or in EnsembleRunner.cu (verified by inspection); no
        // established "reports Anderson counters" export site exists to
        // mirror, so none is invented here for Newton either.
        if (cfg_yaml.newton.enabled) {
            j["newton"] = {{"activations", report.newton_activations},
                           {"steps_accepted", report.newton_steps_accepted},
                           {"step_failures", report.newton_step_failures},
                           {"rescue_events", report.newton_rescue_events},
                           {"jv_evaluations", report.newton_jv_evaluations}};
        }

        j["grid"] = {{"nx", grid.nx},         {"ny", grid.ny},         {"nz", grid.nz},
                     {"dx", (double)grid.dx}, {"dy", (double)grid.dy}, {"dz", (double)grid.dz}};

        return j;
    }

  public:
    // ── raw u1/u2 dumps + meta.json (only device transfer in export path) ─

    static void write_raw_fields(const std::string& u1_path, const std::string& u2_path,
                                 const std::string& meta_path, const Grid3D& grid,
                                 DeviceSpan<const real> u1, DeviceSpan<const real> u2) {
        const std::size_t n = grid.num_cells();
        if (u1.size() != n || u2.size() != n)
            throw std::runtime_error(
                "[StreamfunctionSolverWriter] u1/u2 span size mismatch with grid.num_cells()");

        std::vector<real> host_u1(n), host_u2(n);
        // Only device->host transfer in the streamfunction export path.
        cudaMemcpy(host_u1.data(), u1.data(), n * sizeof(real), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_u2.data(), u2.data(), n * sizeof(real), cudaMemcpyDeviceToHost);

        auto dump = [&](const std::string& path, const std::vector<real>& host) {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            if (!f.is_open())
                throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + path);
            f.write(reinterpret_cast<const char*>(host.data()),
                    static_cast<std::streamsize>(host.size() * sizeof(real)));
        };
        dump(u1_path, host_u1);
        dump(u2_path, host_u2);

        using json = nlohmann::json;
        json j;
        j["nx"] = grid.nx;
        j["ny"] = grid.ny;
        j["nz"] = grid.nz;
        j["n"] = n;
        j["order"] = "x-fastest, idx = i + nx*(j + ny*k)";
        j["dtype"] = (sizeof(real) == 8) ? "float64" : "float32";
        j["byte_order"] = "little-endian";
        j["fields"] = {"u1 (psi1 periodic fluctuation)", "u2 (psi2 periodic fluctuation)"};

        std::ofstream f(meta_path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("[StreamfunctionSolverWriter] Cannot open: " + meta_path);
        f << j.dump(2) << "\n";
    }
};

} // namespace io
} // namespace macroflow3d
