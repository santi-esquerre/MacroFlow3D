#include "streamfunction_operator_test_cases.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/copy.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace macroflow3d::streamfunctions::test {
namespace {

static_assert(std::is_same_v<real, double>, "SF-03 requires double precision real");

constexpr std::size_t kNx = 17;
constexpr std::size_t kNy = 19;
constexpr std::size_t kNz = 23;
constexpr std::size_t kN = kNx * kNy * kNz;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kMeanFactor = 100.0;
constexpr double kComparisonFactor = 200.0;

[[nodiscard]] std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) {
    return ix + kNx * (iy + kNy * iz);
}

[[nodiscard]] std::string grid_description() { return "17x19x23 cell-centered periodic (N=7429)"; }

[[nodiscard]] std::vector<real> shifted_periodic_field(real shift = real(1.375)) {
    std::vector<real> values(kN);
    for (std::size_t iz = 0; iz < kNz; ++iz) {
        for (std::size_t iy = 0; iy < kNy; ++iy) {
            for (std::size_t ix = 0; ix < kNx; ++ix) {
                const double x = (static_cast<double>(ix) + 0.5) / static_cast<double>(kNx);
                const double y = (static_cast<double>(iy) + 0.5) / static_cast<double>(kNy);
                const double z = (static_cast<double>(iz) + 0.5) / static_cast<double>(kNz);
                values[index(ix, iy, iz)] = shift + 0.35 * std::sin(2.0 * kPi * x) -
                    0.20 * std::cos(4.0 * kPi * y) + 0.15 * std::sin(6.0 * kPi * z) +
                    0.10 * std::cos(2.0 * kPi * (x + y - z));
            }
        }
    }
    return values;
}

[[nodiscard]] long double long_double_mean(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real value : values) sum += static_cast<long double>(value);
    return sum / static_cast<long double>(values.size());
}

[[nodiscard]] double rms(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real value : values) sum += static_cast<long double>(value) * value;
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(values.size())));
}

[[nodiscard]] double rms_difference(const std::vector<real>& left,
                                    const std::vector<real>& right) {
    if (left.size() != right.size()) throw std::invalid_argument("RMS size mismatch");
    long double sum = 0.0L;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const long double delta = static_cast<long double>(left[i]) - right[i];
        sum += delta * delta;
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(left.size())));
}

[[nodiscard]] std::vector<real> cpu_projected(const std::vector<real>& values) {
    const long double mean = long_double_mean(values);
    std::vector<real> projected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        projected[i] = static_cast<real>(static_cast<long double>(values[i]) - mean);
    }
    return projected;
}

[[nodiscard]] double scale(const std::vector<real>& values) {
    return std::max(rms(values), 1.0);
}

[[nodiscard]] double mean_limit(const std::vector<real>& values) {
    return kMeanFactor * std::numeric_limits<real>::epsilon() * scale(values);
}

[[nodiscard]] double comparison_limit(const std::vector<real>& values) {
    return kComparisonFactor * std::numeric_limits<real>::epsilon() * scale(values);
}

class ProjectorFixture {
  public:
    ProjectorFixture() : context_(0), d_values_(kN), d_backup_(kN) { workspace_.prepare(kN); }

    void enqueue_upload(const std::vector<real>& values) {
        if (values.size() != kN) throw std::invalid_argument("fixture size mismatch");
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(d_values_.data(), values.data(), kN * sizeof(real),
                                               cudaMemcpyHostToDevice, context_.cuda_stream()));
    }

    void enqueue_project() { projector_.project(context_, d_values_.span(), workspace_); }

    void enqueue_backup() {
        blas::copy(context_, DeviceSpan<const real>(d_values_.span()), d_backup_.span());
    }

    void enqueue_values_copy(std::vector<real>& values) const {
        values.resize(kN);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(values.data(), d_values_.data(), kN * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
    }

    void enqueue_backup_copy(std::vector<real>& values) const {
        values.resize(kN);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(values.data(), d_backup_.data(), kN * sizeof(real),
                                               cudaMemcpyDeviceToHost, context_.cuda_stream()));
    }

    void enqueue_mean_copy(real& value) {
        const auto mean = projector_.mean_device(context_, DeviceSpan<const real>(d_values_.span()), workspace_);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(&value, mean.data(), sizeof(real), cudaMemcpyDeviceToHost,
                                               context_.cuda_stream()));
    }

    void synchronize() const { context_.synchronize(); }

    [[nodiscard]] CudaContext& context() { return context_; }
    [[nodiscard]] DeviceBuffer<real>& values_buffer() { return d_values_; }
    [[nodiscard]] DeviceBuffer<real>& backup_buffer() { return d_backup_; }
    [[nodiscard]] constraints::MeanZeroWorkspace& workspace() { return workspace_; }

  private:
    CudaContext context_;
    DeviceBuffer<real> d_values_;
    DeviceBuffer<real> d_backup_;
    constraints::MeanZeroWorkspace workspace_;
    constraints::MeanZeroProjector projector_;
};

[[nodiscard]] CaseResult case_mean_zero_constant() {
    const std::vector<real> input(kN, real(3.25));
    ProjectorFixture fixture;
    std::vector<real> output;
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_values_copy(output);
    fixture.synchronize();
    const double residual_mean = std::abs(static_cast<double>(long_double_mean(output)));
    const double residual_rms = rms(output);
    const double limit = mean_limit(input);
    return {residual_mean <= limit && residual_rms <= limit, "mean_zero_constant", "gpu-projector",
            grid_description(), residual_mean, residual_rms, "n/a", "n/a",
            "|mean(Px)|,RMS(Px)<=100*eps*max(RMS(x),1)"};
}

[[nodiscard]] CaseResult case_mean_zero_shifted_trig_odd() {
    const auto input = shifted_periodic_field();
    const auto expected = cpu_projected(input);
    ProjectorFixture fixture;
    std::vector<real> output;
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_values_copy(output);
    fixture.synchronize();
    const double residual_mean = std::abs(static_cast<double>(long_double_mean(output)));
    const double comparison_error = rms_difference(output, expected);
    return {residual_mean <= mean_limit(input) && comparison_error <= comparison_limit(input),
            "mean_zero_shifted_trig_odd", "gpu-vs-cpu", grid_description(), residual_mean,
            comparison_error, "n/a", "n/a",
            "mean<=100*eps*scale; CPU/GPU RMS<=200*eps*scale"};
}

[[nodiscard]] CaseResult case_mean_zero_idempotence_odd() {
    const auto input = shifted_periodic_field();
    ProjectorFixture fixture;
    std::vector<real> first, second;
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_backup();
    fixture.enqueue_project();
    fixture.enqueue_backup_copy(first);
    fixture.enqueue_values_copy(second);
    fixture.synchronize();
    const double error = rms_difference(second, first);
    const double limit = comparison_limit(first);
    return {error <= limit, "mean_zero_idempotence_odd", "gpu-projector", grid_description(),
            error, limit, "n/a", "n/a", "RMS(P(Px)-Px)<=200*eps*max(RMS(Px),1)"};
}

[[nodiscard]] CaseResult case_mean_zero_diagnostic() {
    const auto input = shifted_periodic_field();
    ProjectorFixture fixture;
    real before = 0.0;
    real after = 0.0;
    fixture.enqueue_upload(input);
    fixture.enqueue_mean_copy(before);
    fixture.enqueue_project();
    fixture.enqueue_mean_copy(after);
    fixture.synchronize();
    const double expected_before = static_cast<double>(long_double_mean(input));
    const double before_error = std::abs(static_cast<double>(before) - expected_before);
    const double after_mean = std::abs(static_cast<double>(after));
    return {before_error <= comparison_limit(input) && after_mean <= mean_limit(input),
            "mean_zero_diagnostic", "gpu-diagnostic", grid_description(), before_error, after_mean,
            "n/a", "n/a", "diagnostic double mean and projected mean<=100*eps*scale"};
}

[[nodiscard]] CaseResult case_mean_zero_double_precision() {
    const auto input = shifted_periodic_field(real(1.0e8));
    const auto expected = cpu_projected(input);
    ProjectorFixture fixture;
    std::vector<real> output;
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_values_copy(output);
    fixture.synchronize();

    float float_sum = 0.0F;
    for (const real value : input) float_sum += static_cast<float>(value);
    const float float_mean = float_sum / static_cast<float>(input.size());
    std::vector<real> float_mutant(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) float_mutant[i] = input[i] - float_mean;

    const double gpu_error = rms_difference(output, expected);
    const double float_mutant_error = rms_difference(float_mutant, expected);
    const bool mutant_rejected = float_mutant_error > 1.0e-3;
    return {gpu_error <= comparison_limit(input) && mutant_rejected,
            "mean_zero_double_precision", "gpu-precision", grid_description(), gpu_error,
            float_mutant_error, "n/a", "n/a",
            "double CPU/GPU<=200*eps*scale; float-accumulator mutant RMS>1e-3"};
}

[[nodiscard]] CaseResult case_mean_zero_workspace_contract() {
    const auto input = shifted_periodic_field();
    ProjectorFixture fixture;
    const real* const values_pointer = fixture.values_buffer().data();
    const real* const backup_pointer = fixture.backup_buffer().data();
    const std::size_t values_capacity = fixture.values_buffer().capacity();
    const std::size_t backup_capacity = fixture.backup_buffer().capacity();
    const std::size_t workspace_capacity = fixture.workspace().temporary_storage_capacity_bytes();
    std::vector<real> output;
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_backup();
    fixture.enqueue_project();
    fixture.enqueue_values_copy(output);
    fixture.synchronize();

    constraints::MeanZeroWorkspace wrong_workspace;
    wrong_workspace.prepare(kN - 1);
    bool rejected = false;
    try {
        constraints::MeanZeroProjector projector;
        projector.project(fixture.context(), fixture.values_buffer().span(), wrong_workspace);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    const bool stable = fixture.values_buffer().data() == values_pointer &&
                        fixture.backup_buffer().data() == backup_pointer &&
                        fixture.values_buffer().capacity() == values_capacity &&
                        fixture.backup_buffer().capacity() == backup_capacity &&
                        fixture.workspace().temporary_storage_capacity_bytes() == workspace_capacity;
    const double residual_mean = std::abs(static_cast<double>(long_double_mean(output)));
    return {stable && rejected && residual_mean <= mean_limit(input),
            "mean_zero_workspace_contract", "gpu-workspace", grid_description(), residual_mean,
            static_cast<double>(workspace_capacity), "n/a", "n/a",
            "prepared buffers stable; exact-size mismatch rejects before enqueue"};
}

[[nodiscard]] CaseResult case_mean_zero_stream_ordering() {
    const auto input = shifted_periodic_field();
    const auto expected = cpu_projected(input);
    ProjectorFixture fixture;
    std::vector<real> output;
    // H2D -> project -> D2H are all queued on CudaContext::cuda_stream(); the
    // sole explicit host synchronization in this case is below.
    fixture.enqueue_upload(input);
    fixture.enqueue_project();
    fixture.enqueue_values_copy(output);
    fixture.synchronize();
    const double error = rms_difference(output, expected);
    const double residual_mean = std::abs(static_cast<double>(long_double_mean(output)));
    return {error <= comparison_limit(input) && residual_mean <= mean_limit(input),
            "mean_zero_stream_ordering", "gpu-stream", grid_description(), error, residual_mean,
            "n/a", "n/a", "single-stream H2D->project->D2H; CPU/GPU and mean bounds"};
}

} // namespace

CaseRegistry mean_zero_projector_case_registry() {
    return {
        {"mean_zero_constant", case_mean_zero_constant},
        {"mean_zero_shifted_trig_odd", case_mean_zero_shifted_trig_odd},
        {"mean_zero_idempotence_odd", case_mean_zero_idempotence_odd},
        {"mean_zero_diagnostic", case_mean_zero_diagnostic},
        {"mean_zero_double_precision", case_mean_zero_double_precision},
        {"mean_zero_workspace_contract", case_mean_zero_workspace_contract},
        {"mean_zero_stream_ordering", case_mean_zero_stream_ordering},
    };
}

} // namespace macroflow3d::streamfunctions::test
