#include "InvariantPairSearch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

namespace {

static double collapse_floor(double reference_value, double relative_floor) {
    return std::max(reference_value, 1.0e-30) * relative_floor;
}

static auto candidate_order_key(const PairSearchCandidate& candidate) {
    return std::make_tuple(candidate.n_nonzero_fail, candidate.total_fail, candidate.max_fail_count,
                           candidate.final_drift_max, candidate.rel_rms_mismatch,
                           candidate.rms_invariance_sum, candidate.degeneracy_fraction,
                           candidate.mode_i, candidate.mode_j, candidate.angle_deg);
}

} // namespace

CandidateDecision evaluate_pair_candidate(const PairSearchCandidate& candidate,
                                          const CandidateCollapseReference& reference) {
    CandidateDecision out;
    const double min_gradient_floor =
        collapse_floor(reference.reference_min_gradient_rms, reference.min_relative_gradient_rms);
    const double min_range_floor =
        collapse_floor(reference.reference_min_field_range, reference.min_relative_field_range);

    if (candidate.min_gradient_rms < min_gradient_floor ||
        candidate.min_field_range < min_range_floor) {
        out.admissible = false;
        out.rejection_reason = "collapsed_gradients_or_ranges";
        out.score = std::numeric_limits<double>::infinity();
        return out;
    }

    out.admissible = true;
    out.score = static_cast<double>(candidate.n_nonzero_fail) * 1.0e9 +
                static_cast<double>(candidate.total_fail) * 1.0e6 +
                static_cast<double>(candidate.max_fail_count) * 1.0e3 +
                candidate.final_drift_max * 1.0e8 + candidate.rel_rms_mismatch * 1.0e4 +
                candidate.rms_invariance_sum * 1.0e3 + candidate.degeneracy_fraction * 1.0e2;
    return out;
}

bool candidate_is_preferred(const PairSearchCandidate& lhs,
                            const CandidateCollapseReference& lhs_reference,
                            const PairSearchCandidate& rhs,
                            const CandidateCollapseReference& rhs_reference) {
    const CandidateDecision lhs_decision = evaluate_pair_candidate(lhs, lhs_reference);
    const CandidateDecision rhs_decision = evaluate_pair_candidate(rhs, rhs_reference);

    if (lhs_decision.admissible != rhs_decision.admissible)
        return lhs_decision.admissible;
    if (!lhs_decision.admissible)
        return candidate_order_key(lhs) < candidate_order_key(rhs);
    if (lhs_decision.score != rhs_decision.score)
        return lhs_decision.score < rhs_decision.score;
    return candidate_order_key(lhs) < candidate_order_key(rhs);
}

PairSearchCandidate choose_preferred_candidate(const std::vector<PairSearchCandidate>& candidates,
                                               const CandidateCollapseReference& reference) {
    if (candidates.empty())
        throw std::invalid_argument("choose_preferred_candidate requires at least one candidate");

    PairSearchCandidate best = candidates.front();
    for (size_t idx = 1; idx < candidates.size(); ++idx) {
        if (candidate_is_preferred(candidates[idx], reference, best, reference))
            best = candidates[idx];
    }
    return best;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
