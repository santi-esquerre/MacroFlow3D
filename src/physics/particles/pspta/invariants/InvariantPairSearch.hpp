#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

struct CandidateCollapseReference {
    double reference_min_gradient_rms = 0.0;
    double reference_min_field_range = 0.0;
    double min_relative_gradient_rms = 0.2;
    double min_relative_field_range = 0.2;
};

struct PairSearchCandidate {
    int mode_i = -1;
    int mode_j = -1;
    double angle_deg = 0.0;
    double min_gradient_rms = 0.0;
    double min_field_range = 0.0;
    double rel_rms_mismatch = 0.0;
    double rms_invariance_sum = 0.0;
    double degeneracy_fraction = 0.0;
    double final_drift_max = 0.0;
    long long total_fail = 0;
    uint32_t n_nonzero_fail = 0;
    uint32_t max_fail_count = 0;
};

struct CandidateDecision {
    bool admissible = false;
    double score = 0.0;
    std::string rejection_reason;
};

CandidateDecision evaluate_pair_candidate(const PairSearchCandidate& candidate,
                                          const CandidateCollapseReference& reference);

bool candidate_is_preferred(const PairSearchCandidate& lhs,
                            const CandidateCollapseReference& lhs_reference,
                            const PairSearchCandidate& rhs,
                            const CandidateCollapseReference& rhs_reference);

PairSearchCandidate choose_preferred_candidate(const std::vector<PairSearchCandidate>& candidates,
                                               const CandidateCollapseReference& reference);

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
