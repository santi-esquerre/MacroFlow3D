#include "src/physics/particles/pspta/invariants/InvariantPairSearch.hpp"

#include <cassert>
#include <vector>

using namespace macroflow3d::physics::particles::pspta;

int main() {
    CandidateCollapseReference ref;
    ref.reference_min_gradient_rms = 1.0;
    ref.reference_min_field_range = 1.0;

    PairSearchCandidate baseline;
    baseline.mode_i = 0;
    baseline.mode_j = 1;
    baseline.angle_deg = 0.0;
    baseline.min_gradient_rms = 1.0;
    baseline.min_field_range = 1.0;
    baseline.rel_rms_mismatch = 1.05;
    baseline.rms_invariance_sum = 2.0e-2;
    baseline.degeneracy_fraction = 0.25;
    baseline.final_drift_max = 1.0e-6;
    baseline.total_fail = 1800;
    baseline.n_nonzero_fail = 320;

    PairSearchCandidate improved = baseline;
    improved.mode_i = 2;
    improved.mode_j = 4;
    improved.rel_rms_mismatch = 0.65;
    improved.rms_invariance_sum = 7.0e-3;
    improved.degeneracy_fraction = 0.05;
    improved.final_drift_max = 2.0e-7;
    improved.total_fail = 150;
    improved.n_nonzero_fail = 18;

    PairSearchCandidate collapsed = improved;
    collapsed.mode_i = 3;
    collapsed.mode_j = 5;
    collapsed.min_gradient_rms = 1.0e-5;
    collapsed.min_field_range = 5.0e-6;
    collapsed.total_fail = 0;
    collapsed.n_nonzero_fail = 0;
    collapsed.final_drift_max = 1.0e-8;

    const CandidateDecision base_decision = evaluate_pair_candidate(baseline, ref);
    const CandidateDecision improved_decision = evaluate_pair_candidate(improved, ref);
    const CandidateDecision collapsed_decision = evaluate_pair_candidate(collapsed, ref);

    assert(base_decision.admissible);
    assert(improved_decision.admissible);
    assert(!collapsed_decision.admissible);
    assert(collapsed_decision.rejection_reason == "collapsed_gradients_or_ranges");

    const PairSearchCandidate best =
        choose_preferred_candidate({baseline, improved, collapsed}, ref);
    assert(best.mode_i == improved.mode_i);
    assert(best.mode_j == improved.mode_j);

    PairSearchCandidate host_a = baseline;
    host_a.mode_i = -1;
    host_a.mode_j = -1;
    host_a.total_fail = 0;
    host_a.n_nonzero_fail = 0;
    host_a.max_fail_count = 0;
    host_a.final_drift_max = 0.0;
    host_a.rel_rms_mismatch = 1.10;
    host_a.rms_invariance_sum = 2.0e-2;
    host_a.degeneracy_fraction = 0.22;

    PairSearchCandidate host_b = host_a;
    host_b.rel_rms_mismatch = 0.82;
    host_b.rms_invariance_sum = 8.0e-3;
    host_b.degeneracy_fraction = 0.04;

    PairSearchCandidate host_collapsed = host_b;
    host_collapsed.min_gradient_rms = 1.0e-6;
    host_collapsed.min_field_range = 1.0e-6;

    const PairSearchCandidate best_host =
        choose_preferred_candidate({host_a, host_b, host_collapsed}, ref);
    assert(best_host.rel_rms_mismatch == host_b.rel_rms_mismatch);
    assert(best_host.rms_invariance_sum == host_b.rms_invariance_sum);

    return 0;
}
