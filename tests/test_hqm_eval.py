"""
Unit tests for HQM quality evaluator.
"""

from visionguard.eval.hqm_evaluator import HQMEvaluator


def test_hqm_reliability_and_validity():
    hqm = HQMEvaluator()

    scores_orig = [0.85, 0.40, 0.90, 0.20, 0.75]
    scores_retest = [0.84, 0.42, 0.89, 0.21, 0.76]
    scores_parallel = [0.80, 0.45, 0.85, 0.30, 0.70]

    r_test_retest = hqm.calculate_reliability(scores_orig, scores_retest)
    r_parallel = hqm.calculate_reliability(scores_orig, scores_parallel)

    assert r_test_retest > 0.95
    assert r_parallel > 0.85

    validity = hqm.calculate_content_validity(95, 100)
    assert validity == 0.95
