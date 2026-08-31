"""
Unit tests for HQM quality evaluator.
Validates Test-Retest Reliability, Parallel-Forms Reliability, Content Validity Screening Ratio,
and mathematically robust handling of zero-variance and boundary conditions.
"""

import pytest
from visionguard.eval.hqm_evaluator import HQMEvaluator


def test_hqm_reliability_and_validity():
    hqm = HQMEvaluator()

    scores_orig = [0.85, 0.40, 0.90, 0.20, 0.75]
    scores_retest = [0.84, 0.42, 0.89, 0.21, 0.76]
    scores_parallel = [0.80, 0.45, 0.85, 0.30, 0.70]

    r_test_retest = hqm.calculate_reliability(scores_orig, scores_retest)
    r_parallel = hqm.calculate_reliability(scores_orig, scores_parallel)

    assert r_test_retest is not None and r_test_retest > 0.95
    assert r_parallel is not None and r_parallel > 0.85

    validity = hqm.calculate_content_validity(95, 100)
    assert validity == 0.95


def test_hqm_zero_variance_returns_none():
    hqm = HQMEvaluator()

    # Constant arrays have zero variance (Pearson correlation is mathematically undefined)
    scores_const = [0.50, 0.50, 0.50, 0.50]
    scores_varying = [0.20, 0.40, 0.60, 0.80]

    assert hqm.calculate_reliability(scores_const, scores_const) is None
    assert hqm.calculate_reliability(scores_const, scores_varying) is None
    assert hqm.calculate_reliability([0.5], [0.5]) is None
    assert hqm.calculate_reliability([], []) is None


def test_hqm_evaluate_benchmark_quality_full_report():
    hqm = HQMEvaluator()

    orig = [0.90, 0.20, 0.80, 0.10, 0.70]
    retest = [0.88, 0.22, 0.81, 0.12, 0.68]
    parallel = [0.85, 0.25, 0.78, 0.15, 0.65]
    ground_truth = [1.0, 0.0, 1.0, 0.0, 1.0]

    report = hqm.evaluate_benchmark_quality(
        benchmark_name="Test Benchmark",
        original_scores=orig,
        retest_scores=retest,
        parallel_scores=parallel,
        human_ground_truth=ground_truth,
        valid_samples=48,
        total_samples=50
    )

    assert report.test_retest_reliability is not None and report.test_retest_reliability > 0.95
    assert report.parallel_forms_reliability is not None and report.parallel_forms_reliability > 0.90
    assert report.content_validity == 0.96
    assert report.criterion_validity is not None and report.criterion_validity > 0.80
    assert report.overall_quality_score is not None and report.overall_quality_score > 0.85
