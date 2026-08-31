"""
HQM Framework Evaluator for VisionGuard (arXiv:2406.17115v3).
Calculates Test-Retest Reliability, Parallel-Forms Reliability, Content Validity, and Criterion Validity.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.stats import pearsonr
from visionguard.schemas import HQMMetrics


class HQMEvaluator:
    """
    Implements psychometric quality evaluation for hallucination benchmarks (HQM Framework).
    Evaluates Test-Retest Reliability, Parallel-Forms Reliability, Content Validity Screening Ratio,
    and Criterion Validity. Correctly handles zero-variance and boundary conditions by returning
    None (undefined) rather than false perfect reliability.
    """

    def __init__(self):
        pass

    def calculate_reliability(self, scores_a: List[float], scores_b: List[float]) -> Optional[float]:
        """
        Calculates Pearson correlation between two evaluation runs (Test-Retest or Parallel-Forms).
        Returns None (undefined) if variance is 0 in either vector, lengths mismatch, or N < 2.
        """
        if len(scores_a) < 2 or len(scores_b) < 2 or len(scores_a) != len(scores_b):
            return None
        
        arr_a = np.array(scores_a, dtype=float)
        arr_b = np.array(scores_b, dtype=float)

        # Pearson correlation is mathematically undefined when variance is zero
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a == 0.0 or std_b == 0.0:
            return None

        r, _ = pearsonr(arr_a, arr_b)
        if np.isnan(r):
            return None
        return float(np.clip(r, -1.0, 1.0))

    def calculate_content_validity(self, valid_sample_count: int, total_sample_count: int) -> Optional[float]:
        """
        Calculates content validity screening ratio N_valid / N.
        Measures the proportion of benchmark samples satisfying unambiguous image-claim alignment.
        """
        if total_sample_count <= 0:
            return None
        ratio = valid_sample_count / total_sample_count
        return float(np.clip(ratio, 0.0, 1.0))

    def evaluate_benchmark_quality(
        self,
        benchmark_name: str,
        original_scores: List[float],
        retest_scores: List[float],
        parallel_scores: List[float],
        human_ground_truth: List[float],
        valid_samples: int,
        total_samples: int
    ) -> HQMMetrics:
        """
        Runs comprehensive HQM assessment for a benchmark dataset.
        """
        test_retest_r = self.calculate_reliability(original_scores, retest_scores)
        parallel_forms_r = self.calculate_reliability(original_scores, parallel_scores)
        content_val = self.calculate_content_validity(valid_samples, total_samples)
        criterion_val = self.calculate_reliability(original_scores, human_ground_truth)

        valid_metrics = [m for m in [test_retest_r, parallel_forms_r, content_val, criterion_val] if m is not None]
        overall_score = round(float(np.mean(valid_metrics)), 4) if valid_metrics else None

        return HQMMetrics(
            test_retest_reliability=round(test_retest_r, 4) if test_retest_r is not None else None,
            parallel_forms_reliability=round(parallel_forms_r, 4) if parallel_forms_r is not None else None,
            content_validity=round(content_val, 4) if content_val is not None else None,
            criterion_validity=round(criterion_val, 4) if criterion_val is not None else None,
            overall_quality_score=overall_score,
            details={
                "benchmark_name": benchmark_name,
                "test_retest_defined": test_retest_r is not None,
                "parallel_forms_defined": parallel_forms_r is not None,
                "criterion_defined": criterion_val is not None
            }
        )
