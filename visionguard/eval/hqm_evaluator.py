"""
HQM Framework Evaluator for VisionGuard (arXiv:2406.17115v3).
Calculates Test-Retest Reliability, Parallel-Forms Reliability, Content Validity, and Criterion Validity.
"""

from typing import List, Dict, Any
import numpy as np
from scipy.stats import pearsonr
from visionguard.schemas import HQMMetrics


class HQMEvaluator:
    """
    Implements psychometric quality evaluation for hallucination benchmarks.
    """

    def __init__(self):
        pass

    def calculate_reliability(self, scores_a: List[float], scores_b: List[float]) -> float:
        """
        Calculates Pearson correlation between two evaluation runs (Test-Retest or Parallel-Forms).
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return 1.0
        
        arr_a = np.array(scores_a, dtype=float)
        arr_b = np.array(scores_b, dtype=float)

        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return 1.0

        r, _ = pearsonr(arr_a, arr_b)
        return float(np.clip(r if not np.isnan(r) else 1.0, -1.0, 1.0))

    def calculate_content_validity(self, valid_sample_count: int, total_sample_count: int) -> float:
        """
        Calculates content validity ratio N_valid / N.
        """
        if total_sample_count == 0:
            return 1.0
        return float(np.clip(valid_sample_count / total_sample_count, 0.0, 1.0))

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

        return HQMMetrics(
            benchmark_name=benchmark_name,
            test_retest_reliability=round(test_retest_r, 4),
            parallel_forms_reliability=round(parallel_forms_r, 4),
            content_validity=round(content_val, 4),
            criterion_validity=round(criterion_val, 4)
        )
