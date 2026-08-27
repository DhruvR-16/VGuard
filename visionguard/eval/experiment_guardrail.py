"""
Experiment 6: Real Guardrail & Response Repair Effectiveness Evaluator for VisionGuard.
Measures real before/after hallucination mitigation, information retention,
repair success rate, and actual measured repair latency (ms).
Zero hardcoded metrics or synthetic multipliers.
"""

import time
from typing import List, Dict, Any
import numpy as np
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.schemas import Verdict


class GuardrailEffectivenessExperiment:
    """
    Evaluates real response repair effectiveness and hallucination reduction on dataset samples.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.orchestrator = CAMVOrchestrator(use_mock=use_mock)

    def run_experiment(self, sample_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sample_dataset:
            return {"status": "no_data_provided", "mitigation_metrics": {}}

        total_claims_before = 0
        total_claims_retained = 0
        successful_repairs = 0
        repair_latencies_ms = []

        before_main_hals = []
        before_extra_counts = []
        before_overall_hals = []

        after_main_hals = []
        after_extra_counts = []
        after_overall_hals = []

        for sample in sample_dataset:
            resp = sample.get("response_text") or sample.get("model_response") or sample.get("claim") or ""
            if not resp:
                continue

            image_path = sample.get("full_image_path") or sample.get("image_path")
            visual_features = sample.get("visual_features")

            # 1. Analyze and Measure Real Repair Latency over Raw Images
            t0 = time.perf_counter()
            res = self.orchestrator.analyze(
                response_text=resp,
                image_input=image_path,
                visual_features=visual_features if not image_path else None,
                enable_repair=True
            )
            t1 = time.perf_counter()
            repair_latencies_ms.append((t1 - t0) * 1000.0)

            total_claims_before += len(res.claims)

            # Before Metrics
            before_main_hals.append(res.decomposed_metrics.get("main_hal_rate", 0.0))
            before_extra_counts.append(res.decomposed_metrics.get("extra_hal_count", 0.0))
            before_overall_hals.append(res.decomposed_metrics.get("overall_hal_rate", 0.0))

            # After Metrics (from Re-Verified Response)
            if res.repaired_response:
                re_claims = res.repaired_response.re_verified_claims
                total_claims_retained += len(res.repaired_response.verified_facts)

                if res.repaired_response.re_verified_verdict == Verdict.SUPPORTED:
                    successful_repairs += 1

                # Calculate after decomposed metrics from re_verified_claims
                if re_claims:
                    re_main_hal = 1.0 if re_claims[0].verdict in [Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE] else 0.0
                    re_extra_count = float(len([c for c in re_claims[1:] if c.verdict in [Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE]]))
                    re_overall_hal = 1.0 if (re_main_hal > 0 or re_extra_count > 0) else 0.0
                else:
                    re_main_hal = 0.0
                    re_extra_count = 0.0
                    re_overall_hal = 0.0

                after_main_hals.append(re_main_hal)
                after_extra_counts.append(re_extra_count)
                after_overall_hals.append(re_overall_hal)
            else:
                after_main_hals.append(before_main_hals[-1])
                after_extra_counts.append(before_extra_counts[-1])
                after_overall_hals.append(before_overall_hals[-1])

        n = len(sample_dataset)
        avg_before_main = float(np.mean(before_main_hals)) if before_main_hals else 0.0
        avg_before_extra = float(np.mean(before_extra_counts)) if before_extra_counts else 0.0
        avg_before_overall = float(np.mean(before_overall_hals)) if before_overall_hals else 0.0

        avg_after_main = float(np.mean(after_main_hals)) if after_main_hals else 0.0
        avg_after_extra = float(np.mean(after_extra_counts)) if after_extra_counts else 0.0
        avg_after_overall = float(np.mean(after_overall_hals)) if after_overall_hals else 0.0

        reduction_rate = (avg_before_overall - avg_after_overall) / avg_before_overall if avg_before_overall > 0 else 0.0
        retention_rate = total_claims_retained / total_claims_before if total_claims_before > 0 else 1.0
        repair_success_rate = successful_repairs / n if n > 0 else 0.0
        avg_latency = float(np.mean(repair_latencies_ms)) if repair_latencies_ms else 0.0

        return {
            "num_test_responses": n,
            "before_guardrail": {
                "main_hal_rate": round(avg_before_main, 4),
                "extra_hal_count": round(avg_before_extra, 3),
                "overall_hal_rate": round(avg_before_overall, 4)
            },
            "after_guardrail": {
                "main_hal_rate": round(avg_after_main, 4),
                "extra_hal_count": round(avg_after_extra, 3),
                "overall_hal_rate": round(avg_after_overall, 4)
            },
            "mitigation_metrics": {
                "hallucination_reduction": f"{reduction_rate * 100:.1f}%",
                "information_retention_rate": f"{retention_rate * 100:.1f}%",
                "constrained_repair_success_rate": f"{repair_success_rate * 100:.1f}%",
                "average_repair_latency_ms": round(avg_latency, 2)
            },
            "key_finding": f"Response repair reduces overall hallucination rate from {avg_before_overall:.1%} down to {avg_after_overall:.1%} ({reduction_rate * 100:.1f}% reduction) with average repair latency of {avg_latency:.1f}ms."
        }
