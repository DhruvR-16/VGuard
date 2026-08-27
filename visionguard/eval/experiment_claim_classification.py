"""
Experiment 1: Real Claim-Level Classification Evaluator for VisionGuard.
Calculates true precision, recall, F1, macro F1, and per-category F1 over the benchmark dataset.
Zero hardcoded metrics.
"""

from typing import List, Dict, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.schemas import ClaimType, Verdict


class ClaimClassificationExperiment:
    """
    Evaluates fine-grained claim classification accuracy dynamically from data.
    """

    def __init__(self, use_mock: bool = False):
        self.orchestrator = CAMVOrchestrator(use_mock=use_mock)

    def run_experiment(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {"status": "no_data_provided", "error": "Dataset is empty."}

        y_true_binary = []
        y_pred_binary = []
        category_records: Dict[str, Dict[str, List[int]]] = {
            ct.value: {"true": [], "pred": []} for ct in ClaimType
        }

        total_claims = 0

        for sample in dataset:
            response = sample.get("response_text") or sample.get("model_response") or sample.get("ground_truth") or ""
            if not response:
                continue

            gt_verdict_str = sample.get("ground_truth_verdict", "Supported")
            gt_binary = 1 if gt_verdict_str == "Supported" else 0

            res = self.orchestrator.analyze(response, visual_features=sample.get("visual_features"))

            for claim_res in res.claims:
                total_claims += 1
                cat = claim_res.claim.type.value
                pred_binary = 1 if claim_res.verdict == Verdict.SUPPORTED else 0

                y_true_binary.append(gt_binary)
                y_pred_binary.append(pred_binary)

                if cat not in category_records:
                    category_records[cat] = {"true": [], "pred": []}
                category_records[cat]["true"].append(gt_binary)
                category_records[cat]["pred"].append(pred_binary)

        if not y_true_binary:
            return {"status": "no_claims_evaluated", "total_claims": 0}

        # Calculate True Metrics
        p, r, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary", zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        acc = accuracy_score(y_true_binary, y_pred_binary)

        per_category_metrics = {}
        for cat, data in category_records.items():
            if data["true"]:
                cat_p, cat_r, cat_f1, _ = precision_recall_fscore_support(data["true"], data["pred"], average="binary", zero_division=0)
                per_category_metrics[cat] = {
                    "count": len(data["true"]),
                    "precision": round(float(cat_p), 4),
                    "recall": round(float(cat_r), 4),
                    "f1_score": round(float(cat_f1), 4)
                }

        return {
            "total_claims_evaluated": total_claims,
            "overall_accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "per_category_metrics": per_category_metrics
        }
