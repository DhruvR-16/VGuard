"""
Experiment 1: Real Raw-Image Claim-Level Classification Evaluator for VisionGuard (CAM-V).
Evaluates atomic claims directly over raw image paths without synthetic feature injection.
Calculates both:
1. Binary Claim Support Detection (Supported vs. Non-Supported) with Precision, Recall, F1, AUROC, and 95% Bootstrap CIs.
2. Comprehensive 4-Class Multiclass Evaluation (SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE)
   with Macro F1, Weighted F1, 4x4 Confusion Matrix, Per-Class Metrics, and Expected Calibration Error (ECE).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.schemas import ClaimType, Verdict
from visionguard.data_loader import get_atomic_claims_dataset


class ClaimClassificationExperiment:
    """
    Evaluates fine-grained claim classification accuracy dynamically from raw image benchmarks.
    Zero synthetic feature injection; evaluates genuine 4-class uncertainty distributions.
    """

    FOUR_CLASSES = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"]

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.orchestrator = CAMVOrchestrator(use_mock=use_mock)

    def _normalize_verdict_str(self, v_str: Any) -> str:
        if not v_str:
            return "INSUFFICIENT_EVIDENCE"
        s = str(v_str).strip().upper()
        if "SUPPORT" in s:
            return "SUPPORTED"
        elif "CONTRADICT" in s:
            return "CONTRADICTED"
        elif "OUT" in s or "SCOPE" in s or "EXTERNAL" in s:
            return "OUT_OF_SCOPE"
        elif "INSUFFICIENT" in s or "UNCERTAIN" in s:
            return "INSUFFICIENT_EVIDENCE"
        return "INSUFFICIENT_EVIDENCE"

    def _calculate_ece(self, confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
        """
        Calculates Expected Calibration Error (ECE) across confidence bins.
        """
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = [
                j for j, c in enumerate(confidences)
                if bin_lower <= c < bin_upper or (i == n_bins - 1 and bin_lower <= c <= bin_upper)
            ]
            if in_bin:
                bin_acc = float(np.mean([accuracies[j] for j in in_bin]))
                bin_conf = float(np.mean([confidences[j] for j in in_bin]))
                bin_prop = len(in_bin) / n
                ece += bin_prop * abs(bin_acc - bin_conf)

        return round(float(ece), 4)

    def run_experiment(self, dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Fallback to atomic claims dataset if dataset is not provided or lacks atomic claim texts
        if dataset is None or (dataset and "claim" not in dataset[0] and "response_text" not in dataset[0]):
            dataset = get_atomic_claims_dataset()

        if not dataset:
            return {"status": "no_data_provided", "error": "Dataset is empty."}

        y_true_4class = []
        y_pred_4class = []
        y_true_binary = []
        y_pred_binary = []
        confidences = []
        binary_accuracies = []

        category_records_bin: Dict[str, Dict[str, List[int]]] = {
            ct.value: {"true": [], "pred": []} for ct in ClaimType
        }
        category_records_4class: Dict[str, Dict[str, List[str]]] = {
            ct.value: {"true": [], "pred": []} for ct in ClaimType
        }

        total_claims = 0

        for sample in dataset:
            claim_text = sample.get("claim") or sample.get("response_text") or sample.get("model_response") or ""
            if not claim_text:
                continue

            gt_raw = sample.get("ground_truth") or sample.get("ground_truth_verdict") or "SUPPORTED"
            gt_4class = self._normalize_verdict_str(gt_raw)
            gt_binary = 1 if gt_4class == "SUPPORTED" else 0

            # Raw image path without precomputed visual features
            image_path = sample.get("full_image_path") or sample.get("image_path")
            visual_features = sample.get("visual_features") if not image_path else None

            res = self.orchestrator.analyze(
                response_text=claim_text,
                image_input=image_path,
                visual_features=visual_features
            )

            pred_4class = self._normalize_verdict_str(res.overall_verdict.value)
            pred_binary = 1 if pred_4class == "SUPPORTED" else 0
            conf = res.overall_confidence

            total_claims += 1
            y_true_4class.append(gt_4class)
            y_pred_4class.append(pred_4class)
            y_true_binary.append(gt_binary)
            y_pred_binary.append(pred_binary)
            confidences.append(conf)
            binary_accuracies.append(1 if gt_binary == pred_binary else 0)

            # Category tracking
            c_type = sample.get("claim_type", "object_existence")
            if c_type in category_records_bin:
                category_records_bin[c_type]["true"].append(gt_binary)
                category_records_bin[c_type]["pred"].append(pred_binary)
                category_records_4class[c_type]["true"].append(gt_4class)
                category_records_4class[c_type]["pred"].append(pred_4class)

        if not y_true_binary:
            return {"status": "no_claims_evaluated", "total_claims": 0}

        # ---------------------------------------------------------------------
        # 1. Binary Claim Support Metrics
        # ---------------------------------------------------------------------
        bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        bin_macro_p, bin_macro_r, bin_macro_f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="macro", zero_division=0
        )
        bin_acc = accuracy_score(y_true_binary, y_pred_binary)

        try:
            bin_auroc = float(roc_auc_score(y_true_binary, confidences)) if len(set(y_true_binary)) > 1 else 0.50
        except Exception:
            bin_auroc = 0.50

        # Bootstrap 95% CI for Binary F1
        f1_boots = []
        n_samples = len(y_true_binary)
        rng = np.random.RandomState(42)
        for _ in range(200):
            idxs = rng.choice(n_samples, size=n_samples, replace=True)
            y_t_b = [y_true_binary[i] for i in idxs]
            y_p_b = [y_pred_binary[i] for i in idxs]
            if len(set(y_t_b)) > 0:
                _, _, f1_b, _ = precision_recall_fscore_support(y_t_b, y_p_b, average="binary", zero_division=0)
                f1_boots.append(float(f1_b))

        bin_f1_ci_95 = (
            [round(float(np.percentile(f1_boots, 2.5)), 4), round(float(np.percentile(f1_boots, 97.5)), 4)]
            if f1_boots else [round(float(bin_f1), 4), round(float(bin_f1), 4)]
        )

        # ---------------------------------------------------------------------
        # 2. Comprehensive 4-Class Multiclass Metrics
        # ---------------------------------------------------------------------
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_4class, y_pred_4class, labels=self.FOUR_CLASSES, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true_4class, y_pred_4class, labels=self.FOUR_CLASSES, average="weighted", zero_division=0
        )
        per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
            y_true_4class, y_pred_4class, labels=self.FOUR_CLASSES, average=None, zero_division=0
        )
        multiclass_acc = accuracy_score(y_true_4class, y_pred_4class)
        cm_matrix = confusion_matrix(y_true_4class, y_pred_4class, labels=self.FOUR_CLASSES).tolist()
        ece_val = self._calculate_ece(confidences, binary_accuracies)

        four_class_breakdown = {}
        for idx, c_name in enumerate(self.FOUR_CLASSES):
            four_class_breakdown[c_name] = {
                "support_count": int(per_class_sup[idx]),
                "precision": round(float(per_class_p[idx]), 4),
                "recall": round(float(per_class_r[idx]), 4),
                "f1_score": round(float(per_class_f1[idx]), 4)
            }

        # ---------------------------------------------------------------------
        # 3. Per-Category Claim Metrics
        # ---------------------------------------------------------------------
        per_category_metrics = {}
        for cat, data in category_records_bin.items():
            if data["true"]:
                cat_p, cat_r, cat_f1, _ = precision_recall_fscore_support(
                    data["true"], data["pred"], average="binary", zero_division=0
                )
                per_category_metrics[cat] = {
                    "count": len(data["true"]),
                    "precision": round(float(cat_p), 4),
                    "recall": round(float(cat_r), 4),
                    "f1_score": round(float(cat_f1), 4)
                }

        return {
            "total_claims_evaluated": total_claims,
            "binary_support_evaluation": {
                "metric_description": "Binary Support Detection: Supported vs. Not Supported",
                "overall_accuracy": round(float(bin_acc), 4),
                "precision": round(float(bin_p), 4),
                "recall": round(float(bin_r), 4),
                "f1_score": round(float(bin_f1), 4),
                "macro_f1": round(float(bin_macro_f1), 4),
                "auroc": round(float(bin_auroc), 4),
                "f1_ci_95": bin_f1_ci_95,
                "expected_calibration_error": ece_val
            },
            "four_class_evaluation": {
                "metric_description": "4-State Classification: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE",
                "multiclass_accuracy": round(float(multiclass_acc), 4),
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "macro_f1": round(float(macro_f1), 4),
                "weighted_f1": round(float(weighted_f1), 4),
                "confusion_matrix": {
                    "labels": self.FOUR_CLASSES,
                    "matrix": cm_matrix
                },
                "per_class_metrics": four_class_breakdown
            },
            "per_category_metrics": per_category_metrics
        }
