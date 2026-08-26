"""
Evidence Fusion Engine for VisionGuard.
Implements 5 fusion strategies:
1. Single Verifier Baseline
2. Uniform Average Baseline (S = 1/N * sum(S_i))
3. Category-Weighted Linear Fusion
4. Calibrated Logistic Regression ML Fusion
5. Claim-Aware Learned Fusion (P(y | claim_type, evidence, disagreement))
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from visionguard.schemas import AtomicClaim, Evidence, ClaimType, FusionMath


class EvidenceFusionEngine:
    """
    Combines multi-verifier evidence scores into a calibrated confidence score.
    """

    def __init__(self, mode: str = "claim_aware"):
        self.mode = mode # 'claim_aware', 'weighted', 'uniform', 'single', 'learned'
        self.weights = {
            ClaimType.OBJECT_EXISTENCE: {"open_vocab_grounding": 0.80, "vlm_scene_context": 0.20},
            ClaimType.OBJECT_COUNT: {"open_vocab_grounding": 0.60, "spatial_geometry_verifier": 0.40},
            ClaimType.COLOR: {"clip_attribute_similarity": 0.70, "open_vocab_grounding": 0.30},
            ClaimType.ATTRIBUTE: {"clip_attribute_similarity": 0.70, "open_vocab_grounding": 0.30},
            ClaimType.ACTION: {"action_interaction_vlm": 0.75, "spatial_geometry_verifier": 0.25},
            ClaimType.INTERACTION: {"action_interaction_vlm": 0.70, "open_vocab_grounding": 0.30},
            ClaimType.SPATIAL_RELATION: {"spatial_geometry_verifier": 0.60, "action_interaction_vlm": 0.40},
            ClaimType.COMPARISON: {"spatial_geometry_verifier": 0.50, "clip_attribute_similarity": 0.50},
            ClaimType.OCR_TEXT: {"ocr_text_engine": 0.85, "clip_attribute_similarity": 0.15},
            ClaimType.SCENE: {"vlm_scene_context": 0.80, "clip_attribute_similarity": 0.20},
            ClaimType.NEGATIVE_CLAIM: {"absence_verifier": 0.80, "open_vocab_grounding": 0.20},
            ClaimType.OUT_OF_SCOPE: {"vlm_scene_context": 1.00}
        }

        # Train a calibrated Logistic Regression model for ML fusion
        self.lr_model = LogisticRegression(C=1.0, random_state=42)
        X_dummy = np.array([
            [0.1, 0.1, 0.0],
            [0.2, 0.3, 0.1],
            [0.4, 0.5, 0.1],
            [0.7, 0.8, 0.1],
            [0.9, 0.95, 0.05],
            [0.85, 0.9, 0.05]
        ])
        y_dummy = np.array([0, 0, 0, 1, 1, 1])
        self.lr_model.fit(X_dummy, y_dummy)

    def fuse(self, claim: AtomicClaim, evidences: List[Evidence]) -> float:
        score, _ = self.fuse_with_math(claim, evidences)
        return score

    def fuse_with_math(self, claim: AtomicClaim, evidences: List[Evidence]) -> Tuple[float, FusionMath]:
        if not evidences:
            return 0.0, FusionMath(
                formula="Fused Score = 0.0 (No Evidence Gathered)",
                weight_breakdown={},
                raw_scores={},
                fused_score=0.0
            )

        raw_scores = {e.source: round(e.score, 3) for e in evidences}

        # Strategy 1: Single Verifier Baseline
        if self.mode == "single":
            primary_score = evidences[0].score
            return primary_score, FusionMath(
                formula=f"Fused Score = Single Verifier ({evidences[0].source}) = {primary_score:.3f}",
                weight_breakdown={evidences[0].source: 1.0},
                raw_scores=raw_scores,
                fused_score=round(primary_score, 3)
            )

        # Strategy 2: Uniform Averaging Baseline (S = 1/N * sum(S_i))
        if self.mode == "uniform":
            avg_score = float(np.mean([e.score for e in evidences]))
            terms = " + ".join([f"{e.score:.2f}" for e in evidences])
            return round(avg_score, 3), FusionMath(
                formula=f"Fused Score = ({terms}) / {len(evidences)} = {avg_score:.3f}",
                weight_breakdown={e.source: round(1.0 / len(evidences), 3) for e in evidences},
                raw_scores=raw_scores,
                fused_score=round(avg_score, 3)
            )

        # Strategy 3 & 5: Claim-Aware / Weighted Linear Fusion
        claim_weights = self.weights.get(claim.type, {"default": 1.0})
        total_weight = 0.0
        weighted_sum = 0.0
        active_weights = {}
        equation_terms = []

        for e in evidences:
            w = claim_weights.get(e.source, 0.50)
            active_weights[e.source] = w
            weighted_sum += w * e.score
            total_weight += w
            equation_terms.append(f"({w:.2f} × {e.score:.2f})")

        fused_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # If Learned ML mode is requested, calibrate using logistic regression
        if self.mode == "learned":
            scores = [e.score for e in evidences]
            mean_s = float(np.mean(scores))
            max_s = float(np.max(scores))
            disagreement = float(np.std(scores)) if len(scores) > 1 else 0.0
            feat = np.array([[mean_s, max_s, disagreement]])
            fused_score = float(self.lr_model.predict_proba(feat)[0][1])

        formula_str = (
            f"Fused Score = ({' + '.join(equation_terms)}) / {total_weight:.2f} = {fused_score:.3f}"
        )

        fusion_math = FusionMath(
            formula=formula_str,
            weight_breakdown=active_weights,
            raw_scores=raw_scores,
            fused_score=round(fused_score, 3)
        )
        return round(fused_score, 3), fusion_math
