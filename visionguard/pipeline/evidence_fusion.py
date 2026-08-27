"""
Directional Evidence Fusion Engine for VisionGuard.
Fuses multi-evidence directional distributions (support, contradiction, uncertainty)
using claim-aware weighted combinations and calibrated models.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from visionguard.schemas import AtomicClaim, Evidence, ClaimType, FusionMath


class EvidenceFusionEngine:
    """
    Directional Evidence Fusion Engine combining multi-verifier distributions.
    """

    def __init__(self, mode: str = "claim_aware"):
        self.mode = mode
        self.weights = {
            ClaimType.OBJECT_EXISTENCE: {"open_vocab_grounding": 0.85, "vlm_scene_context": 0.15},
            ClaimType.OBJECT_COUNT: {"open_vocab_grounding": 0.65, "spatial_geometry_verifier": 0.35},
            ClaimType.COLOR: {"clip_attribute_similarity": 0.75, "open_vocab_grounding": 0.25},
            ClaimType.ATTRIBUTE: {"clip_attribute_similarity": 0.75, "open_vocab_grounding": 0.25},
            ClaimType.ACTION: {"action_interaction_vlm": 0.75, "spatial_geometry_verifier": 0.25},
            ClaimType.INTERACTION: {"action_interaction_vlm": 0.70, "open_vocab_grounding": 0.30},
            ClaimType.SPATIAL_RELATION: {"spatial_geometry_verifier": 0.65, "action_interaction_vlm": 0.35},
            ClaimType.COMPARISON: {"spatial_geometry_verifier": 0.50, "clip_attribute_similarity": 0.50},
            ClaimType.OCR_TEXT: {"ocr_text_engine": 0.85, "clip_attribute_similarity": 0.15},
            ClaimType.SCENE: {"vlm_scene_context": 0.80, "clip_attribute_similarity": 0.20},
            ClaimType.EXTERNAL_KNOWLEDGE: {"vlm_scene_context": 1.00}
        }

    def fuse(self, claim: AtomicClaim, evidences: List[Evidence]) -> float:
        score, _, _, _ = self.fuse_directional(claim, evidences)
        return score

    def fuse_directional(
        self,
        claim: AtomicClaim,
        evidences: List[Evidence]
    ) -> Tuple[float, float, float, FusionMath]:
        """
        Fuses directional distributions across gathered evidence.
        Returns: (fused_support_score, fused_contradict_score, fused_uncertainty_score, FusionMath).
        """
        if not evidences:
            math_obj = FusionMath(
                formula="Fused = 0.0 (No Evidence)",
                weight_breakdown={},
                raw_scores={},
                support_fused=0.0,
                contradict_fused=0.0,
                uncertainty_fused=1.0,
                fused_score=0.0
            )
            return 0.0, 0.0, 1.0, math_obj

        claim_weights = self.weights.get(claim.type, {"default": 1.0})
        total_w = 0.0
        weighted_support = 0.0
        weighted_contradict = 0.0
        weighted_uncertainty = 0.0

        active_weights = {}
        raw_scores = {}
        terms = []

        for e in evidences:
            w = claim_weights.get(e.source, 0.50) if self.mode != "uniform" else (1.0 / len(evidences))
            active_weights[e.source] = round(w, 3)
            raw_scores[e.source] = round(e.support_score, 3)

            weighted_support += w * e.support_score
            weighted_contradict += w * e.contradiction_score
            weighted_uncertainty += w * e.uncertainty_score
            total_w += w
            terms.append(f"({w:.2f} × {e.support_score:.2f})")

        fused_supp = weighted_support / total_w if total_w > 0 else 0.0
        fused_cont = weighted_contradict / total_w if total_w > 0 else 0.0
        fused_unc = weighted_uncertainty / total_w if total_w > 0 else 1.0

        formula_str = (
            f"Support = ({' + '.join(terms)}) / {total_w:.2f} = {fused_supp:.3f} | Contradict = {fused_cont:.3f} | Uncertainty = {fused_unc:.3f}"
        )

        math_obj = FusionMath(
            formula=formula_str,
            weight_breakdown=active_weights,
            raw_scores=raw_scores,
            support_fused=round(fused_supp, 3),
            contradict_fused=round(fused_cont, 3),
            uncertainty_fused=round(fused_unc, 3),
            fused_score=round(fused_supp, 3)
        )

        return round(fused_supp, 3), round(fused_cont, 3), round(fused_unc, 3), math_obj

    def fuse_with_math(self, claim: AtomicClaim, evidences: List[Evidence]) -> Tuple[float, FusionMath]:
        supp, _, _, math_obj = self.fuse_directional(claim, evidences)
        return supp, math_obj
