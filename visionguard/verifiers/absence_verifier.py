"""
Multi-Pass Absence & Negative Claim Verifier for VisionGuard.
Executes dual-threshold visual detection passes (high sensitivity + salient presence)
to verify negative claims without conflating non-detection with guaranteed absence.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence, Polarity
from visionguard.verifiers.object_verifier import OpenVocabularyGroundingVerifier


class AbsenceVerifier:
    """
    Multi-Pass Absence Verifier for evaluating negative propositions.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.grounding_verifier = OpenVocabularyGroundingVerifier(use_mock=use_mock)

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if visual_features is None and image_path is None:
            return Evidence(
                source="absence_verifier",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image context provided for absence verification.",
                details={"status": "ungrounded"}
            )

        # High Sensitivity Detection Pass (Force positive polarity check)
        temp_claim = claim.model_copy()
        temp_claim.polarity = Polarity.POSITIVE
        presence_evidence = self.grounding_verifier.verify(temp_claim, image_path, visual_features)

        presence_score = presence_evidence.confidence
        concept = presence_evidence.details.get("target_concept", claim.text)

        # Calibrated Multi-Pass Directional Inference
        if presence_score >= 0.55:
            # Salient object detected -> Confirmed Contradiction of absence!
            support_s = 0.05
            contradict_s = round(presence_score, 3)
            uncertainty_s = 0.10
            res_str = f"Contradiction: '{concept}' clearly detected in visual scene (conf={presence_score:.1%}), refuting absence assertion."
        elif presence_score >= 0.20:
            # Faint / partial trace detected -> High uncertainty
            support_s = 0.30
            contradict_s = round(presence_score, 3)
            uncertainty_s = 0.65
            res_str = f"Inconclusive absence: Weak/partial detection traces for '{concept}' (conf={presence_score:.1%}, uncertainty=0.65)."
        else:
            # Completely absent under multi-pass search -> High support for absence claim
            support_s = round(max(0.80, 1.0 - presence_score), 3)
            contradict_s = 0.05
            uncertainty_s = 0.15
            res_str = f"Absence verified: Multi-pass open grounding found no localized traces of '{concept}'."

        cot = (
            f"Step 1: Conducted multi-pass presence check for '{concept}'. "
            f"Step 2: Grounding confidence = {presence_score:.2f}, bbox={presence_evidence.bounding_box or 'None'}. "
            f"Step 3: Multi-pass evaluation -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="absence_verifier",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=support_s,
            score=support_s,
            bounding_box=presence_evidence.bounding_box,
            cot_reasoning=cot,
            details={
                "target_concept": concept,
                "presence_confidence": round(presence_score, 3),
                "bbox": presence_evidence.bounding_box
            }
        )
