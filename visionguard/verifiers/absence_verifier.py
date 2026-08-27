"""
Negative Claim & Absence Verifier for VisionGuard.
Verifies assertions of non-existence or absence with calibrated uncertainty
to ensure non-detection is not conflated with guaranteed absence.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence, Polarity
from visionguard.verifiers.object_verifier import OpenVocabularyGroundingVerifier


class AbsenceVerifier:
    """
    Absence Verifier for evaluating negative propositions.
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

        # Delegate to open-vocabulary grounding with forced POSITIVE polarity to check presence
        temp_claim = claim.model_copy()
        temp_claim.polarity = Polarity.POSITIVE
        presence_evidence = self.grounding_verifier.verify(temp_claim, image_path, visual_features)

        presence_score = presence_evidence.confidence
        concept = presence_evidence.details.get("target_concept", claim.text)

        # Directional evidence for negative claim
        if presence_score >= 0.60:
            # Concept clearly detected -> Contradicts absence claim!
            support_s = 0.05
            contradict_s = round(presence_score, 3)
            uncertainty_s = 0.10
            res_str = f"Contradiction: '{concept}' is present (detector confidence {presence_score:.1%}), refuting absence claim."
        elif presence_score >= 0.25:
            # Ambiguous presence -> High uncertainty
            support_s = 0.35
            contradict_s = round(presence_score, 3)
            uncertainty_s = 0.60
            res_str = f"Inconclusive absence: Weak visual traces of '{concept}' detected (confidence {presence_score:.1%})."
        else:
            # Not detected -> Supports absence with calibrated confidence
            support_s = round(max(0.75, 1.0 - presence_score), 3)
            contradict_s = 0.05
            uncertainty_s = 0.20
            res_str = f"Absence verified: '{concept}' was not localized by open-vocabulary visual grounding."

        cot = (
            f"Step 1: Tested presence of concept '{concept}' across visual field. "
            f"Step 2: Grounding presence confidence = {presence_score:.2f}. "
            f"Step 3: Directional evidence for absence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="absence_verifier",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=support_s,
            score=support_s,
            cot_reasoning=cot,
            details={
                "target_concept": concept,
                "presence_confidence": round(presence_score, 3),
                "bbox": presence_evidence.bounding_box
            }
        )
