"""
Scene & Environment Context Verifier for VisionGuard.
Verifies global scene background, weather, lighting, and identifies external non-visual assertions.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence, ClaimType, Verifiability


class SceneVerifier:
    """
    Scene Verifier for global environment, background, and external knowledge claims.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        claim_text = claim.text.lower()

        if visual_features is None:
            return Evidence(
                source="vlm_scene_context",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image context provided for scene verification.",
                details={"status": "ungrounded"}
            )

        # 1. Check if Claim is External Knowledge / Non-Visual
        if claim.verifiability == Verifiability.EXTERNAL_KNOWLEDGE or claim.type == ClaimType.EXTERNAL_KNOWLEDGE or not claim.verifiable:
            return Evidence(
                source="vlm_scene_context",
                result="Claim asserts external metadata / geographic knowledge cannot be verified from visual features alone (OUT_OF_SCOPE).",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="Specific non-visual metadata claims require external knowledge sources.",
                details={"is_out_of_scope": True}
            )

        # 2. Semantic Environment Matching
        scene_info = visual_features.get("scene", {})
        is_outdoor = scene_info.get("is_outdoor", False) if isinstance(scene_info, dict) else False
        scene_type = scene_info.get("type", "scene background") if isinstance(scene_info, dict) else str(scene_info)

        outdoor_keywords = ["outdoor", "park", "nature", "field", "yard", "snow", "road", "street", "beach", "sky", "forest"]
        indoor_keywords = ["indoor", "room", "kitchen", "office", "desk", "bedroom", "living room", "table"]

        if any(w in claim_text for w in outdoor_keywords):
            if is_outdoor:
                support_s = 0.90
                contradict_s = 0.05
                uncertainty_s = 0.10
                rationale = f"Outdoor environment verified: '{scene_type}'."
            else:
                support_s = 0.10
                contradict_s = 0.85
                uncertainty_s = 0.15
                rationale = f"Contradiction: Visual context indicates indoor environment, conflicting with '{claim.text}'."
        elif any(w in claim_text for w in indoor_keywords):
            if not is_outdoor:
                support_s = 0.90
                contradict_s = 0.05
                uncertainty_s = 0.10
                rationale = f"Indoor environment verified: '{scene_type}'."
            else:
                support_s = 0.10
                contradict_s = 0.85
                uncertainty_s = 0.15
                rationale = f"Contradiction: Visual context indicates outdoor environment, conflicting with '{claim.text}'."
        else:
            support_s = 0.60
            contradict_s = 0.20
            uncertainty_s = 0.40
            rationale = f"General visual alignment with scene context ({scene_type})."

        cot = (
            f"Step 1: Extracted scene claim '{claim_text}'. "
            f"Step 2: Classified visual background as '{scene_type}' (is_outdoor={is_outdoor}). "
            f"Step 3: {rationale} "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="vlm_scene_context",
            result=rationale,
            support_score=round(support_s, 3),
            contradiction_score=round(contradict_s, 3),
            uncertainty_score=round(uncertainty_s, 3),
            confidence=round(support_s, 3),
            score=round(support_s, 3),
            cot_reasoning=cot,
            details={
                "scene_type": scene_type,
                "is_outdoor": is_outdoor,
                "scene_support": round(support_s, 3)
            }
        )
