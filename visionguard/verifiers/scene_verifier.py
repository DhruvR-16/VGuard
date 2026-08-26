"""
Scene & Out-of-Scope Verifier for VisionGuard.
Verifies global scene environment, background, and location claims.
Explicitly identifies OUT_OF_SCOPE assertions (e.g. ungrounded geographic city names)
that cannot be verified from visual features alone.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence, ClaimType


class SceneVerifier:
    """
    Specialized verifier for Scene Claims and Out-of-Scope identification.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        text = claim.text.lower()
        scope = claim.linguistic_structure.spatial_scope if claim.linguistic_structure else "scene"

        if visual_features is None:
            return Evidence(
                source="vlm_scene_context",
                result="No visual context provided",
                score=0.0,
                cot_reasoning="No image input provided for scene verification.",
                details={"status": "ungrounded"}
            )

        # 1. Check for Out-of-Scope External Knowledge Claims
        unverifiable_cities = ["paris", "tokyo", "london", "new york", "beijing", "berlin", "rome", "san francisco"]
        if claim.type == ClaimType.OUT_OF_SCOPE or any(city in text for city in unverifiable_cities):
            return Evidence(
                source="vlm_scene_context",
                result="Geographic / external claim cannot be verified from visual context alone (OUT_OF_SCOPE)",
                score=0.05,
                cot_reasoning="Specific city/location claims cannot be verified from visual context alone without external metadata.",
                details={"is_out_of_scope": True, "verifier_type": "External_Knowledge_Filter"}
            )

        scene_info = visual_features.get("scene", {})
        if isinstance(scene_info, dict):
            scene_type = scene_info.get("type", "unspecified scene background")
            is_outdoor = scene_info.get("is_outdoor", False)
            is_uncertain = scene_info.get("is_uncertain", False)
        elif isinstance(scene_info, str):
            scene_type = scene_info
            is_outdoor = any(w in scene_info.lower() for w in ["park", "yard", "outdoor", "snow", "field", "street", "nature"])
            is_uncertain = False
        else:
            scene_type = "unspecified scene background"
            is_outdoor = False
            is_uncertain = True

        if is_uncertain:
            score = 0.50
            rationale = "Scene background is ambiguous or not clearly identifiable from visual features."
        elif "outdoor" in text or "park" in text or "nature" in text or "field" in text or "yard" in text or "snow" in text:
            score = 0.90 if is_outdoor else 0.15
            rationale = f"Environment matched as {'Outdoor' if is_outdoor else 'Contradicted (Indoor)'}."
        elif "indoor" in text or "room" in text or "kitchen" in text or "desk" in text:
            score = 0.90 if not is_outdoor else 0.15
            rationale = f"Environment matched as {'Indoor' if not is_outdoor else 'Contradicted (Outdoor)'}."
        else:
            score = 0.60
            rationale = f"General visual context alignment score: {score:.2f}."

        cot = (
            f"Step 1: Extracted scene context '{scope}'. "
            f"Step 2: Environment classification: '{scene_type}' (is_uncertain={is_uncertain}). "
            f"Step 3: {rationale} Score = {score:.2f}."
        )

        return Evidence(
            source="vlm_scene_context",
            result=f"Scene matched: {scene_type} (Score {score:.2f})",
            score=round(score, 3),
            cot_reasoning=cot,
            details={
                "detected_scene": scene_type,
                "verifier_type": "Visual_Environment_Scene_Classifier",
                "is_outdoor": is_outdoor,
                "is_uncertain": is_uncertain,
                "scene_match_score": round(score, 3)
            }
        )
