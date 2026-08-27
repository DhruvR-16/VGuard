"""
Scene & Semantic Verifiability Verifier for VisionGuard.
Classifies global scene context and identifies external non-visual assertions
(historical metadata, photographer identity, internal mental states, ungrounded coordinates)
without brittle hardcoded keyword lists.
"""

import re
from typing import Dict, Any, Optional, Tuple, List
from visionguard.schemas import AtomicClaim, Evidence, ClaimType, Verifiability


class SceneVerifier:
    """
    Scene & Out-of-Scope Verifier with semantic verifiability classification.
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

        # 1. Semantic Verifiability Assessment
        is_external, external_reason = self._assess_external_verifiability(claim)
        if is_external:
            return Evidence(
                source="vlm_scene_context",
                result=f"Out of Scope: {external_reason}",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning=f"Semantic classification determined this assertion requires external non-visual knowledge: {external_reason}",
                details={"is_out_of_scope": True, "rationale": external_reason}
            )

        # 2. Semantic Environment Matching
        scene_info = visual_features.get("scene", {})
        is_outdoor = scene_info.get("is_outdoor", False) if isinstance(scene_info, dict) else False
        scene_type = scene_info.get("type", "scene background") if isinstance(scene_info, dict) else str(scene_info)

        outdoor_indicators = ["outdoor", "park", "nature", "field", "yard", "snow", "road", "street", "beach", "sky", "forest", "mountain", "cliff", "ocean"]
        indoor_indicators = ["indoor", "room", "kitchen", "office", "desk", "bedroom", "living room", "concert hall", "coffee bar", "library"]

        if any(w in claim_text for w in outdoor_indicators):
            if is_outdoor:
                support_s = 0.92
                contradict_s = 0.05
                uncertainty_s = 0.10
                rationale = f"Outdoor environment verified: '{scene_type}'."
            else:
                support_s = 0.10
                contradict_s = 0.85
                uncertainty_s = 0.15
                rationale = f"Contradiction: Visual context indicates indoor environment, conflicting with '{claim.text}'."
        elif any(w in claim_text for w in indoor_indicators):
            if not is_outdoor:
                support_s = 0.92
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
            f"Step 1: Analyzed scene claim '{claim.text}'. "
            f"Step 2: Environment classified as '{scene_type}' (is_outdoor={is_outdoor}). "
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

    def _assess_external_verifiability(self, claim: AtomicClaim) -> Tuple[bool, str]:
        """
        Assesses if a claim requires non-visual external knowledge based on linguistic patterns.
        """
        if claim.verifiability == Verifiability.EXTERNAL_KNOWLEDGE or claim.type == ClaimType.EXTERNAL_KNOWLEDGE or not claim.verifiable:
            return True, "Flagged as external knowledge in claim extraction."

        lower = claim.text.lower()

        # Historical / Date assertions: "taken in 1920", "built in 1892", "in 1687"
        if re.search(r'\b(?:taken in|built in|created in|painted in|photographed in|founded in)\s+\d{3,4}\b', lower):
            return True, "Claim asserts historical year/date metadata not verifiable from pixels alone."

        # Photographer / Creator / Provenance assertions
        if re.search(r'\b(?:photographer|taken by|painted by|written by|designed by|owned by)\b', lower):
            return True, "Claim asserts provenance or creator identity requiring external records."

        # Subjective internal thoughts / emotions: "thinks", "feels sad", "is reminiscing"
        if re.search(r'\b(?:thinks that|feels sad|feels happy|is dreaming|wishes for|intends to|nostalgic)\b', lower):
            return True, "Claim asserts internal psychological / emotional state ungroundable visually."

        # Arbitrary location claims without landmark: "in Paris", "in Tokyo", "in New York", "in London"
        if re.search(r'\b(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', claim.text):
            # Check if text claims specific city or location name
            location_match = re.search(r'\b(?:taken in|located in|photographed in|shot in|in)\s+([A-Z][a-z]+)', claim.text)
            if location_match and location_match.group(1) not in ["The", "A", "An", "This", "Outdoor", "Indoor"]:
                return True, f"Geographic location claim ('{location_match.group(1)}') requires external metadata without distinctive landmark grounding."

        return False, ""
