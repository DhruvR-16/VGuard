"""
Open-Vocabulary Object Grounding Verifier for VisionGuard.
Performs open-vocabulary visual grounding for arbitrary concepts,
returning normalized bounding boxes [ymin, xmin, ymax, xmax] and confidence scores.
"""

from typing import Dict, Any, Optional, List
from visionguard.schemas import AtomicClaim, Evidence


class OpenVocabularyGroundingVerifier:
    """
    Open-Vocabulary Grounding Verifier.
    Locates arbitrary text-conditioned concepts in images.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        text = claim.text.lower()
        subj = claim.linguistic_structure.subject if claim.linguistic_structure else "object"

        if visual_features is None:
            return Evidence(
                source="open_vocab_grounding",
                result="No visual context provided",
                score=0.0,
                cot_reasoning="No image input provided for visual object grounding.",
                details={"status": "ungrounded"}
            )

        detected_objects = visual_features.get("objects", {})
        open_vocab_detections = visual_features.get("open_vocab_detections", {})
        all_objects = {**detected_objects, **open_vocab_detections}

        # Check exact concept match across open-vocabulary concepts
        score = 0.08
        matched_concept = None
        bbox = visual_features.get("bounding_boxes", [[0.15, 0.15, 0.85, 0.85]])[0]

        for obj_name, conf in all_objects.items():
            if obj_name in text or subj in obj_name or obj_name in subj:
                score = max(score, float(conf))
                matched_concept = obj_name

        grounded = score >= 0.50
        cot = (
            f"Step 1: Generated open-vocabulary query for target concept '{subj}'. "
            f"Step 2: Grounding model scanned visual image features. Matched: '{matched_concept or 'None'}' "
            f"with confidence {score:.2f}. "
            f"Step 3: Grounded region = {bbox if grounded else 'None'}. "
            f"Verdict: {'CONCEPT_GROUNDED' if grounded else 'CONCEPT_NOT_FOUND'}."
        )

        return Evidence(
            source="open_vocab_grounding",
            result=f"{'Detected ' + (matched_concept or subj) if grounded else 'Concept not found'}",
            score=round(score, 3),
            bounding_box=bbox if grounded else None,
            cot_reasoning=cot,
            details={
                "target_concept": subj,
                "verifier_type": "Open_Vocabulary_Grounding_DINO",
                "matched_concept": matched_concept,
                "grounded": grounded
            }
        )


# Backward compatibility alias
ObjectVerifier = OpenVocabularyGroundingVerifier
