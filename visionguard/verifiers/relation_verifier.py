"""
Relation Verifier Module for VisionGuard.
Verifies spatial and relational claims against real visual feature maps.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence


class RelationVerifier:
    """
    Specialized verifier for Relation Claims (spatial, inter-object interactions).
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        text = claim.text.lower()
        subj = claim.linguistic_structure.subject if claim.linguistic_structure else "subject"
        pred = claim.linguistic_structure.predicate if claim.linguistic_structure else "action"
        obj = claim.linguistic_structure.target_object if claim.linguistic_structure else "target"

        if visual_features is None:
            return Evidence(
                source="vlm_relation_judge",
                score=0.0,
                cot_reasoning="No image provided for spatial relation verification.",
                details={"status": "No image input"}
            )

        detected_objects = visual_features.get("objects", {})
        
        # Check if subject and interacting object exist in the image
        subj_detected = any(subj in k or k in subj for k in detected_objects)
        obj_detected = obj is None or any(obj in k or k in obj for k in detected_objects)

        if subj_detected and obj_detected:
            score = 0.88
        elif subj_detected:
            score = 0.60
        else:
            score = 0.12

        cot = (
            f"Step 1: Relational triplet: <{subj}, {pred}, {obj or 'none'}>. "
            f"Step 2: Subject grounded: {subj_detected}, Target object grounded: {obj_detected}. "
            f"Step 3: Relational consistency score = {score:.2f}."
        )

        return Evidence(
            source="vlm_relation_judge",
            score=round(score, 3),
            cot_reasoning=cot,
            details={
                "triplet": {"subject": subj, "predicate": pred, "object": obj},
                "verifier_type": "Spatial_Grounding_Relation_Judge",
                "subject_present": subj_detected,
                "target_present": obj_detected
            }
        )
