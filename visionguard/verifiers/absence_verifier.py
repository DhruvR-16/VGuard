"""
Absence & Negative Claim Verifier for VisionGuard.
Verifies negative assertions (e.g., 'There are no cars', 'The man is not wearing a hat')
by evaluating the non-existence / absence of target concepts in the image.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence


class AbsenceVerifier:
    """
    Evaluates negative claims asserting non-existence or absence.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        subj = claim.linguistic_structure.subject if claim.linguistic_structure else "object"

        if visual_features is None:
            return Evidence(
                source="absence_verifier",
                result="No visual context provided",
                score=0.0,
                cot_reasoning="No image input provided for absence verification.",
                details={"status": "ungrounded"}
            )

        detected_objects = visual_features.get("objects", {})
        open_vocab_detections = visual_features.get("open_vocab_detections", {})
        all_objects = {**detected_objects, **open_vocab_detections}

        # Check if the supposedly absent concept is actually present
        detected_presence_score = 0.0
        for obj_name, conf in all_objects.items():
            if obj_name in subj.lower() or subj.lower() in obj_name:
                detected_presence_score = max(detected_presence_score, float(conf))

        # Invert score: High absence confidence = low presence
        if detected_presence_score > 0.50:
            absence_score = 0.10 # Concept IS present, so negative claim is false!
            result_str = f"Concept '{subj}' is PRESENT in image (presence score {detected_presence_score:.2f}). Negative claim contradicted."
        else:
            absence_score = 0.92 # Concept is truly absent!
            result_str = f"Concept '{subj}' confirmed ABSENT from image. Negative claim supported."

        cot = (
            f"Step 1: Identified negative claim asserting absence of '{subj}'. "
            f"Step 2: Scanned visual features for presence of '{subj}' (detected presence = {detected_presence_score:.2f}). "
            f"Step 3: Calculated absence verification score = {absence_score:.2f}. "
            f"Verdict: {'ABSENCE_CONFIRMED' if absence_score >= 0.6 else 'PRESENCE_DETECTED_CONTRADICTED'}."
        )

        return Evidence(
            source="absence_verifier",
            result=result_str,
            score=round(absence_score, 3),
            cot_reasoning=cot,
            details={
                "target_concept": subj,
                "detected_presence_score": detected_presence_score,
                "absence_score": absence_score,
                "verifier_type": "Absence_Negative_Claim_Verifier"
            }
        )
