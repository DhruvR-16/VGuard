"""
Real Open-Vocabulary Grounding Verifier for VisionGuard.
Queries YOLO-World / Open-Vocabulary Grounding backend with arbitrary text phrases
and produces real bounding box coordinates, detection confidences, and directional evidence.
"""

from typing import Dict, Any, Optional, List
from visionguard.schemas import AtomicClaim, Evidence, Polarity
from visionguard.models.registry import registry


class OpenVocabularyGroundingVerifier:
    """
    Open-Vocabulary Grounding Verifier powered by YOLO-World.
    Detects arbitrary open-domain entities and outputs normalized bounding boxes.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.registry = registry

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        # Extract target concept to ground
        concept = self._extract_target_concept(claim)

        if visual_features is None and image_path is None:
            return Evidence(
                source="open_vocab_grounding",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image input provided for visual grounding.",
                details={"status": "ungrounded", "concept": concept}
            )

        # 1. Attempt Real Model Inference if image is available
        image_obj = visual_features.get("image_obj") if visual_features else None
        target_image = image_obj or image_path

        best_bbox = None
        best_conf = 0.0
        match_source = "detector"

        if target_image is not None and not self.use_mock:
            detections = self.registry.run_open_vocab_grounding(target_image, [concept], conf_threshold=0.10)
            if detections:
                top_det = max(detections, key=lambda d: d["confidence"])
                best_conf = top_det["confidence"]
                best_bbox = top_det["bbox"]
                match_source = "yolo_world_inference"

        # 2. Check pre-extracted visual_features as fallback or dataset inputs
        if best_conf == 0.0 and visual_features is not None:
            obj_details = visual_features.get("object_details", {})
            objects = visual_features.get("objects", {})

            # Check exact concept
            if concept in obj_details:
                best_conf = obj_details[concept].get("confidence", 0.90)
                best_bbox = obj_details[concept].get("bbox")
                match_source = "feature_exact"
            elif concept in objects:
                best_conf = float(objects[concept])
                best_bbox = visual_features.get("bounding_boxes", [None])[0] if visual_features.get("bounding_boxes") else None
                match_source = "feature_exact"
            else:
                # Substring / partial match
                for obj_name, conf_val in objects.items():
                    if obj_name in concept or concept in obj_name:
                        best_conf = float(conf_val) if isinstance(conf_val, (int, float)) else 0.85
                        match_source = f"feature_partial_{obj_name}"
                        break

        # 3. Directional Evidence Calculation with Polarity Awareness
        is_negative_claim = (claim.polarity == Polarity.NEGATIVE)

        if is_negative_claim:
            # For "There are no cars":
            # If car IS detected with high confidence -> Contradicted
            # If car is NOT detected -> Supported
            if best_conf >= 0.50:
                support_s = 0.05
                contradict_s = round(best_conf, 3)
                uncertainty_s = 0.10
                res_str = f"Contradicted: Concept '{concept}' was detected (conf={best_conf:.2f}), conflicting with negative assertion."
            else:
                support_s = round(1.0 - best_conf, 3)
                contradict_s = 0.10
                uncertainty_s = 0.20
                res_str = f"Supported: Concept '{concept}' is absent from visual grounding (conf={best_conf:.2f})."
        else:
            # Positive claim: "A car is present"
            if best_conf >= 0.60:
                support_s = round(best_conf, 3)
                contradict_s = round(max(0.0, 1.0 - best_conf), 3)
                uncertainty_s = 0.10
                res_str = f"Grounding verified '{concept}' with confidence {best_conf:.1%}."
            elif best_conf >= 0.25:
                support_s = round(best_conf, 3)
                contradict_s = 0.20
                uncertainty_s = 0.50
                res_str = f"Weak visual grounding for '{concept}' (confidence {best_conf:.1%})."
            else:
                # Non-detection: High uncertainty, not necessarily proof of contradiction
                support_s = 0.10
                contradict_s = 0.25
                uncertainty_s = 0.70
                res_str = f"Concept '{concept}' not localized by open-vocabulary detector (uncertainty=0.70)."

        cot = (
            f"Step 1: Open-vocabulary target concept '{concept}'. "
            f"Step 2: Grounding backend '{match_source}' returned confidence {best_conf:.2f}. "
            f"Step 3: Bounding box region: {best_bbox or 'None'}. "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="open_vocab_grounding",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=best_conf,
            score=support_s,
            bounding_box=best_bbox,
            cot_reasoning=cot,
            details={
                "target_concept": concept,
                "detection_confidence": best_conf,
                "bounding_box": best_bbox,
                "match_source": match_source,
                "is_negative_claim": is_negative_claim
            }
        )

    def _extract_target_concept(self, claim: AtomicClaim) -> str:
        """
        Extracts the core entity or phrase to ground.
        """
        if claim.linguistic_structure:
            subj = claim.linguistic_structure.subject.strip().lower()
            mods = " ".join(claim.linguistic_structure.modifiers).strip().lower()
            if mods and mods not in subj:
                return f"{mods} {subj}"
            if subj:
                return subj

        text = claim.text.lower()
        stopwords = ["there", "is", "a", "an", "the", "are", "some", "present", "visible", "exists", "on", "in", "near"]
        words = [w for w in text.split() if w not in stopwords and len(w) > 2]
        return " ".join(words) if words else text


# Alias
ObjectVerifier = OpenVocabularyGroundingVerifier
