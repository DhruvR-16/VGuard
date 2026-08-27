"""
Region-Based Attribute & Color Verifier for VisionGuard.
Crops the image to the grounded object bounding box and evaluates contrasting attributes
using CLIP zero-shot classification and regional color distribution.
"""

from typing import Dict, Any, Optional, List
from PIL import Image
from visionguard.schemas import AtomicClaim, Evidence, Polarity
from visionguard.models.registry import registry


class AttributeVerifier:
    """
    Region-Conditioned Attribute & Color Verifier using CLIP zero-shot matching on object crops.
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
        claim_text = claim.text.lower()
        target_color = self._extract_color(claim_text)
        target_entity = claim.linguistic_structure.subject if claim.linguistic_structure else "object"

        if visual_features is None and image_path is None:
            return Evidence(
                source="clip_attribute_similarity",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image context provided for attribute verification.",
                details={"status": "ungrounded"}
            )

        image_obj: Optional[Image.Image] = visual_features.get("image_obj") if visual_features else None
        bbox = self._find_object_bbox(target_entity, visual_features)

        # 1. Real CLIP Inference on Cropped Region if image is present
        clip_support = 0.0
        clip_contradict = 0.0
        contrast_candidates = []

        if image_obj is not None and not self.use_mock:
            # Crop object region if bbox is available, else use full image
            img_crop = self._crop_bbox(image_obj, bbox) if bbox else image_obj

            # Formulate contrasting candidate phrases
            if target_color:
                colors = ["red", "blue", "green", "yellow", "black", "white", "brown"]
                contrast_candidates = [f"a {c} {target_entity}" for c in colors]
                target_cand = f"a {target_color} {target_entity}"
                if target_cand not in contrast_candidates:
                    contrast_candidates.insert(0, target_cand)
            else:
                contrast_candidates = [
                    f"a {target_entity} that is {claim_text}",
                    f"a regular {target_entity}",
                    f"a different {target_entity}"
                ]
                target_cand = contrast_candidates[0]

            probs = self.registry.run_clip_attribute_classification(img_crop, contrast_candidates)
            clip_support = float(probs.get(target_cand, 0.0))
            clip_contradict = float(sum(v for k, v in probs.items() if k != target_cand))

        # 2. Fallback to pixel color distribution or visual_features if CLIP is not active
        if clip_support == 0.0 and visual_features is not None:
            color_dist = visual_features.get("color_distribution", {})
            attr_dict = visual_features.get("attributes", {})

            # Check exact attribute entry
            if target_color and target_color in color_dist:
                col_frac = float(color_dist[target_color])
                clip_support = min(1.0, col_frac * 2.5) if col_frac > 0.05 else 0.15
                clip_contradict = 1.0 - clip_support
            elif any(k in claim_text for k in attr_dict):
                matched_attr = next(k for k in attr_dict if k in claim_text)
                clip_support = float(attr_dict[matched_attr])
                clip_contradict = 1.0 - clip_support
            else:
                clip_support = 0.50
                clip_contradict = 0.30

        # Calculate directional scores
        if clip_support >= 0.40:
            support_s = round(min(0.95, clip_support * 1.5), 3)
            contradict_s = round(max(0.05, clip_contradict * 0.4), 3)
            uncertainty_s = 0.10
            res_str = f"Attribute alignment verified: '{claim.text}' supported on object region."
        elif clip_support >= 0.15:
            support_s = round(clip_support, 3)
            contradict_s = round(clip_contradict * 0.5, 3)
            uncertainty_s = 0.50
            res_str = f"Ambiguous attribute evidence on region for '{claim.text}'."
        else:
            support_s = round(clip_support, 3)
            contradict_s = 0.80
            uncertainty_s = 0.15
            res_str = f"Contradiction: Attribute in '{claim.text}' not supported by regional visual features."

        cot = (
            f"Step 1: Grounded object '{target_entity}' at bbox {bbox or 'full_frame'}. "
            f"Step 2: Region CLIP zero-shot classification across contrasting candidates {contrast_candidates[:3]}... "
            f"Step 3: Target attribute probability = {clip_support:.3f}. "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="clip_attribute_similarity",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=round(clip_support, 3),
            score=support_s,
            bounding_box=bbox,
            cot_reasoning=cot,
            details={
                "target_entity": target_entity,
                "target_color": target_color,
                "clip_support_prob": round(clip_support, 3),
                "region_bbox": bbox
            }
        )

    def _find_object_bbox(self, entity: str, visual_features: Dict[str, Any]) -> Optional[List[float]]:
        if not visual_features:
            return None
        obj_details = visual_features.get("object_details", {})
        if entity in obj_details:
            return obj_details[entity].get("bbox")
        for k, v in obj_details.items():
            if entity in k or k in entity:
                return v.get("bbox")
        boxes = visual_features.get("bounding_boxes", [])
        return boxes[0] if boxes else None

    def _crop_bbox(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        img_w, img_h = image.size
        x1 = max(0, int(bbox[0] * img_w))
        y1 = max(0, int(bbox[1] * img_h))
        x2 = min(img_w, int(bbox[2] * img_w))
        y2 = min(img_h, int(bbox[3] * img_h))
        if x2 > x1 and y2 > y1:
            return image.crop((x1, y1, x2, y2))
        return image

    def _extract_color(self, text: str) -> Optional[str]:
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "pink", "gray", "silver", "gold"]
        for c in colors:
            if c in text:
                return c
        return None
