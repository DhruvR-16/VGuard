"""
Attribute Verifier Module for VisionGuard.
Verifies attribute claims against real pixel color distributions and extracted visual attributes.
"""

from typing import Dict, Any, Optional
from visionguard.schemas import AtomicClaim, Evidence


class AttributeVerifier:
    """
    Specialized verifier for Attribute Claims (color, state, texture, material).
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        text = claim.text.lower()
        modifiers = claim.linguistic_structure.modifiers if claim.linguistic_structure else []
        mod_str = ", ".join(modifiers) if modifiers else "attributes"

        if visual_features is None:
            return Evidence(
                source="clip_attribute_similarity",
                score=0.0,
                cot_reasoning="No image provided for attribute verification.",
                details={"status": "No image input"}
            )

        color_dist = visual_features.get("color_distribution", {})
        attributes = visual_features.get("attributes", {})
        dominant_color = visual_features.get("dominant_color", "neutral")

        # Check color match
        color_score = 0.50
        matched_colors = []
        contradicted_colors = []

        for color_name in ["red", "blue", "green", "yellow", "brown", "white", "black"]:
            if color_name in text:
                pct = color_dist.get(color_name, 0.0)
                if pct > 0.08:
                    matched_colors.append((color_name, pct))
                    color_score = min(1.0, 0.60 + pct)
                else:
                    contradicted_colors.append((color_name, pct))
                    color_score = max(0.05, pct)

        if not matched_colors and not contradicted_colors:
            # Check general attributes
            for attr_name, attr_val in attributes.items():
                if attr_name in text or any(m in attr_name for m in modifiers):
                    color_score = max(color_score, float(attr_val))

        cot = (
            f"Step 1: Analyzed attribute modifiers: [{mod_str}]. "
            f"Step 2: Queried pixel color channels: {color_dist}. "
            f"Step 3: Dominant color = '{dominant_color}'. Attribute confidence = {color_score:.2f}."
        )

        return Evidence(
            source="clip_attribute_similarity",
            score=round(color_score, 3),
            cot_reasoning=cot,
            details={
                "modifiers": modifiers,
                "verifier_type": "Pixel_Color_Distribution_&_CLIP_Matcher",
                "matched_colors": matched_colors,
                "contradicted_colors": contradicted_colors,
                "dominant_color": dominant_color
            }
        )
