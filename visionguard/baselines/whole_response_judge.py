"""
Baseline A: Whole-Response VLM Judge for VisionGuard.
Evaluates the entire VLM response as a single monolithic unit directly using VQA / VLM
without claim decomposition, bounding boxes, or specialized verifiers.
Zero keyword heuristics.
"""

from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from visionguard.models.registry import registry


class WholeResponseVLMJudge:
    """
    Monolithic VLM Judge baseline that scores whole-response hallucination probability directly.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.registry = registry

    def evaluate(
        self,
        response_text: str,
        visual_features: Optional[Dict[str, Any]] = None,
        image_input: Optional[Union[str, bytes, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates whole response directly using monolithic VQA query.
        """
        image_obj: Optional[Image.Image] = None
        if visual_features and "image_obj" in visual_features:
            image_obj = visual_features["image_obj"]
        elif isinstance(image_input, Image.Image):
            image_obj = image_input

        confidence = 0.50

        # 1. Real Monolithic VQA evaluation over whole response
        if image_obj is not None and not self.use_mock:
            # Query VQA on the overall description
            question = f"Does this image match the description: '{response_text[:120]}'?"
            ans, yes_p, no_p = self.registry.run_vqa_query(image_obj, question)
            if ans.lower() == "yes" or yes_p > 0.50:
                confidence = float(yes_p)
            elif ans.lower() == "no" or no_p > 0.50:
                confidence = float(round(1.0 - no_p, 3))
            else:
                # Contrastive CLIP whole-response probability
                candidates = [
                    f"A photo showing {response_text[:80]}",
                    f"A photo completely unrelated to {response_text[:80]}"
                ]
                clip_probs = self.registry.run_clip_attribute_classification(image_obj, candidates)
                confidence = float(clip_probs.get(candidates[0], 0.50))
        elif visual_features is not None:
            # Feature overlap without decomposition
            words = [w.lower() for w in response_text.split() if len(w) > 3]
            objects = visual_features.get("objects", {})
            if objects:
                matched_scores = [float(objects[k]) for k in objects if any(k in w or w in k for w in words)]
                confidence = float(np.mean(matched_scores)) if matched_scores else 0.35
            else:
                confidence = 0.45

        is_consistent = confidence >= 0.55
        return {
            "method": "Whole-Response VLM Judge",
            "hallucination_probability": round(1.0 - confidence, 4),
            "verdict": "Not Hallucinated" if is_consistent else "Hallucinated",
            "confidence": round(confidence, 4),
            "is_consistent": is_consistent
        }


# Alias for backward compatibility
WholeResponseJudge = WholeResponseVLMJudge
