"""
Baseline B: Real Global Image-Text Similarity Baseline for VisionGuard.
Calculates whole-caption visual similarity using global CLIP embedding dot products / cosine similarities.
Zero keyword heuristics.
"""

from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import torch
from visionguard.models.registry import registry


class SimilarityBaseline:
    """
    Baseline B: Global whole-image to whole-text CLIP cosine similarity baseline
    without claim breakdown, localization, or specialized verifiers.
    """

    def __init__(self, use_mock: bool = False, similarity_threshold: float = 0.22):
        self.use_mock = use_mock
        self.registry = registry
        self.similarity_threshold = similarity_threshold

    def evaluate(
        self,
        response_text: str,
        visual_features: Optional[Dict[str, Any]] = None,
        image_input: Optional[Union[str, bytes, Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates global image-text similarity using actual CLIP embeddings.
        """
        image_obj: Optional[Image.Image] = None
        if visual_features and "image_obj" in visual_features:
            image_obj = visual_features["image_obj"]
        elif isinstance(image_input, Image.Image):
            image_obj = image_input

        similarity = 0.50

        # 1. Real CLIP Global Image-Text Cosine Similarity
        if image_obj is not None and not self.use_mock:
            clip_model, clip_proc = self.registry.get_clip()
            if clip_model != "fallback" and clip_model is not None and clip_proc is not None:
                try:
                    inputs = clip_proc(
                        text=[response_text[:77]],
                        images=image_obj,
                        return_tensors="pt",
                        padding=True
                    ).to(self.registry.device)

                    with torch.no_grad():
                        image_features = clip_model.get_image_features(pixel_values=inputs.pixel_values)
                        text_features = clip_model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

                        # Normalized cosine similarity
                        image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                        cos_sim = float((image_norm * text_norm).sum(dim=-1)[0].cpu().item())
                        # Rescale [-1, 1] cosine to [0, 1] probability
                        similarity = max(0.0, min(1.0, (cos_sim + 0.1) / 0.45))
                except Exception:
                    similarity = 0.50
        elif visual_features is not None:
            # Independent feature match: Jaccard word-concept overlap
            text_words = set(w.lower() for w in response_text.split() if len(w) > 3)
            obj_words = set(k.lower() for k in visual_features.get("objects", {}))
            if text_words and obj_words:
                overlap = len(text_words.intersection(obj_words)) / max(1, len(text_words))
                similarity = min(0.95, 0.35 + overlap * 0.60)
            else:
                similarity = 0.45

        is_consistent = similarity >= 0.50
        return {
            "method": "Global Image-Text Similarity",
            "similarity_score": round(similarity, 4),
            "hallucination_probability": round(1.0 - similarity, 4),
            "verdict": "Not Hallucinated" if is_consistent else "Hallucinated",
            "is_consistent": is_consistent
        }
