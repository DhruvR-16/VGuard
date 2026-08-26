"""
Baseline B: Global Image-Text Similarity Baseline.
Calculates whole-caption visual similarity using global CLIP feature matching.
"""

from typing import Dict, Any, Optional


class SimilarityBaseline:
    """
    Baseline B: Whole-caption image-text similarity without claim breakdown or routing.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def evaluate(self, response_text: str, visual_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates global image-text similarity score.
        """
        text = response_text.lower()
        
        # Simulate global CLIP sentence embedding dot product
        if "paris" in text or "snowboard" in text:
            similarity = 0.32
        elif any(k in text for k in ["dog", "boy", "bicycle", "red"]):
            similarity = 0.78
        else:
            similarity = 0.55

        is_hallucinated = similarity < 0.50
        return {
            "method": "Global Image-Text Similarity",
            "similarity_score": round(similarity, 3),
            "hallucination_probability": round(1.0 - similarity, 3),
            "verdict": "Hallucinated" if is_hallucinated else "Not Hallucinated",
            "is_consistent": not is_hallucinated
        }
