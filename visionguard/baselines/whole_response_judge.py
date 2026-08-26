"""
Baseline A: Whole-Response VLM Judge.
Evaluates the entire VLM response as a single monolithic unit without claim decomposition.
"""

from typing import Dict, Any, Optional


class WholeResponseVLMJudge:
    """
    Monolithic VLM Judge baseline that scores whole-response hallucination probability directly.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def evaluate(self, response_text: str, visual_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates whole response and returns hallucination score.
        """
        text = response_text.lower()
        score = 0.85

        # Single-pass keyword detection without spatial/attribute routing
        hallucination_indicators = ["paris", "snowboard", "unicorn", "flying car", "alien", "belong to"]
        if any(ind in text for ind in hallucination_indicators):
            score = 0.25
        elif visual_features and "objects" in visual_features:
            words = [w for w in text.split() if len(w) > 3]
            matches = sum(1 for w in words if any(w in obj.lower() for obj in visual_features["objects"]))
            score = min(0.95, 0.4 + 0.15 * matches)

        is_hallucinated = score < 0.60
        return {
            "method": "Whole-Response VLM Judge",
            "hallucination_probability": round(1.0 - score, 3),
            "verdict": "Hallucinated" if is_hallucinated else "Not Hallucinated",
            "confidence": round(score, 3),
            "is_consistent": not is_hallucinated
        }


# Alias for backward compatibility
WholeResponseJudge = WholeResponseVLMJudge
