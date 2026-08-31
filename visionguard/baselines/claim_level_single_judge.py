"""
Baseline: Independent Claim-Level Single Model Judge for VisionGuard.
Directly evaluates an atomic claim against the raw visual context using a single
vision-language model (e.g. CLIP / VQA) without bounding box localization,
spatial geometry calculus, OCR extraction, or multi-evidence fusion.
"""

from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np
from visionguard.schemas import AtomicClaim, Verdict
from visionguard.models.registry import registry


class ClaimLevelSingleJudge:
    """
    Independent Single-Model Claim Verifier baseline.
    Evaluates individual atomic claims directly against the image without CAM-V's
    grounding, geometry, OCR, or multi-evidence directional fusion engine.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.registry = registry

    def _load_image(self, image_input: Optional[Union[str, bytes, Image.Image]]) -> Optional[Image.Image]:
        if image_input is None:
            return None
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        try:
            if isinstance(image_input, bytes):
                import io
                return Image.open(io.BytesIO(image_input)).convert("RGB")
            if isinstance(image_input, str) and os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
        except Exception:
            return None
        return None

    def evaluate_claim(
        self,
        claim: Union[AtomicClaim, str],
        image_input: Optional[Union[str, bytes, Image.Image]] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        claim_text = claim.text if isinstance(claim, AtomicClaim) else str(claim)

        image_obj: Optional[Image.Image] = None
        if visual_features and "image_obj" in visual_features:
            image_obj = visual_features["image_obj"]
        elif image_input is not None:
            image_obj = self._load_image(image_input)

        # 1. Real Single-Model VQA/CLIP Assessment if image is available
        confidence = 0.50
        if image_obj is not None and not self.use_mock:
            # Query standard VQA with direct yes/no on the claim
            q_text = f"Is the following statement true: '{claim_text}'?"
            ans, yes_p, no_p = self.registry.run_vqa_query(image_obj, q_text)
            if ans.lower() == "yes" or yes_p > 0.55:
                confidence = yes_p
            elif ans.lower() == "no" or no_p > 0.55:
                confidence = round(1.0 - no_p, 3)
            else:
                # Fallback to zero-shot CLIP probability over affirmation vs negation
                candidates = [
                    f"A photo showing {claim_text}",
                    f"A photo without {claim_text}"
                ]
                clip_probs = self.registry.run_clip_attribute_classification(image_obj, candidates)
                confidence = float(clip_probs.get(candidates[0], 0.50))
        elif visual_features is not None:
            # Fallback for pre-extracted feature testing: direct keyword overlap heuristic
            words = [w.lower() for w in claim_text.split() if len(w) > 3]
            objects = visual_features.get("objects", {})
            matched = [objects[k] for k in objects if any(k in w or w in k for w in words)]
            if matched:
                confidence = float(np.mean(matched))
            else:
                confidence = 0.40

        is_supported = confidence >= 0.60
        verdict = Verdict.SUPPORTED if is_supported else (Verdict.CONTRADICTED if confidence <= 0.35 else Verdict.INSUFFICIENT_EVIDENCE)

        return {
            "method": "Claim-Level Single Judge",
            "claim": claim_text,
            "confidence": round(confidence, 4),
            "verdict": verdict.value,
            "is_consistent": is_supported
        }

    def evaluate_response_claims(
        self,
        claims: List[Union[AtomicClaim, str]],
        image_input: Optional[Union[str, bytes, Image.Image]] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a set of atomic claims independently and computes mean score.
        """
        if not claims:
            return {"mean_confidence": 0.50, "claim_evaluations": [], "is_consistent": True}

        evals = [self.evaluate_claim(c, image_input, visual_features) for c in claims]
        mean_conf = float(np.mean([e["confidence"] for e in evals]))
        all_supported = all(e["is_consistent"] for e in evals)

        return {
            "method": "Claim-Level Single Judge",
            "mean_confidence": round(mean_conf, 4),
            "claim_evaluations": evals,
            "is_consistent": all_supported
        }
