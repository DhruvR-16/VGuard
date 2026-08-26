"""
OCR Text Verifier for VisionGuard.
Verifies embedded text, signs, labels, and document tokens within image regions.
"""

import re
from typing import Dict, Any, Optional, List
from visionguard.schemas import AtomicClaim, Evidence


class OCRVerifier:
    """
    Verifies assertions regarding embedded text within images.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(self, claim: AtomicClaim, image_path: Optional[str] = None, visual_features: Optional[Dict[str, Any]] = None) -> Evidence:
        text = claim.text.lower()

        if visual_features is None:
            return Evidence(
                source="ocr_text_engine",
                result="No visual context provided",
                score=0.0,
                cot_reasoning="No image input provided for OCR text verification.",
                details={"status": "ungrounded"}
            )

        ocr_tokens = visual_features.get("ocr_tokens", [])
        extracted_text_in_claim = self._extract_claimed_text(claim.text)

        score = 0.05
        matched_token = None

        if extracted_text_in_claim:
            for token in ocr_tokens:
                if token.lower() in extracted_text_in_claim.lower() or extracted_text_in_claim.lower() in token.lower():
                    score = 0.95
                    matched_token = token
                    break
        elif ocr_tokens:
            for token in ocr_tokens:
                if token.lower() in text:
                    score = 0.90
                    matched_token = token
                    break

        cot = (
            f"Step 1: Extracted claimed text string: '{extracted_text_in_claim or text}'. "
            f"Step 2: Scanned OCR tokens in image: {ocr_tokens}. "
            f"Step 3: Matched token = '{matched_token or 'None'}'. Confidence = {score:.2f}."
        )

        return Evidence(
            source="ocr_text_engine",
            result=f"{'Text verified: ' + matched_token if matched_token else 'Text not found in image'}",
            score=round(score, 3),
            cot_reasoning=cot,
            details={
                "claimed_text": extracted_text_in_claim,
                "matched_token": matched_token,
                "verifier_type": "Optical_Character_Recognition_Engine"
            }
        )

    def _extract_claimed_text(self, text: str) -> Optional[str]:
        # Look for quoted text e.g. 'STOP' or "STOP"
        quote_match = re.search(r'[\'"]([^\'"]+)[\'"]', text)
        if quote_match:
            return quote_match.group(1).strip()
        # Look for 'says X' or 'reads X'
        says_match = re.search(r'\b(?:says|reads|written|labeled)\s+([A-Za-z0-9_\-]+)', text, re.IGNORECASE)
        if says_match:
            return says_match.group(1).strip()
        return None
