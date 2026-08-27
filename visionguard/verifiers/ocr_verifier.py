"""
OCR & Embedded Text Verifier for VisionGuard.
Matches extracted textual tokens, signage, and numbers against real Tesseract OCR extractions
using fuzzy string distance, case normalization, and token alignment.
"""

import re
from typing import Dict, Any, Optional, List
from visionguard.schemas import AtomicClaim, Evidence


class OCRVerifier:
    """
    Optical Character Recognition Verifier for embedded text, signage, and numbers.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        claim_text = claim.text.strip()

        if visual_features is None:
            return Evidence(
                source="ocr_text_engine",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No visual features provided for OCR text verification.",
                details={"status": "ungrounded"}
            )

        ocr_tokens = visual_features.get("ocr_tokens", [])
        expected_text = self._extract_expected_text(claim)

        # 1. Real Fuzzy Token Matching over Tesseract Tokens
        best_match_token = None
        best_match_score = 0.0
        matched_bbox = None

        clean_expected = self._normalize(expected_text)

        for tok in ocr_tokens:
            tok_text = tok.get("text", "") if isinstance(tok, dict) else str(tok)
            clean_tok = self._normalize(tok_text)
            tok_conf = float(tok.get("confidence", 0.90)) if isinstance(tok, dict) else 0.90

            sim = self._calculate_similarity(clean_expected, clean_tok)
            weighted_sim = sim * tok_conf

            if weighted_sim > best_match_score:
                best_match_score = weighted_sim
                best_match_token = tok_text
                matched_bbox = tok.get("bbox") if isinstance(tok, dict) else None

        # 2. Directional Evidence Computation
        if best_match_score >= 0.70:
            support_s = round(best_match_score, 3)
            contradict_s = 0.05
            uncertainty_s = 0.10
            res_str = f"OCR matched text '{best_match_token}' with expected '{expected_text}' (score {best_match_score:.1%})."
        elif best_match_score >= 0.35:
            support_s = round(best_match_score, 3)
            contradict_s = 0.25
            uncertainty_s = 0.45
            res_str = f"Partial OCR token match '{best_match_token}' for '{expected_text}' (score {best_match_score:.1%})."
        else:
            support_s = 0.05
            contradict_s = 0.75
            uncertainty_s = 0.20
            res_str = f"OCR text '{expected_text}' not detected in image."

        cot = (
            f"Step 1: Extracted expected text query '{expected_text}'. "
            f"Step 2: Scanned {len(ocr_tokens)} OCR tokens from Tesseract. "
            f"Step 3: Best token match: '{best_match_token}' with similarity {best_match_score:.2f} at bbox {matched_bbox or 'None'}. "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="ocr_text_engine",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=round(best_match_score, 3),
            score=support_s,
            bounding_box=matched_bbox,
            cot_reasoning=cot,
            details={
                "expected_text": expected_text,
                "matched_token": best_match_token,
                "match_similarity": round(best_match_score, 3),
                "matched_bbox": matched_bbox
            }
        )

    def _extract_expected_text(self, claim: AtomicClaim) -> str:
        """
        Extracts embedded quotes, uppercase signs, or keywords from claim text.
        """
        # Check for quoted text: "STOP" or 'SLOW'
        quotes = re.findall(r'["\']([^"\']+)["\']', claim.text)
        if quotes:
            return quotes[0]

        # Check for capitalized words: STOP, EXIT, SPEED
        caps = re.findall(r'\b[A-Z0-9]{2,}\b', claim.text)
        if caps:
            return " ".join(caps)

        # Fallback to predicate target
        if claim.linguistic_structure and claim.linguistic_structure.predicate:
            pred = claim.linguistic_structure.predicate
            return pred.replace("says", "").replace("reads", "").strip()

        return claim.text

    def _normalize(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        if s1 in s2 or s2 in s1:
            return min(len(s1), len(s2)) / float(max(len(s1), len(s2)))

        # Simple character overlap ratio
        overlap = sum(1 for ch in s1 if ch in s2)
        return float(overlap) / max(len(s1), len(s2))
