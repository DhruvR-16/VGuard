"""
Unit tests for structured claim extraction, negation polarity tracking, and dependency graphs.
"""

import pytest
from visionguard.pipeline.claim_extractor import ClaimExtractor
from visionguard.schemas import ClaimType, Polarity


def test_claim_extractor_basic():
    extractor = ClaimExtractor()
    text = "A young boy wearing a red shirt is riding a bicycle near a park."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 2
    assert any(c.linguistic_structure.subject in ["boy", "person"] for c in claims)


def test_claim_extractor_negation():
    extractor = ClaimExtractor()
    text = "There are no cars on the road."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 1
    assert claims[0].polarity == Polarity.NEGATIVE
    assert claims[0].type == ClaimType.NEGATIVE_CLAIM


def test_claim_extractor_ocr_and_out_of_scope():
    extractor = ClaimExtractor()
    text = "The sign says STOP and the photo was taken in Paris."
    claims = extractor.extract_claims(text)
    types = [c.type for c in claims]
    assert ClaimType.OCR_TEXT in types or ClaimType.OUT_OF_SCOPE in types
