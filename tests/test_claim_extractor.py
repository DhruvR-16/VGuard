"""
Unit tests for open-domain structured claim extraction, negation polarity, and dependency graphs.
"""

import pytest
from visionguard.pipeline.claim_extractor import ClaimExtractor
from visionguard.schemas import ClaimType, Polarity, Verifiability


def test_open_domain_claim_extractor():
    extractor = ClaimExtractor()
    text = "A violinist is playing a cello beside a microphone."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 1
    assert claims[0].linguistic_structure.subject == "violinist"
    assert claims[0].linguistic_structure.predicate == "playing"


def test_negation_polarity_extraction():
    extractor = ClaimExtractor()
    text = "There are no cars on the road."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 1
    assert claims[0].polarity == Polarity.NEGATIVE


def test_external_knowledge_extraction():
    extractor = ClaimExtractor()
    text = "This photograph was taken in Paris during 1920."
    claims = extractor.extract_claims(text)
    assert len(claims) >= 1
    assert claims[0].verifiability == Verifiability.EXTERNAL_KNOWLEDGE
    assert claims[0].type == ClaimType.EXTERNAL_KNOWLEDGE
