"""
Unit tests for specialized verifiers: Open-Vocabulary Grounding, Spatial, Action, OCR, Absence, Scene.
"""

import pytest
from visionguard.schemas import AtomicClaim, ClaimType, Polarity, LinguisticStructure
from visionguard.verifiers import (
    OpenVocabularyGroundingVerifier,
    SpatialVerifier,
    ActionInteractionVerifier,
    OCRVerifier,
    AbsenceVerifier,
    SceneVerifier
)


def test_open_vocab_grounding_verifier():
    verifier = OpenVocabularyGroundingVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C1",
        text="A vintage bicycle exists",
        type=ClaimType.OBJECT_EXISTENCE,
        linguistic_structure=LinguisticStructure(subject="bicycle", predicate="exists")
    )
    features = {"objects": {"bicycle": 0.95}, "bounding_boxes": [[0.1, 0.1, 0.8, 0.8]]}
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.score >= 0.80
    assert evidence.bounding_box is not None


def test_spatial_verifier():
    verifier = SpatialVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C2",
        text="The cup is on top of the table",
        type=ClaimType.SPATIAL_RELATION,
        linguistic_structure=LinguisticStructure(subject="cup", predicate="on top of", target_object="table")
    )
    features = {"objects": {"cup": 0.90, "table": 0.92}, "bounding_boxes": [[0.1, 0.1, 0.4, 0.4]]}
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.score >= 0.70


def test_absence_verifier():
    verifier = AbsenceVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C3",
        text="There are no cars",
        type=ClaimType.NEGATIVE_CLAIM,
        polarity=Polarity.NEGATIVE,
        linguistic_structure=LinguisticStructure(subject="cars", predicate="exists")
    )
    features = {"objects": {"dog": 0.90}} # Car is absent!
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.score >= 0.85


def test_ocr_verifier():
    verifier = OCRVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C4",
        text="The sign says STOP",
        type=ClaimType.OCR_TEXT,
        linguistic_structure=LinguisticStructure(subject="sign", predicate="says STOP")
    )
    features = {"ocr_tokens": ["STOP", "SPEED_LIMIT_25"]}
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.score >= 0.90
