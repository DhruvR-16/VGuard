"""
Unit tests for specialized verifiers with directional evidence and real coordinate calculus.
"""

import pytest
from visionguard.schemas import AtomicClaim, ClaimType, Polarity, LinguisticStructure
from visionguard.verifiers import (
    OpenVocabularyGroundingVerifier,
    SpatialVerifier,
    ActionInteractionVerifier,
    OCRVerifier,
    AbsenceVerifier
)


def test_open_vocab_grounding_directional():
    verifier = OpenVocabularyGroundingVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C1",
        text="A purple backpack is present",
        type=ClaimType.OBJECT_EXISTENCE,
        linguistic_structure=LinguisticStructure(subject="backpack", predicate="exists", modifiers=["purple"])
    )
    features = {
        "objects": {"purple backpack": 0.92},
        "object_details": {"purple backpack": {"confidence": 0.92, "bbox": [0.2, 0.3, 0.6, 0.8]}}
    }
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.support_score >= 0.80
    assert evidence.bounding_box == [0.2, 0.3, 0.6, 0.8]


def test_spatial_coordinate_calculus():
    verifier = SpatialVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C2",
        text="The cup is on top of the desk",
        type=ClaimType.SPATIAL_RELATION,
        linguistic_structure=LinguisticStructure(subject="cup", predicate="on top of", target_object="desk")
    )
    # Cup center_y (0.25) is above Desk center_y (0.65) with horizontal overlap
    features = {
        "object_details": {
            "cup": {"confidence": 0.92, "bbox": [0.35, 0.15, 0.55, 0.35]},
            "desk": {"confidence": 0.95, "bbox": [0.10, 0.40, 0.90, 0.90]}
        }
    }
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.support_score >= 0.85
    assert evidence.contradiction_score <= 0.10


def test_ocr_fuzzy_matching():
    verifier = OCRVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C3",
        text="The sign says STOP",
        type=ClaimType.OCR_TEXT,
        linguistic_structure=LinguisticStructure(subject="sign", predicate="says STOP")
    )
    features = {
        "ocr_tokens": [{"text": "STOP", "confidence": 0.98, "bbox": [0.3, 0.2, 0.7, 0.5]}]
    }
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.support_score >= 0.90
    assert evidence.bounding_box == [0.3, 0.2, 0.7, 0.5]


def test_absence_directional_evidence():
    verifier = AbsenceVerifier(use_mock=True)
    claim = AtomicClaim(
        claim_id="C4",
        text="There are no cars",
        type=ClaimType.OBJECT_EXISTENCE,
        polarity=Polarity.NEGATIVE,
        linguistic_structure=LinguisticStructure(subject="cars", predicate="exists")
    )
    features = {"objects": {"dog": 0.92}}
    evidence = verifier.verify(claim, visual_features=features)
    assert evidence.support_score >= 0.70
    assert evidence.uncertainty_score <= 0.30
