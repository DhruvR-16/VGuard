"""
Tests for CountVerifier in VisionGuard.
Tests integer parsing, qualitative word mapping, NMS bbox clustering, and directional evidence.
"""

import pytest
from visionguard.verifiers.count_verifier import CountVerifier
from visionguard.schemas import AtomicClaim, ClaimType, Polarity, Certainty, Verifiability


@pytest.fixture
def count_verifier():
    return CountVerifier()


def test_count_verifier_exact_integer_support(count_verifier):
    claim = AtomicClaim(
        claim_id="c1",
        text="There are three cats on the chair",
        type=ClaimType.OBJECT_COUNT,
        polarity=Polarity.POSITIVE,
        certainty=Certainty.ASSERTED,
        verifiability=Verifiability.VISUAL,
        verifiable=True
    )
    # Visual features with 3 distinct bounding boxes
    visual_features = {
        "objects": {"cat": 0.92},
        "bounding_boxes": [
            [0.1, 0.1, 0.2, 0.2],
            [0.4, 0.4, 0.5, 0.5],
            [0.7, 0.7, 0.8, 0.8]
        ]
    }
    evidence = count_verifier.verify(claim, visual_features=visual_features)
    assert evidence.support_score > 0.70
    assert evidence.contradiction_score < 0.30
    assert "Detected 3" in evidence.details.get("target_entity", "") or "3" in str(evidence.details.get("detected_count", ""))


def test_count_verifier_exact_integer_contradiction(count_verifier):
    claim = AtomicClaim(
        claim_id="c2",
        text="There are five dogs in the park",
        type=ClaimType.OBJECT_COUNT,
        polarity=Polarity.POSITIVE,
        certainty=Certainty.ASSERTED,
        verifiability=Verifiability.VISUAL,
        verifiable=True
    )
    # Only 1 dog box
    visual_features = {
        "objects": {"dog": 0.95},
        "bounding_boxes": [
            [0.1, 0.1, 0.3, 0.3]
        ]
    }
    evidence = count_verifier.verify(claim, visual_features=visual_features)
    assert evidence.contradiction_score > 0.70
    assert evidence.support_score < 0.30


def test_count_verifier_qualitative_several(count_verifier):
    claim = AtomicClaim(
        claim_id="c3",
        text="There are several strawberries in the bowl",
        type=ClaimType.OBJECT_COUNT,
        polarity=Polarity.POSITIVE,
        certainty=Certainty.ASSERTED,
        verifiability=Verifiability.VISUAL,
        verifiable=True
    )
    # 4 strawberries
    visual_features = {
        "objects": {"strawberry": 0.90},
        "bounding_boxes": [
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
            [0.4, 0.4, 0.5, 0.5]
        ]
    }
    evidence = count_verifier.verify(claim, visual_features=visual_features)
    assert evidence.support_score >= 0.75
    assert evidence.contradiction_score <= 0.25


def test_count_verifier_nms_duplicate_suppression(count_verifier):
    claim = AtomicClaim(
        claim_id="c4",
        text="There is a single cat on the chair",
        type=ClaimType.OBJECT_COUNT,
        polarity=Polarity.POSITIVE,
        certainty=Certainty.ASSERTED,
        verifiability=Verifiability.VISUAL,
        verifiable=True
    )
    # Two heavily overlapping detections of the same cat (IoU > 0.50)
    visual_features = {
        "objects": {"cat": 0.94},
        "bounding_boxes": [
            [0.20, 0.20, 0.60, 0.60],
            [0.21, 0.22, 0.59, 0.61]
        ]
    }
    evidence = count_verifier.verify(claim, visual_features=visual_features)
    # NMS should merge them into 1 instance, matching "single cat"
    assert evidence.support_score >= 0.75
    assert evidence.details.get("detected_count") == 1
