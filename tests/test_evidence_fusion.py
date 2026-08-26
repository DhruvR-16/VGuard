"""
Unit tests for Evidence Fusion Engine and Uncertainty Engine.
"""

from visionguard.pipeline.evidence_fusion import EvidenceFusionEngine
from visionguard.pipeline.uncertainty_engine import UncertaintyEngine
from visionguard.schemas import AtomicClaim, ClaimType, Evidence, Verdict


def test_fusion_and_verdict():
    fusion = EvidenceFusionEngine(mode="weighted")
    uncertainty = UncertaintyEngine()

    claim = AtomicClaim(claim_id="C1", text="There is a dog.", type=ClaimType.OBJECT_EXISTENCE)
    evidences = [
        Evidence(source="open_vocab_grounding", score=0.92),
        Evidence(source="vlm_scene_context", score=0.88)
    ]

    fused_score = fusion.fuse(claim, evidences)
    assert 0.80 <= fused_score <= 1.0

    verdict, explanation = uncertainty.determine_verdict(claim, fused_score)
    assert verdict == Verdict.SUPPORTED


def test_contradicted_verdict():
    fusion = EvidenceFusionEngine(mode="weighted")
    uncertainty = UncertaintyEngine()

    claim = AtomicClaim(claim_id="C1", text="An alien is driving a car.", type=ClaimType.OBJECT_EXISTENCE)
    evidences = [
        Evidence(source="open_vocab_grounding", score=0.05),
        Evidence(source="vlm_scene_context", score=0.10)
    ]

    fused_score = fusion.fuse(claim, evidences)
    assert fused_score <= 0.35

    verdict, explanation = uncertainty.determine_verdict(claim, fused_score)
    assert verdict == Verdict.CONTRADICTED
