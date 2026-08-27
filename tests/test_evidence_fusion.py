"""
Unit tests for Directional Evidence Fusion and Uncertainty Engine.
"""

from visionguard.pipeline.evidence_fusion import EvidenceFusionEngine
from visionguard.pipeline.uncertainty_engine import UncertaintyEngine
from visionguard.schemas import AtomicClaim, ClaimType, Evidence, Verdict


def test_directional_fusion_and_supported():
    fusion = EvidenceFusionEngine(mode="claim_aware")
    uncertainty = UncertaintyEngine()

    claim = AtomicClaim(claim_id="C1", text="There is a dog.", type=ClaimType.OBJECT_EXISTENCE)
    evidences = [
        Evidence(source="open_vocab_grounding", result="Dog detected", support_score=0.92, contradiction_score=0.05, uncertainty_score=0.10, confidence=0.92),
        Evidence(source="vlm_scene_context", result="Scene matches", support_score=0.88, contradiction_score=0.05, uncertainty_score=0.10, confidence=0.88)
    ]

    supp, cont, unc, math_obj = fusion.fuse_directional(claim, evidences)
    assert supp >= 0.85
    assert cont <= 0.10

    verdict, explanation = uncertainty.determine_verdict(claim, supp, cont, unc)
    assert verdict == Verdict.SUPPORTED


def test_directional_contradiction():
    fusion = EvidenceFusionEngine(mode="claim_aware")
    uncertainty = UncertaintyEngine()

    claim = AtomicClaim(claim_id="C2", text="An alien is driving a car.", type=ClaimType.OBJECT_EXISTENCE)
    evidences = [
        Evidence(source="open_vocab_grounding", result="Alien not localized", support_score=0.05, contradiction_score=0.75, uncertainty_score=0.20, confidence=0.05),
        Evidence(source="vlm_scene_context", result="Scene mismatch", support_score=0.10, contradiction_score=0.80, uncertainty_score=0.10, confidence=0.10)
    ]

    supp, cont, unc, math_obj = fusion.fuse_directional(claim, evidences)
    assert cont >= 0.70

    verdict, explanation = uncertainty.determine_verdict(claim, supp, cont, unc)
    assert verdict == Verdict.CONTRADICTED
