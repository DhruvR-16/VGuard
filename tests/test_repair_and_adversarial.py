"""
Unit tests for Response Repair with Re-Verification and Adversarial Generation.
"""

import pytest
from visionguard.repair.repair_engine import ResponseRepairEngine
from visionguard.eval.adversarial import AdversarialHallucinationGenerator
from visionguard.schemas import AtomicClaim, ClaimVerificationResult, Verdict, ClaimType


def test_response_repair_with_reverification():
    engine = ResponseRepairEngine(use_mock=True)
    c1 = AtomicClaim(claim_id="C1", text="A brown dog is present", type=ClaimType.OBJECT_EXISTENCE)
    c2 = AtomicClaim(claim_id="C2", text="The dog is flying", type=ClaimType.ACTION)

    r1 = ClaimVerificationResult(claim=c1, fused_score=0.92, support_score=0.92, contradiction_score=0.05, uncertainty_score=0.10, verdict=Verdict.SUPPORTED, explanation="Grounded")
    r2 = ClaimVerificationResult(claim=c2, fused_score=0.10, support_score=0.10, contradiction_score=0.85, uncertainty_score=0.10, verdict=Verdict.CONTRADICTED, explanation="Contradicted")

    repaired = engine.repair_response(
        original_response="A brown dog is present. The dog is flying.",
        verification_results=[r1, r2],
        visual_features={"objects": {"dog": 0.92, "brown dog": 0.90}}
    )

    assert "dog is present" in repaired.repaired_text.lower()
    assert "flying" not in repaired.repaired_text.lower()
    assert len(repaired.removed_claims) == 1
    assert repaired.re_verified_verdict == Verdict.SUPPORTED


def test_adversarial_generator():
    generator = AdversarialHallucinationGenerator()
    base_claims = ["A violinist is playing a cello beside a microphone."]
    cases = generator.generate_adversarial_suite(base_claims)

    assert len(cases) >= 1
    types = [c.perturbation_type for c in cases]
    assert any(t in ["relation_swap", "negation_inversion", "count_perturbation", "object_swap", "entity_swap"] for t in types)
