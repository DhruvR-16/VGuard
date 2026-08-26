"""
Unit tests for Response Repair and Adversarial Hallucination Generation.
"""

import pytest
from visionguard.repair.repair_engine import ResponseRepairEngine
from visionguard.eval.adversarial import AdversarialHallucinationGenerator
from visionguard.schemas import AtomicClaim, ClaimVerificationResult, Verdict, ClaimType


def test_response_repair_engine():
    engine = ResponseRepairEngine()
    c1 = AtomicClaim(claim_id="C1", text="A brown dog is present", type=ClaimType.OBJECT_EXISTENCE)
    c2 = AtomicClaim(claim_id="C2", text="The dog is flying", type=ClaimType.ACTION)

    r1 = ClaimVerificationResult(claim=c1, fused_score=0.92, verdict=Verdict.SUPPORTED, explanation="Grounded")
    r2 = ClaimVerificationResult(claim=c2, fused_score=0.10, verdict=Verdict.CONTRADICTED, explanation="Contradicted")

    repaired = engine.repair_response(
        original_response="A brown dog is present. The dog is flying.",
        verification_results=[r1, r2]
    )

    assert "dog is present" in repaired.repaired_text.lower()
    assert "flying" not in repaired.repaired_text.lower()
    assert len(repaired.removed_claims) == 1
    assert repaired.re_verified_verdict == Verdict.SUPPORTED


def test_adversarial_generator():
    generator = AdversarialHallucinationGenerator()
    base_claims = ["A man is riding a bicycle near the river."]
    cases = generator.generate_adversarial_suite(base_claims)

    assert len(cases) >= 2
    types = [c.perturbation_type for c in cases]
    assert "object_swap" in types or "entity_swap" in types or "relation_swap" in types
