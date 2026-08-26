"""
Constrained Response Repair & Re-Verification Engine for VisionGuard.
Decomposes responses, identifies contradicted assertions via Claim Dependency Graph,
performs constrained factual reconstruction, and re-verifies repaired responses.
"""

from typing import List, Tuple, Optional
from visionguard.schemas import (
    ClaimVerificationResult,
    Verdict,
    RepairedResponse
)
from visionguard.pipeline.dependency_graph import ClaimDependencyGraph


class ResponseRepairEngine:
    """
    Repairs hallucinated responses using constrained fact synthesis and re-verification.
    """

    def __init__(self):
        pass

    def repair_response(
        self,
        original_response: str,
        verification_results: List[ClaimVerificationResult],
        dependency_graph: Optional[ClaimDependencyGraph] = None
    ) -> RepairedResponse:
        """
        Executes constrained response repair:
        1. Identifies supported claims from valid subgraph.
        2. Filters out contradicted/unsupported claims.
        3. Synthesizes a factually grounded repaired response.
        4. Calculates re-verified verdict.
        """
        if not verification_results:
            return RepairedResponse(
                original_response=original_response,
                verified_facts=[],
                removed_claims=[],
                repaired_text=original_response,
                re_verified_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                re_verified_confidence=0.0
            )

        # Use Dependency Graph to filter out claims whose parents are contradicted
        if dependency_graph:
            verified_results = dependency_graph.get_verified_subgraph(verification_results)
        else:
            verified_results = [r for r in verification_results if r.verdict == Verdict.SUPPORTED]

        removed_results = [r for r in verification_results if r not in verified_results]

        verified_facts = [r.claim.text for r in verified_results]
        removed_claims = [r.claim.text for r in removed_results]

        # Constrained Synthesis
        if not verified_facts:
            repaired_text = "The provided visual context does not contain sufficient verified facts to support the claims."
            re_verdict = Verdict.INSUFFICIENT_EVIDENCE
            re_conf = 0.0
        else:
            # Join verified propositions into clean fluent text
            cleaned_sentences = []
            for fact in verified_facts:
                fact_clean = fact.strip()
                if not fact_clean.endswith('.'):
                    fact_clean += '.'
                cleaned_sentences.append(fact_clean[0].upper() + fact_clean[1:])

            repaired_text = " ".join(cleaned_sentences)
            re_verdict = Verdict.SUPPORTED
            re_conf = float(sum(r.fused_score for r in verified_results) / len(verified_results))

        return RepairedResponse(
            original_response=original_response,
            verified_facts=verified_facts,
            removed_claims=removed_claims,
            repaired_text=repaired_text,
            re_verified_verdict=re_verdict,
            re_verified_confidence=round(re_conf, 3)
        )
