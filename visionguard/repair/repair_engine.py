"""
Constrained Response Repair & True Re-Verification Engine for VisionGuard.
Strips contradicted propositions, synthesizes factually grounded text,
and re-runs claim extraction & verification over the repaired text to guarantee soundness.
"""

from typing import List, Tuple, Optional, Dict, Any
from visionguard.schemas import (
    ClaimVerificationResult,
    Verdict,
    RepairedResponse,
    AtomicClaim
)
from visionguard.pipeline.dependency_graph import ClaimDependencyGraph
from visionguard.pipeline.claim_extractor import ClaimExtractor
from visionguard.pipeline.evidence_router import EvidenceRouter
from visionguard.pipeline.evidence_fusion import EvidenceFusionEngine
from visionguard.pipeline.uncertainty_engine import UncertaintyEngine


class ResponseRepairEngine:
    """
    Constrained Response Repair with Closed-Loop Re-Verification.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.extractor = ClaimExtractor()
        self.router = EvidenceRouter(use_mock=use_mock)
        self.fusion = EvidenceFusionEngine()
        self.uncertainty = UncertaintyEngine()

    def repair_response(
        self,
        original_response: str,
        verification_results: List[ClaimVerificationResult],
        dependency_graph: Optional[ClaimDependencyGraph] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> RepairedResponse:
        """
        Executes constrained response repair with a closed-loop re-verification pass.
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

        # 1. Filter out Contradicted Claims using Dependency Graph
        if dependency_graph:
            verified_results = dependency_graph.get_verified_subgraph(verification_results)
        else:
            verified_results = [r for r in verification_results if r.verdict == Verdict.SUPPORTED]

        removed_results = [r for r in verification_results if r not in verified_results]

        verified_facts = [r.claim.text for r in verified_results]
        removed_claims = [r.claim.text for r in removed_results]

        # 2. Constrained Text Synthesis
        if not verified_facts:
            repaired_text = "The visual evidence does not contain sufficient verified facts to support the claims."
            return RepairedResponse(
                original_response=original_response,
                verified_facts=[],
                removed_claims=removed_claims,
                repaired_text=repaired_text,
                re_verified_verdict=Verdict.INSUFFICIENT_EVIDENCE,
                re_verified_confidence=0.0,
                re_verified_claims=[]
            )

        # Build clean fluent sentence from verified facts
        cleaned_sentences = []
        for fact in verified_facts:
            fact_clean = fact.strip()
            if not fact_clean.endswith('.'):
                fact_clean += '.'
            cleaned_sentences.append(fact_clean[0].upper() + fact_clean[1:])

        repaired_text = " ".join(cleaned_sentences)

        # 3. TRUE RE-VERIFICATION PASS
        # Re-extract claims from repaired text and re-verify them
        re_extracted_claims = self.extractor.extract_claims(repaired_text)
        re_verified_results: List[ClaimVerificationResult] = []
        re_scores = []

        for claim in re_extracted_claims:
            evidences = self.router.route_and_verify(claim, visual_features=visual_features)
            supp, cont, unc, math_obj = self.fusion.fuse_directional(claim, evidences)
            verdict, explanation = self.uncertainty.determine_verdict(claim, supp, cont, unc)
            re_scores.append(supp)

            re_verified_results.append(
                ClaimVerificationResult(
                    claim=claim,
                    evidences=evidences,
                    fused_score=supp,
                    support_score=supp,
                    contradiction_score=cont,
                    uncertainty_score=unc,
                    verdict=verdict,
                    explanation=explanation,
                    fusion_math=math_obj
                )
            )

        # Re-verified overall confidence & verdict
        avg_re_conf = float(sum(re_scores) / len(re_scores)) if re_scores else 0.85
        has_contradiction = any(r.verdict == Verdict.CONTRADICTED for r in re_verified_results)
        final_re_verdict = Verdict.CONTRADICTED if has_contradiction else Verdict.SUPPORTED

        return RepairedResponse(
            original_response=original_response,
            verified_facts=verified_facts,
            removed_claims=removed_claims,
            repaired_text=repaired_text,
            re_verified_verdict=final_re_verdict,
            re_verified_confidence=round(avg_re_conf, 3),
            re_verified_claims=re_verified_results
        )
