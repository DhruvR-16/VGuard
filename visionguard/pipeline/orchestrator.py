"""
CAM-V Orchestrator for VisionGuard.
Coordinates end-to-end open-vocabulary visual grounding, directional evidence fusion,
4-state uncertainty modeling, and closed-loop response repair with true re-verification.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np

from visionguard.schemas import (
    AtomicClaim,
    ClaimVerificationResult,
    VisionGuardAnalysisResult,
    Verdict,
    ClaimType,
    RepairedResponse
)
from visionguard.pipeline.claim_extractor import ClaimExtractor
from visionguard.pipeline.claim_classifier import ClaimClassifier
from visionguard.pipeline.evidence_router import EvidenceRouter
from visionguard.pipeline.evidence_fusion import EvidenceFusionEngine
from visionguard.pipeline.uncertainty_engine import UncertaintyEngine
from visionguard.pipeline.image_processor import AutomatedImageProcessor
from visionguard.pipeline.dependency_graph import ClaimDependencyGraph
from visionguard.repair.repair_engine import ResponseRepairEngine


class CAMVOrchestrator:
    """
    Main Orchestrator executing the complete CAM-V verification and repair pipeline.
    """

    def __init__(self, fusion_mode: str = "claim_aware", use_mock: bool = False):
        self.use_mock = use_mock
        self.extractor = ClaimExtractor()
        self.classifier = ClaimClassifier()
        self.router = EvidenceRouter(use_mock=use_mock)
        self.fusion_engine = EvidenceFusionEngine(mode=fusion_mode)
        self.uncertainty_engine = UncertaintyEngine()
        self.image_processor = AutomatedImageProcessor(use_mock=use_mock)
        self.repair_engine = ResponseRepairEngine(use_mock=use_mock)

    def analyze(
        self,
        response_text: Optional[str] = None,
        image_input: Optional[Union[str, bytes, Any]] = None,
        visual_features: Optional[Dict[str, Any]] = None,
        image_id: Optional[str] = "img_001",
        enable_repair: bool = True
    ) -> VisionGuardAnalysisResult:
        # Step 0: Extract Initial Claims to identify target concepts for open-vocabulary grounding
        target_concepts = []
        if response_text and response_text.strip():
            temp_claims = self.extractor.extract_claims(response_text)
            for c in temp_claims:
                if c.linguistic_structure and c.linguistic_structure.subject:
                    target_concepts.append(c.linguistic_structure.subject)
                    if c.linguistic_structure.modifiers:
                        target_concepts.append(" ".join(c.linguistic_structure.modifiers) + " " + c.linguistic_structure.subject)
                    if c.linguistic_structure.target_object:
                        target_concepts.append(c.linguistic_structure.target_object)

        # Step 1: Process Image if provided
        has_visual_context = False
        if image_input is not None and visual_features is None:
            _, extracted_features = self.image_processor.process_image_file(image_input, target_concepts=target_concepts)
            visual_features = extracted_features
            has_visual_context = True
        elif visual_features is not None:
            has_visual_context = True

        if (not response_text or not response_text.strip()) and visual_features:
            response_text = self.image_processor.auto_generate_caption(visual_features)

        if not response_text or not response_text.strip():
            raise ValueError("No input provided. Please upload an image or provide text to analyze.")

        # Step 2: Claim Extraction & Dependency Graph Construction
        atomic_claims = self.extractor.extract_claims(response_text)
        dependency_graph = ClaimDependencyGraph(atomic_claims)

        # Step 3: Claim Classification
        classified_claims = self.classifier.classify_batch(atomic_claims)

        # Step 4: Ungrounded Validation if NO visual context is attached
        if not has_visual_context:
            verification_results = []
            for claim in classified_claims:
                claim.verifiable = False
                verification_results.append(
                    ClaimVerificationResult(
                        claim=claim,
                        evidences=[],
                        fused_score=0.0,
                        support_score=0.0,
                        contradiction_score=0.0,
                        uncertainty_score=1.0,
                        verdict=Verdict.OUT_OF_SCOPE,
                        explanation="No image was provided for visual grounding. Upload an image to verify this claim.",
                        deep_diagnostics={
                            "linguistic_structure": claim.linguistic_structure.model_dump() if claim.linguistic_structure else {},
                            "bounding_boxes": [],
                            "cot_logs": ["No image input provided to ground visual claim."],
                            "verifiers_used": [],
                            "support_rationale": "Ungrounded: Image required for visual verification."
                        }
                    )
                )

            return VisionGuardAnalysisResult(
                image_id=image_id,
                response_text=response_text,
                claims=verification_results,
                overall_verdict=Verdict.OUT_OF_SCOPE,
                overall_confidence=0.0,
                hallucination_probability=0.0,
                decomposed_metrics={
                    "main_hal_rate": 0.0,
                    "extra_hal_count": 0.0,
                    "overall_hal_rate": 0.0,
                },
                verdict_counts={Verdict.OUT_OF_SCOPE.value: len(verification_results)},
                repaired_response=None
            )

        # Step 5: Route to Specialized Verifiers, Collect Directional Evidence, Fuse & Assign Verdict
        verification_results: List[ClaimVerificationResult] = []
        verdict_counts = {v.value: 0 for v in Verdict}
        support_scores = []

        for claim in classified_claims:
            evidences = self.router.route_and_verify(claim, visual_features=visual_features)
            supp, cont, unc, math_obj = self.fusion_engine.fuse_directional(claim, evidences)
            support_scores.append(supp)

            verdict, explanation = self.uncertainty_engine.determine_verdict(claim, supp, cont, unc)
            verdict_counts[verdict.value] = verdict_counts.get(verdict.value, 0) + 1

            deep_diagnostics = {
                "linguistic_structure": claim.linguistic_structure.model_dump() if claim.linguistic_structure else {},
                "dependencies": claim.dependencies,
                "bounding_boxes": [e.bounding_box for e in evidences if e.bounding_box],
                "cot_logs": [e.cot_reasoning for e in evidences if e.cot_reasoning],
                "verifiers_used": [e.source for e in evidences],
                "support_rationale": (
                    f"Claim supported (Support={supp:.1%}, Contradict={cont:.1%}, Uncertainty={unc:.1%})."
                    if verdict == Verdict.SUPPORTED else
                    f"Contradiction/Uncertainty detected (Support={supp:.1%}, Contradict={cont:.1%}, Uncertainty={unc:.1%})."
                )
            }

            verification_results.append(
                ClaimVerificationResult(
                    claim=claim,
                    evidences=evidences,
                    fused_score=round(supp, 3),
                    support_score=round(supp, 3),
                    contradiction_score=round(cont, 3),
                    uncertainty_score=round(unc, 3),
                    verdict=verdict,
                    explanation=explanation,
                    fusion_math=math_obj,
                    deep_diagnostics=deep_diagnostics
                )
            )

        # Step 6: Propagate Contradictions via Dependency Graph
        verification_results = dependency_graph.propagate_contradictions(verification_results)

        # Step 7: Closed-Loop Response Repair with True Re-Verification
        repaired_response = None
        if enable_repair:
            repaired_response = self.repair_engine.repair_response(
                original_response=response_text,
                verification_results=verification_results,
                dependency_graph=dependency_graph,
                visual_features=visual_features
            )

        # Step 8: Calculate Decomposed Metrics
        avg_confidence = float(np.mean(support_scores)) if support_scores else 0.50
        hallucinated_claims = [r for r in verification_results if r.verdict == Verdict.CONTRADICTED]
        uncertain_claims = [r for r in verification_results if r.verdict in [Verdict.INSUFFICIENT_EVIDENCE, Verdict.OUT_OF_SCOPE]]

        hallucination_prob = (len(hallucinated_claims) + 0.5 * len(uncertain_claims)) / len(verification_results) if verification_results else 0.0

        if verdict_counts[Verdict.CONTRADICTED.value] > 0:
            overall_verdict = Verdict.CONTRADICTED
        elif verdict_counts[Verdict.INSUFFICIENT_EVIDENCE.value] > 0:
            overall_verdict = Verdict.INSUFFICIENT_EVIDENCE
        elif verdict_counts[Verdict.OUT_OF_SCOPE.value] == len(verification_results):
            overall_verdict = Verdict.OUT_OF_SCOPE
        else:
            overall_verdict = Verdict.SUPPORTED

        main_answer_hal = 1.0 if (verification_results and verification_results[0].verdict in [Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE]) else 0.0
        extra_hal_count = float(len([r for r in verification_results[1:] if r.verdict in [Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE]]))
        overall_hal_rate = 1.0 if (main_answer_hal > 0 or extra_hal_count > 0) else 0.0

        return VisionGuardAnalysisResult(
            image_id=image_id,
            response_text=response_text,
            claims=verification_results,
            overall_verdict=overall_verdict,
            overall_confidence=round(avg_confidence, 3),
            hallucination_probability=round(hallucination_prob, 3),
            decomposed_metrics={
                "main_hal_rate": main_answer_hal,
                "extra_hal_count": extra_hal_count,
                "overall_hal_rate": overall_hal_rate,
            },
            verdict_counts=verdict_counts,
            repaired_response=repaired_response
        )
