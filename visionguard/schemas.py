"""
Pydantic Schemas & Data Structures for VisionGuard (CAM-V).
Defines directional evidence representation, 12 semantic claim types,
orthogonal polarity, formal 4-state verdicts, and traceable evidence provenance.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    OBJECT_EXISTENCE = "object_existence"
    OBJECT_COUNT = "object_count"
    COLOR = "color"
    ATTRIBUTE = "attribute"
    ACTION = "action"
    INTERACTION = "interaction"
    SPATIAL_RELATION = "spatial_relation"
    COMPARISON = "comparison"
    OCR_TEXT = "ocr_text"
    SCENE = "scene"
    IDENTITY = "identity"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    # Legacy alias support
    NEGATIVE_CLAIM = "object_existence"
    OUT_OF_SCOPE = "external_knowledge"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Verifiability(str, Enum):
    VISUAL = "visual"
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    AMBIGUOUS = "ambiguous"


class Certainty(str, Enum):
    ASSERTED = "asserted"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"


class Verdict(str, Enum):
    SUPPORTED = "Supported"
    CONTRADICTED = "Contradicted"
    INSUFFICIENT_EVIDENCE = "Insufficient_Evidence"
    OUT_OF_SCOPE = "Out_Of_Scope"


class LinguisticStructure(BaseModel):
    subject: str
    predicate: str
    target_object: Optional[str] = None
    modifiers: List[str] = Field(default_factory=list)
    spatial_scope: Optional[str] = None


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    type: ClaimType
    polarity: Polarity = Polarity.POSITIVE
    verifiability: Verifiability = Verifiability.VISUAL
    certainty: Certainty = Certainty.ASSERTED
    verifiable: bool = True
    linguistic_structure: Optional[LinguisticStructure] = None
    dependencies: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    source: str
    result: str
    support_score: float = 0.0
    contradiction_score: float = 0.0
    uncertainty_score: float = 1.0
    confidence: float = 0.0
    # Backward compatibility score property
    score: float = 0.0
    bounding_box: Optional[List[float]] = None  # Normalized [x1, y1, x2, y2]
    cot_reasoning: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        # If scalar score was provided without directional distribution, auto-populate
        if self.score > 0 and self.support_score == 0.0 and self.contradiction_score == 0.0:
            if self.score >= 0.70:
                self.support_score = self.score
                self.contradiction_score = round(1.0 - self.score, 3)
                self.uncertainty_score = 0.10
            elif self.score <= 0.35:
                self.contradiction_score = round(1.0 - self.score, 3)
                self.support_score = self.score
                self.uncertainty_score = 0.10
            else:
                self.uncertainty_score = 0.70
                self.support_score = self.score
                self.contradiction_score = 0.20
            self.confidence = self.score
        elif self.support_score > 0 and self.score == 0.0:
            self.score = self.support_score


class FusionMath(BaseModel):
    formula: str
    weight_breakdown: Dict[str, float] = Field(default_factory=dict)
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    support_fused: float = 0.0
    contradict_fused: float = 0.0
    uncertainty_fused: float = 0.0
    fused_score: float = 0.0


class ClaimVerificationResult(BaseModel):
    claim: AtomicClaim
    evidences: List[Evidence] = Field(default_factory=list)
    fused_score: float = 0.0
    support_score: float = 0.0
    contradiction_score: float = 0.0
    uncertainty_score: float = 1.0
    verdict: Verdict
    explanation: str
    fusion_math: Optional[FusionMath] = None
    deep_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class RepairedResponse(BaseModel):
    original_response: str
    verified_facts: List[str]
    removed_claims: List[str]
    repaired_text: str
    re_verified_verdict: Verdict
    re_verified_confidence: float
    re_verified_claims: List[ClaimVerificationResult] = Field(default_factory=list)


class VisionGuardAnalysisResult(BaseModel):
    image_id: Optional[str] = None
    response_text: str
    claims: List[ClaimVerificationResult]
    overall_verdict: Verdict
    overall_confidence: float
    hallucination_probability: float
    decomposed_metrics: Dict[str, float]
    verdict_counts: Dict[str, int]
    repaired_response: Optional[RepairedResponse] = None


class AdversarialClaim(BaseModel):
    original_claim: str
    perturbed_claim: str
    perturbation_type: str
    ground_truth: Verdict = Verdict.CONTRADICTED


class ExperimentMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    auroc: float


class HQMMetrics(BaseModel):
    test_retest_reliability: Optional[float] = None
    parallel_forms_reliability: Optional[float] = None
    content_validity: Optional[float] = None
    criterion_validity: Optional[float] = None
    overall_quality_score: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkQualityReport(BaseModel):
    benchmark_name: str
    task_type: str
    metric_type: str
    metrics: HQMMetrics
    is_acceptable: bool

