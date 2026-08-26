"""
Data models and schemas for VisionGuard (CAM-V).
Defines the 12-class claim taxonomy, 4 formal uncertainty verdicts, evidence provenance,
claim dependency graph, adversarial perturbation structures, and response repair schemas.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """12-class fine-grained claim taxonomy."""
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
    NEGATIVE_CLAIM = "negative_claim"
    OUT_OF_SCOPE = "out_of_scope"


class Polarity(str, Enum):
    """Polarity of the claim (positive assertion vs negative assertion)."""
    POSITIVE = "positive"   # e.g., "A dog is present"
    NEGATIVE = "negative"   # e.g., "There are no cars"


class Certainty(str, Enum):
    """Linguistic certainty of the claim."""
    ASSERTED = "asserted"   # Explicit factual statement
    PROBABLE = "probable"   # "Likely", "probably"
    UNCERTAIN = "uncertain" # "Might be", "possibly"


class Verdict(str, Enum):
    """4 formally defined final decision states."""
    SUPPORTED = "Supported"                         # Sufficient visual evidence supporting claim
    CONTRADICTED = "Contradicted"                   # Sufficient visual evidence conflicts with claim
    INSUFFICIENT_EVIDENCE = "Insufficient_Evidence" # Visual evidence is ambiguous or weak
    OUT_OF_SCOPE = "Out_Of_Scope"                   # Claim cannot be verified from image alone


class LinguisticStructure(BaseModel):
    """Structured semantic breakdown of a claim."""
    subject: str = Field(..., description="Target visual entity, e.g. 'boy', 'dog'")
    predicate: str = Field(..., description="Action or state, e.g. 'is riding', 'sitting'")
    target_object: Optional[str] = Field(None, description="Interacting object, e.g. 'bicycle'")
    modifiers: List[str] = Field(default_factory=list, description="Attributes/colors, e.g. ['red', 'young']")
    spatial_scope: Optional[str] = Field(None, description="Spatial location, e.g. 'near park'")


class AtomicClaim(BaseModel):
    """An atomic, independently verifiable proposition."""
    claim_id: str
    text: str
    type: ClaimType = ClaimType.OBJECT_EXISTENCE
    polarity: Polarity = Polarity.POSITIVE
    certainty: Certainty = Certainty.ASSERTED
    verifiable: bool = True
    linguistic_structure: Optional[LinguisticStructure] = None
    dependencies: List[str] = Field(default_factory=list, description="IDs of parent claims in Dependency Graph")


class Evidence(BaseModel):
    """Traceable evidence unit gathered by a specialized verifier."""
    source: str = Field(..., description="Identifier of verifier, e.g. 'open_vocab_grounding'")
    result: str = Field(default="verified", description="Human-readable result summary")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence [0.0, 1.0]")
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = Field(None, description="[ymin, xmin, ymax, xmax] in [0, 1]")
    region: Optional[List[float]] = None
    cot_reasoning: Optional[str] = Field(None, description="Step-by-step reasoning log")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic parameters")

    def model_post_init(self, __context):
        if self.confidence is None:
            self.confidence = self.score
        if self.region is None and self.bounding_box is not None:
            self.region = self.bounding_box


class FusionMath(BaseModel):
    """Mathematical equation breakdown for evidence fusion."""
    formula: str
    weight_breakdown: Dict[str, float]
    raw_scores: Dict[str, float]
    fused_score: float


class ClaimVerificationResult(BaseModel):
    """Complete verification result for an atomic claim."""
    claim: AtomicClaim
    evidences: List[Evidence] = Field(default_factory=list)
    fused_score: float
    verdict: Verdict
    explanation: str
    fusion_math: Optional[FusionMath] = None
    deep_diagnostics: Dict[str, Any] = Field(default_factory=dict)


class RepairedResponse(BaseModel):
    """Constrained response repair output."""
    original_response: str
    verified_facts: List[str]
    removed_claims: List[str]
    repaired_text: str
    re_verified_verdict: Verdict
    re_verified_confidence: float


class VisionGuardAnalysisResult(BaseModel):
    """Top-level output returned by the CAM-V pipeline."""
    image_id: Optional[str] = None
    response_text: str
    claims: List[ClaimVerificationResult]
    overall_verdict: Verdict
    overall_confidence: float
    hallucination_probability: float
    decomposed_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "main_hal_rate": 0.0,
            "extra_hal_count": 0.0,
            "overall_hal_rate": 0.0
        }
    )
    verdict_counts: Dict[str, int] = Field(default_factory=dict)
    repaired_response: Optional[RepairedResponse] = None


class AdversarialClaim(BaseModel):
    """Systematically perturbed adversarial claim."""
    original_claim: str
    perturbed_claim: str
    perturbation_type: str  # object_swap, attribute_swap, count_change, relation_change, negation, entity_swap, ocr_change
    ground_truth: Verdict = Verdict.CONTRADICTED


class HQMMetrics(BaseModel):
    """HQM Psychometric quality indicators."""
    test_retest_reliability: float
    parallel_forms_reliability: float
    content_validity: float
    criterion_validity: float


class ModelLeaderboardEntry(BaseModel):
    """Entry in the 8-dimension perceptual leaderboard."""
    rank: int
    model: str
    main_hal_rate: float
    extra_hal_count: float
    overall_hal_rate: float
    dimensions: Dict[str, float]


class ExperimentMetrics(BaseModel):
    """Metrics for evaluation experiments."""
    precision: float
    recall: float
    f1_score: float
    macro_f1: float
    auroc: float
    cohens_kappa: Optional[float] = None
    test_retest_r: Optional[float] = None
    parallel_forms_rho: Optional[float] = None
