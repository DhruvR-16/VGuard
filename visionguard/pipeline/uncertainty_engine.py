"""
Directional Uncertainty Engine for VisionGuard.
Maps directional evidence distributions (support, contradiction, uncertainty)
into 4 formal decision states: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE.
Requires strict evidential threshold for active contradiction, distinguishing
evidence for absence/incompatibility from mere absence of visual evidence.
"""

from typing import Tuple
from visionguard.schemas import AtomicClaim, Verdict, ClaimType, Verifiability


class UncertaintyEngine:
    """
    Directional Uncertainty Engine.
    """

    def __init__(
        self,
        support_threshold: float = 0.65,
        contradict_threshold: float = 0.68,
        margin: float = 0.15
    ):
        self.support_threshold = support_threshold
        self.contradict_threshold = contradict_threshold
        self.margin = margin

    def determine_verdict(
        self,
        claim: AtomicClaim,
        support_score: float,
        contradiction_score: float = 0.0,
        uncertainty_score: float = 0.0
    ) -> Tuple[Verdict, str]:
        # 1. Out of Scope Check (Non-Visual / External Knowledge)
        if (
            claim.verifiability == Verifiability.EXTERNAL_KNOWLEDGE
            or claim.type == ClaimType.EXTERNAL_KNOWLEDGE
            or not claim.verifiable
        ):
            return (
                Verdict.OUT_OF_SCOPE,
                "Claim requires external non-visual metadata and cannot be verified from image features alone."
            )

        # 2. Supported Check
        if support_score >= self.support_threshold and support_score > (contradiction_score + self.margin):
            return (
                Verdict.SUPPORTED,
                f"Sufficient visual evidence supports claim with confidence {support_score:.1%} (contradiction={contradiction_score:.1%})."
            )

        # 3. Contradicted Check (Requires high contradiction confidence exceeding support by margin)
        elif contradiction_score >= self.contradict_threshold and contradiction_score > (support_score + self.margin):
            return (
                Verdict.CONTRADICTED,
                f"Visual contradiction detected (contradiction confidence {contradiction_score:.1%}, support={support_score:.1%})."
            )

        # 4. Insufficient Evidence Check (Absence of evidence, weak localization, or ambiguous signals)
        else:
            return (
                Verdict.INSUFFICIENT_EVIDENCE,
                f"Visual evidence is ambiguous or inconclusive (support={support_score:.1%}, contradict={contradiction_score:.1%}, uncertainty={uncertainty_score:.1%})."
            )
