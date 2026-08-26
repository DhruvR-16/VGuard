"""
Uncertainty Engine for VisionGuard.
Maps multi-evidence fused scores and contextual constraints into the 4 formal decision states:
1. SUPPORTED (Score >= 0.70)
2. CONTRADICTED (Score < 0.40)
3. INSUFFICIENT_EVIDENCE (0.40 <= Score < 0.70)
4. OUT_OF_SCOPE (Non-visual / external knowledge assertion)
"""

from typing import Tuple
from visionguard.schemas import AtomicClaim, Verdict, ClaimType


class UncertaintyEngine:
    """
    Classifies atomic claim confidence into 4 formal uncertainty verdicts.
    """

    def __init__(
        self,
        support_threshold: float = 0.70,
        contradict_threshold: float = 0.40
    ):
        self.support_threshold = support_threshold
        self.contradict_threshold = contradict_threshold

    def determine_verdict(self, claim: AtomicClaim, fused_score: float) -> Tuple[Verdict, str]:
        # 1. Out of Scope Check
        if claim.type == ClaimType.OUT_OF_SCOPE or not claim.verifiable:
            return (
                Verdict.OUT_OF_SCOPE,
                "Claim requires external geographical / non-visual knowledge and cannot be verified from the image alone."
            )

        # 2. Supported
        if fused_score >= self.support_threshold:
            return (
                Verdict.SUPPORTED,
                f"Sufficient visual evidence supports claim with confidence {fused_score:.1%}."
            )

        # 3. Contradicted
        elif fused_score < self.contradict_threshold:
            return (
                Verdict.CONTRADICTED,
                f"Visual contradiction detected. Evidence confidence {fused_score:.1%} is below contradiction threshold."
            )

        # 4. Insufficient Evidence
        else:
            return (
                Verdict.INSUFFICIENT_EVIDENCE,
                f"Available visual evidence is inconclusive or ambiguous (confidence {fused_score:.1%})."
            )
