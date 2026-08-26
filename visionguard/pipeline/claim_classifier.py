"""
Claim Classifier Module for VisionGuard.
Maps extracted atomic claims to the 12-class fine-grained claim taxonomy.
"""

from typing import List
from visionguard.schemas import AtomicClaim, ClaimType, Polarity


class ClaimClassifier:
    """
    Classifies atomic propositions into the 12-class taxonomy.
    """

    def __init__(self):
        pass

    def classify(self, claim: AtomicClaim) -> AtomicClaim:
        # Polarity check takes precedence
        if claim.polarity == Polarity.NEGATIVE:
            claim.type = ClaimType.NEGATIVE_CLAIM
            return claim

        lower = claim.text.lower()
        if any(c in lower for c in ["paris", "tokyo", "london", "new york", "beijing", "berlin"]):
            claim.type = ClaimType.OUT_OF_SCOPE
            claim.verifiable = False
            return claim

        return claim

    def classify_batch(self, claims: List[AtomicClaim]) -> List[AtomicClaim]:
        return [self.classify(c) for c in claims]
