"""
Specialized Verifier Router for VisionGuard.
Routes atomic claims across the 12-class taxonomy to specialized primary and secondary verifiers.
"""

from typing import List, Dict, Any, Optional
from visionguard.schemas import AtomicClaim, ClaimType, Evidence
from visionguard.verifiers.object_verifier import OpenVocabularyGroundingVerifier
from visionguard.verifiers.attribute_verifier import AttributeVerifier
from visionguard.verifiers.spatial_verifier import SpatialVerifier
from visionguard.verifiers.action_verifier import ActionInteractionVerifier
from visionguard.verifiers.ocr_verifier import OCRVerifier
from visionguard.verifiers.scene_verifier import SceneVerifier
from visionguard.verifiers.absence_verifier import AbsenceVerifier


class EvidenceRouter:
    """
    Routes atomic claims to domain-specialized verifiers according to the 12-class taxonomy.
    """

    def __init__(self, use_mock: bool = False):
        self.object_verifier = OpenVocabularyGroundingVerifier(use_mock=use_mock)
        self.attribute_verifier = AttributeVerifier(use_mock=use_mock)
        self.spatial_verifier = SpatialVerifier(use_mock=use_mock)
        self.action_verifier = ActionInteractionVerifier(use_mock=use_mock)
        self.ocr_verifier = OCRVerifier(use_mock=use_mock)
        self.scene_verifier = SceneVerifier(use_mock=use_mock)
        self.absence_verifier = AbsenceVerifier(use_mock=use_mock)

    def route_and_verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> List[Evidence]:
        """
        Gathers multi-source evidence based on claim type.
        """
        evidences: List[Evidence] = []

        if claim.type == ClaimType.OBJECT_EXISTENCE:
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.OBJECT_COUNT:
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.spatial_verifier.verify(claim, image_path, visual_features))

        elif claim.type in [ClaimType.COLOR, ClaimType.ATTRIBUTE]:
            evidences.append(self.attribute_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.ACTION:
            evidences.append(self.action_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.spatial_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.INTERACTION:
            evidences.append(self.action_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.SPATIAL_RELATION:
            evidences.append(self.spatial_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.action_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.COMPARISON:
            evidences.append(self.spatial_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.attribute_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.OCR_TEXT:
            evidences.append(self.ocr_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.SCENE:
            evidences.append(self.scene_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.NEGATIVE_CLAIM:
            evidences.append(self.absence_verifier.verify(claim, image_path, visual_features))
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))

        elif claim.type == ClaimType.OUT_OF_SCOPE:
            evidences.append(self.scene_verifier.verify(claim, image_path, visual_features))

        else:
            evidences.append(self.object_verifier.verify(claim, image_path, visual_features))

        return evidences
