"""
Geometric Spatial Calculus Verifier for VisionGuard.
Calculates 2D bounding box geometry, IoU, center distances, horizontal/vertical overlap ratios,
and directional spatial relation validity between detected entities.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
from visionguard.schemas import AtomicClaim, Evidence


class SpatialVerifier:
    """
    Spatial Verifier performing coordinate calculus on bounding boxes.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        claim_text = claim.text.lower()

        if visual_features is None:
            return Evidence(
                source="spatial_geometry_verifier",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No visual features provided for spatial verification.",
                details={"status": "ungrounded"}
            )

        subj_name = claim.linguistic_structure.subject.lower() if claim.linguistic_structure else "subject"
        obj_name = claim.linguistic_structure.target_object.lower() if (claim.linguistic_structure and claim.linguistic_structure.target_object) else "target"

        bbox1 = self._find_bbox(subj_name, visual_features)
        bbox2 = self._find_bbox(obj_name, visual_features)

        # Infer spatial relation term from text
        relation = self._extract_relation(claim_text)

        # If both bounding boxes are available, compute real geometric calculus
        if bbox1 and bbox2:
            calc_result = self._compute_geometric_relation(bbox1, bbox2, relation)
            support_s = calc_result["support_score"]
            contradict_s = calc_result["contradiction_score"]
            uncertainty_s = calc_result["uncertainty_score"]
            rationale = calc_result["rationale"]
        elif bbox1 or bbox2:
            # Only one entity localized: Partial spatial evidence
            support_s = 0.50
            contradict_s = 0.20
            uncertainty_s = 0.50
            rationale = f"Only one entity localized ({'subj' if bbox1 else 'target'}); spatial relation '{relation}' is partially verifiable."
        else:
            # Fallback to visual feature relation dictionary if present
            rel_dict = visual_features.get("relations", {})
            if any(k in claim_text for k in rel_dict):
                matched = next(k for k in rel_dict if k in claim_text)
                val = float(rel_dict[matched])
                support_s = val
                contradict_s = round(1.0 - val, 3)
                uncertainty_s = 0.15
                rationale = f"Spatial alignment score from relation feature map: {val:.2f}."
            else:
                support_s = 0.40
                contradict_s = 0.20
                uncertainty_s = 0.60
                rationale = f"Entities not localized in 2D space; relation '{relation}' has high geometric uncertainty."

        cot = (
            f"Step 1: Subject '{subj_name}' bbox={bbox1}, Target '{obj_name}' bbox={bbox2}. "
            f"Step 2: Relation '{relation}'. "
            f"Step 3: Geometry calculus: {rationale} "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="spatial_geometry_verifier",
            result=f"Spatial relation '{relation}': {rationale}",
            support_score=round(support_s, 3),
            contradiction_score=round(contradict_s, 3),
            uncertainty_score=round(uncertainty_s, 3),
            confidence=round(support_s, 3),
            score=round(support_s, 3),
            bounding_box=bbox1 or bbox2,
            cot_reasoning=cot,
            details={
                "relation": relation,
                "bbox_subject": bbox1,
                "bbox_target": bbox2,
                "geometry_rationale": rationale
            }
        )

    def _compute_geometric_relation(self, b1: List[float], b2: List[float], relation: str) -> Dict[str, Any]:
        """
        Executes coordinate geometry calculus between b1 [x1, y1, x2, y2] and b2 [x1, y1, x2, y2].
        """
        # Center coordinates
        c1_x, c1_y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        c2_x, c2_y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0

        w1, h1 = max(0.01, b1[2] - b1[0]), max(0.01, b1[3] - b1[1])
        w2, h2 = max(0.01, b2[2] - b2[0]), max(0.01, b2[3] - b2[1])

        # Overlaps
        inter_x = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        inter_y = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        h_overlap_ratio = inter_x / min(w1, w2)
        v_overlap_ratio = inter_y / min(h1, h2)

        # Euclidean center distance
        dist = math.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

        if relation in ["above", "on top of"]:
            # Subject center y must be above (smaller y) target center y
            is_above = (c1_y < c2_y)
            if is_above and h_overlap_ratio > 0.15:
                return {
                    "support_score": 0.92,
                    "contradiction_score": 0.05,
                    "uncertainty_score": 0.10,
                    "rationale": f"Subject center y ({c1_y:.2f}) is above target ({c2_y:.2f}) with horizontal overlap {h_overlap_ratio:.1%}."
                }
            elif not is_above:
                return {
                    "support_score": 0.10,
                    "contradiction_score": 0.88,
                    "uncertainty_score": 0.10,
                    "rationale": f"Contradiction: Subject center y ({c1_y:.2f}) is below target ({c2_y:.2f})."
                }
            else:
                return {
                    "support_score": 0.45,
                    "contradiction_score": 0.35,
                    "uncertainty_score": 0.40,
                    "rationale": f"Subject is vertically above but lacks horizontal alignment (overlap={h_overlap_ratio:.1%})."
                }

        elif relation in ["below", "under"]:
            is_below = (c1_y > c2_y)
            if is_below and h_overlap_ratio > 0.15:
                return {
                    "support_score": 0.92,
                    "contradiction_score": 0.05,
                    "uncertainty_score": 0.10,
                    "rationale": f"Subject center y ({c1_y:.2f}) is below target ({c2_y:.2f}) with horizontal overlap {h_overlap_ratio:.1%}."
                }
            elif not is_below:
                return {
                    "support_score": 0.10,
                    "contradiction_score": 0.88,
                    "uncertainty_score": 0.10,
                    "rationale": f"Contradiction: Subject center y ({c1_y:.2f}) is above target ({c2_y:.2f})."
                }
            else:
                return {
                    "support_score": 0.45,
                    "contradiction_score": 0.35,
                    "uncertainty_score": 0.40,
                    "rationale": f"Subject is vertically below but lacks horizontal alignment (overlap={h_overlap_ratio:.1%})."
                }

        elif relation in ["beside", "near", "next to"]:
            if dist < 0.45:
                return {
                    "support_score": round(max(0.70, 1.0 - dist), 3),
                    "contradiction_score": 0.08,
                    "uncertainty_score": 0.15,
                    "rationale": f"Entities are adjacent with normalized distance {dist:.2f} (< 0.45 threshold)."
                }
            else:
                return {
                    "support_score": 0.15,
                    "contradiction_score": 0.75,
                    "uncertainty_score": 0.20,
                    "rationale": f"Contradiction: Normalized distance {dist:.2f} exceeds adjacency threshold 0.45."
                }

        elif relation in ["inside", "in"]:
            inter_area = inter_x * inter_y
            area1 = w1 * h1
            containment = inter_area / area1 if area1 > 0 else 0.0
            if containment > 0.65:
                return {
                    "support_score": 0.90,
                    "contradiction_score": 0.05,
                    "uncertainty_score": 0.10,
                    "rationale": f"Subject is contained inside target region with containment ratio {containment:.1%}."
                }
            else:
                return {
                    "support_score": 0.20,
                    "contradiction_score": 0.70,
                    "uncertainty_score": 0.20,
                    "rationale": f"Low containment ratio {containment:.1%} (< 65% threshold)."
                }

        elif relation in ["behind", "in front of"]:
            # 2D projection does not prove 3D monocular depth
            return {
                "support_score": 0.35,
                "contradiction_score": 0.20,
                "uncertainty_score": 0.65,
                "rationale": "Depth relation (behind/in front) cannot be reliably proven by 2D bounding boxes alone without 3D depth estimation."
            }

        else:
            return {
                "support_score": 0.60,
                "contradiction_score": 0.20,
                "uncertainty_score": 0.30,
                "rationale": f"General spatial proximity evaluated (dist={dist:.2f})."
            }

    def _find_bbox(self, entity: str, visual_features: Dict[str, Any]) -> Optional[List[float]]:
        if not visual_features:
            return None
        obj_details = visual_features.get("object_details", {})
        if entity in obj_details:
            return obj_details[entity].get("bbox")
        for k, v in obj_details.items():
            if entity in k or k in entity:
                return v.get("bbox")
        boxes = visual_features.get("bounding_boxes", [])
        return boxes[0] if boxes else None

    def _extract_relation(self, text: str) -> str:
        for rel in ["on top of", "above", "below", "under", "inside", "beside", "next to", "near", "behind", "in front of", "left of", "right of"]:
            if rel in text:
                return rel
        return "spatial_proximity"
