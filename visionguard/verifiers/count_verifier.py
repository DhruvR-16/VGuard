"""
Object Count Verifier for VisionGuard.
Queries YOLO-World / open-vocabulary grounding backend for instance detections,
applies Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes,
counts valid localized instances, and evaluates quantitative assertions.
"""

import re
from typing import Dict, Any, Optional, List, Tuple, Union
from visionguard.schemas import AtomicClaim, Evidence, Polarity
from visionguard.models.registry import registry


class CountVerifier:
    """
    Instance Counting Verifier evaluating numerical and qualitative quantity claims.
    """

    def __init__(self, use_mock: bool = False, iou_threshold: float = 0.40, conf_threshold: float = 0.18):
        self.use_mock = use_mock
        self.registry = registry
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        self.word_to_num = {
            "zero": 0, "none": 0,
            "single": 1, "lone": 1, "one": 1,
            "pair": 2, "couple": 2, "two": 2,
            "triple": 3, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        if visual_features is None and image_path is None:
            return Evidence(
                source="count_verifier",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image context provided for count verification.",
                details={"status": "ungrounded"}
            )

        # 1. Parse entity and expected count from claim
        entity, expected_count, count_type = self._parse_count_and_entity(claim)

        # 2. Run open-vocabulary grounding detection across raw image or visual features
        raw_detections = []
        target_image = image_path or (visual_features.get("image_path") if visual_features else None)

        if target_image and not self.use_mock:
            # Query detector with singular and plural variants
            variants = [entity]
            if not entity.endswith("s"):
                variants.append(entity + "s")
            elif entity.endswith("s") and len(entity) > 3:
                variants.append(entity[:-1])

            raw_detections = self.registry.run_open_vocab_grounding(
                target_image,
                variants,
                conf_threshold=self.conf_threshold
            )

        # 3. Fallback to pre-extracted visual_features if real inference produced no results
        valid_boxes = []
        if raw_detections:
            # Apply Non-Maximum Suppression to remove duplicate boxes
            valid_boxes = self._apply_nms(raw_detections, self.iou_threshold)
        elif visual_features is not None:
            extracted_boxes = self._extract_boxes_from_features(entity, visual_features)
            valid_boxes = self._apply_nms(extracted_boxes, self.iou_threshold)

        detected_count = len(valid_boxes)
        avg_conf = float(sum(d.get("confidence", 0.85) for d in valid_boxes) / detected_count) if detected_count > 0 else 0.0

        # 4. Compare detected count with claimed quantity
        support_s, contradict_s, unc_s, verdict_summary = self._evaluate_count_match(
            detected_count=detected_count,
            expected_count=expected_count,
            count_type=count_type,
            avg_conf=avg_conf,
            polarity=claim.polarity
        )

        top_box = valid_boxes[0]["bbox"] if valid_boxes else None

        cot = (
            f"Step 1: Parsed entity '{entity}' with claimed quantity '{expected_count}' (type: {count_type}). "
            f"Step 2: Grounding detected {len(raw_detections)} raw boxes; NMS retained {detected_count} distinct instances (avg_conf={avg_conf:.2f}). "
            f"Step 3: {verdict_summary} "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={unc_s:.2f}."
        )

        return Evidence(
            source="count_verifier",
            result=verdict_summary,
            support_score=round(support_s, 3),
            contradiction_score=round(contradict_s, 3),
            uncertainty_score=round(unc_s, 3),
            confidence=round(support_s, 3),
            score=round(support_s, 3),
            bounding_box=top_box,
            cot_reasoning=cot,
            details={
                "target_entity": entity,
                "claimed_count": expected_count,
                "detected_count": detected_count,
                "retained_bboxes": [b["bbox"] for b in valid_boxes],
                "count_type": count_type
            }
        )

    def _parse_count_and_entity(self, claim: AtomicClaim) -> Tuple[str, Union[int, str], str]:
        """
        Extracts entity name, numerical or qualitative quantity, and count qualifier.
        """
        text = claim.text.lower()

        # Check qualitative qualifiers first
        if "several" in text:
            entity = self._extract_entity_after(text, "several")
            return entity, "several", "range_several"
        if "multiple" in text or "many" in text:
            kw = "multiple" if "multiple" in text else "many"
            entity = self._extract_entity_after(text, kw)
            return entity, "multiple", "min_two"
        if "pair of" in text or "a pair" in text:
            entity = self._extract_entity_after(text, "pair")
            return entity, 2, "exact"

        # Check explicit digit numbers (e.g. "3 dogs", "12 cars")
        digit_match = re.search(r'\b(\d+)\s+([a-z]+)', text)
        if digit_match:
            num = int(digit_match.group(1))
            entity = digit_match.group(2).rstrip("s")
            return entity, num, "exact"

        # Check number words in order of length descending ("single", "three", "two", etc.)
        for word in sorted(self.word_to_num.keys(), key=len, reverse=True):
            num = self.word_to_num[word]
            pattern = r'\b' + word + r'\s+([a-z]+)'
            match = re.search(pattern, text)
            if match:
                matched_target = match.group(1).rstrip("s")
                # Ensure the matched target is not a preposition/stopword
                if matched_target not in ["of", "in", "on", "at", "by", "is", "are"]:
                    return matched_target, num, "exact"

        # Default fallback to subject
        subj = claim.linguistic_structure.subject.lower() if claim.linguistic_structure else "object"
        entity = subj.rstrip("s")
        return entity, 1, "exact"

    def _extract_entity_after(self, text: str, keyword: str) -> str:
        tokens = text.split()
        if keyword in tokens:
            idx = tokens.index(keyword)
            # Find next non-stopword
            for t in tokens[idx + 1:]:
                clean = re.sub(r'[^\w]', '', t)
                if clean and clean not in ["of", "the", "a", "an", "in", "on", "are", "is"]:
                    return clean.rstrip("s")
        return "object"

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        """
        Applies Non-Maximum Suppression to remove overlapping duplicate detections.
        """
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
        retained = []

        while sorted_dets:
            current = sorted_dets.pop(0)
            retained.append(current)

            remaining = []
            for other in sorted_dets:
                iou = self._compute_iou(current["bbox"], other["bbox"])
                if iou < iou_thresh:
                    remaining.append(other)
            sorted_dets = remaining

        return retained

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Computes Intersection over Union (IoU) between two normalized [x1, y1, x2, y2] boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _extract_boxes_from_features(self, entity: str, visual_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts bounding boxes from pre-extracted visual features for testing or feature datasets.
        """
        boxes = []
        obj_details = visual_features.get("object_details", {})

        for name, data in obj_details.items():
            if entity in name.lower() or name.lower() in entity:
                bbox = data.get("bbox") if isinstance(data, dict) else None
                conf = data.get("confidence", 0.90) if isinstance(data, dict) else 0.90
                if bbox:
                    boxes.append({"bbox": bbox, "confidence": conf, "label": name})

        if not boxes and visual_features.get("bounding_boxes"):
            for b in visual_features["bounding_boxes"]:
                boxes.append({"bbox": b, "confidence": 0.85, "label": entity})

        return boxes

    def _evaluate_count_match(
        self,
        detected_count: int,
        expected_count: Union[int, str],
        count_type: str,
        avg_conf: float,
        polarity: Polarity
    ) -> Tuple[float, float, float, str]:
        """
        Compares detected instance count against claimed expectation with directional distributions.
        """
        is_negative = (polarity == Polarity.NEGATIVE)

        if count_type == "exact":
            exp_int = int(expected_count)
            if detected_count == exp_int:
                if detected_count == 0:
                    supp = 0.85 if is_negative else 0.15
                    cont = 0.10 if is_negative else 0.80
                    unc = 0.05
                    summary = f"Exact count 0 verified for negative assertion." if is_negative else f"Count 0 detected, contradicting positive claim."
                else:
                    supp = min(0.95, 0.70 + 0.25 * avg_conf)
                    cont = 0.05
                    unc = 1.0 - supp - cont
                    summary = f"Exact count verified: Detected {detected_count} distinct instances."
            else:
                diff = abs(detected_count - exp_int)
                cont = min(0.95, 0.65 + 0.10 * diff)
                supp = max(0.05, 0.30 - 0.10 * diff)
                unc = max(0.05, 1.0 - supp - cont)
                summary = f"Count mismatch: Detected {detected_count} instances, conflicting with claimed count of {exp_int}."

        elif count_type == "range_several":
            # "several" typically expects between 3 and 8 instances
            if 3 <= detected_count <= 8:
                supp = min(0.92, 0.70 + 0.20 * avg_conf)
                cont = 0.08
                unc = 1.0 - supp - cont
                summary = f"Qualitative quantity 'several' supported by {detected_count} localized instances."
            elif detected_count > 0:
                supp = 0.45
                cont = 0.35
                unc = 0.20
                summary = f"Detected {detected_count} instances; borderline support for 'several'."
            else:
                supp = 0.10
                cont = 0.80
                unc = 0.10
                summary = f"No localized instances found for 'several'."

        elif count_type == "min_two":
            # "multiple" / "many" expects at least 2 instances
            if detected_count >= 2:
                supp = min(0.92, 0.70 + 0.20 * avg_conf)
                cont = 0.08
                unc = 1.0 - supp - cont
                summary = f"Multiplicity verified: Detected {detected_count} distinct instances."
            elif detected_count == 1:
                supp = 0.30
                cont = 0.55
                unc = 0.15
                summary = f"Only 1 instance detected, contradicting claimed multiplicity."
            else:
                supp = 0.08
                cont = 0.82
                unc = 0.10
                summary = f"Zero instances detected for claimed multiplicity."

        else:
            supp = 0.50
            cont = 0.20
            unc = 0.30
            summary = f"Ambiguous count type '{count_type}'."

        return supp, cont, unc, summary
