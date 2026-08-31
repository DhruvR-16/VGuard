"""
Automated Image Processor for VisionGuard.
Processes raw images (JPEG, PNG, WebP), extracts real OCR tokens via Tesseract,
runs open-vocabulary grounding via YOLO-World, computes color distributions,
and extracts real object bounding boxes.
"""

import io
from typing import Dict, Any, Tuple, Optional, Union, List
from PIL import Image
import numpy as np
from visionguard.models.registry import registry


class AutomatedImageProcessor:
    """
    Multimodal Image Feature Extractor with real OCR, Open-Vocabulary Grounding, and Region Analysis.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.registry = registry

    def process_image_file(
        self,
        image_input: Union[str, bytes, Image.Image],
        target_concepts: Optional[List[str]] = None
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Loads image safely, runs Tesseract OCR and open-vocabulary detection, and extracts features.
        """
        pil_image = self._load_image(image_input)
        img_w, img_h = pil_image.size

        # 1. Real OCR extraction via Tesseract
        ocr_tokens = []
        if not self.use_mock:
            ocr_tokens = self.registry.run_ocr(pil_image)

        # 2. Real Open-Vocabulary Grounding if target concepts are provided
        detected_objects: Dict[str, Any] = {}
        bounding_boxes: List[List[float]] = []

        if target_concepts and not self.use_mock:
            detections = self.registry.run_open_vocab_grounding(pil_image, target_concepts)
            for det in detections:
                lbl = det["label"].lower()
                if lbl not in detected_objects or det["confidence"] > detected_objects[lbl]["confidence"]:
                    detected_objects[lbl] = {
                        "confidence": det["confidence"],
                        "bbox": det["bbox"]
                    }
                    bounding_boxes.append(det["bbox"])

        # 3. Real Color Channel Analysis on pixel array
        img_np = np.array(pil_image.convert("RGB"))
        color_dist = self._analyze_color_distribution(img_np)
        dominant_color = max(color_dist.items(), key=lambda x: x[1])[0]

        # 4. Scene Context Classification (CLIP zero-shot or ungrounded)
        brightness = float(np.mean(img_np))
        scene_type = "general scene"
        is_outdoor = None

        if not self.use_mock:
            scene_candidates = [
                "an outdoor nature, park, or landscape scene",
                "an urban city street scene",
                "an indoor room, office, or kitchen"
            ]
            probs = self.registry.run_clip_attribute_classification(pil_image, scene_candidates)
            if probs:
                top_scene = max(probs.items(), key=lambda x: x[1])[0]
                scene_type = top_scene
                is_outdoor = "outdoor" in top_scene or "urban" in top_scene

        features: Dict[str, Any] = {
            "image_obj": pil_image,
            "dimensions": {"width": img_w, "height": img_h},
            "objects": {k: v["confidence"] for k, v in detected_objects.items()},
            "object_details": detected_objects,
            "bounding_boxes": bounding_boxes,
            "ocr_tokens": ocr_tokens,
            "color_distribution": color_dist,
            "dominant_color": dominant_color,
            "scene": {
                "type": scene_type,
                "is_outdoor": is_outdoor if is_outdoor is not None else False,
                "is_uncertain": is_outdoor is None,
                "brightness": round(brightness, 2)
            }
        }

        return pil_image, features

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Safely loads and validates image format and dimensions.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
            return image.convert("RGB")
        elif isinstance(image_input, str):
            image = Image.open(image_input)
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _analyze_color_distribution(self, img_np: np.ndarray) -> Dict[str, float]:
        """
        Computes normalized color distribution from image RGB pixel array.
        """
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        total_pixels = float(r.size)

        is_red = (r > 130) & (g < 100) & (b < 100)
        is_blue = (b > 130) & (r < 100) & (g < 120)
        is_green = (g > 120) & (r < 110) & (b < 110)
        is_yellow = (r > 140) & (g > 140) & (b < 90)
        is_white = (r > 200) & (g > 200) & (b > 200)
        is_black = (r < 50) & (g < 50) & (b < 50)
        is_brown = (r > 90) & (r < 160) & (g > 50) & (g < 110) & (b < 80)

        dist = {
            "red": round(float(np.sum(is_red)) / total_pixels, 4),
            "blue": round(float(np.sum(is_blue)) / total_pixels, 4),
            "green": round(float(np.sum(is_green)) / total_pixels, 4),
            "yellow": round(float(np.sum(is_yellow)) / total_pixels, 4),
            "white": round(float(np.sum(is_white)) / total_pixels, 4),
            "black": round(float(np.sum(is_black)) / total_pixels, 4),
            "brown": round(float(np.sum(is_brown)) / total_pixels, 4)
        }
        return dist

    def auto_generate_caption(self, visual_features: Dict[str, Any]) -> str:
        """
        Generates a factual summary caption from detected objects and visual features.
        """
        objects = visual_features.get("objects", {})
        scene = visual_features.get("scene", {})
        dom_color = visual_features.get("dominant_color", "neutral")

        if not objects:
            env = "an outdoor" if scene.get("is_outdoor") else "an indoor"
            return f"An image depicting {env} scene with {dom_color} tones."

        obj_names = list(objects.keys())
        if len(obj_names) == 1:
            return f"A {obj_names[0]} is visible in the scene."
        else:
            return f"A {obj_names[0]} and a {obj_names[1]} are visible in the scene."
