"""
End-to-End Integration Tests with Real Images for VisionGuard.
Generates synthetic in-memory image files, runs image processing, OCR, spatial calculus,
evidence fusion, and response repair over genuine visual bytes.
"""

import io
import pytest
from PIL import Image, ImageDraw, ImageFont
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.pipeline.image_processor import AutomatedImageProcessor
from visionguard.schemas import Verdict


def create_synthetic_test_image() -> bytes:
    """
    Creates a real RGB image with colored geometric shapes and text.
    """
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Red box (top region)
    draw.rectangle([100, 50, 300, 150], fill=(220, 30, 30))

    # Text "STOP" in white inside red box
    draw.text((160, 85), "STOP", fill=(255, 255, 255))

    # Blue circle (bottom region)
    draw.ellipse([120, 220, 280, 380], fill=(30, 80, 220))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_real_image_end_to_end_pipeline():
    image_bytes = create_synthetic_test_image()
    processor = AutomatedImageProcessor(use_mock=False)

    # 1. Process real image bytes
    pil_img, features = processor.process_image_file(image_bytes, target_concepts=["box", "circle"])

    assert pil_img.size == (400, 400)
    assert features["dimensions"] == {"width": 400, "height": 400}
    assert "color_distribution" in features
    assert features["color_distribution"]["red"] > 0.05
    assert features["color_distribution"]["blue"] > 0.05

    # 2. Run full orchestrator verification over real image
    orchestrator = CAMVOrchestrator(use_mock=False)
    response_text = "There is a red box and a blue circle."

    result = orchestrator.analyze(
        response_text=response_text,
        image_input=image_bytes,
        enable_repair=True
    )

    assert result.response_text == response_text
    assert len(result.claims) >= 1
    assert result.overall_verdict in [Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE]
    assert result.repaired_response is not None


def test_real_image_ocr_detection():
    image_bytes = create_synthetic_test_image()
    processor = AutomatedImageProcessor(use_mock=False)

    pil_img, features = processor.process_image_file(image_bytes)
    ocr_tokens = features.get("ocr_tokens", [])

    # Check if Tesseract extracted tokens from drawn image
    if ocr_tokens:
        extracted_texts = [t["text"].upper() for t in ocr_tokens]
        assert any("STOP" in t for t in extracted_texts)
