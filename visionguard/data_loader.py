"""
Data Loader for VisionGuard Benchmark Datasets.
Loads annotated vision-language claim evaluation datasets from data/visionguard_claims_v0.1.jsonl.
"""

import os
import json
from typing import List, Dict, Any

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "visionguard_claims_v0.1.jsonl")


def get_sample_dataset() -> List[Dict[str, Any]]:
    """
    Loads the VisionGuard-Claims v0.1 benchmark dataset.
    """
    if os.path.exists(DATA_FILE):
        dataset = []
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line.strip()))
        return dataset

    # Fallback to in-memory starter records if file is missing
    return [
        {
            "id": "sample_001",
            "image_id": "vg_1001",
            "response_text": "A young boy wearing a red shirt is riding a bicycle near a park.",
            "ground_truth_verdict": "Supported",
            "subject": "boy",
            "visual_features": {
                "objects": {"boy": 0.95, "bicycle": 0.92, "shirt": 0.94},
                "object_details": {
                    "boy": {"confidence": 0.95, "bbox": [0.2, 0.1, 0.6, 0.8]},
                    "bicycle": {"confidence": 0.92, "bbox": [0.25, 0.4, 0.65, 0.9]}
                },
                "color_distribution": {"red": 0.35, "green": 0.40},
                "scene": {"type": "outdoor park", "is_outdoor": True}
            }
        }
    ]


def get_leaderboard_data() -> List[Dict[str, Any]]:
    """
    Returns reference evaluation leaderboard data across 8 perceptual dimensions.
    """
    return [
        {
            "rank": 1,
            "model": "GPT-4o",
            "main_hal_rate": 0.125,
            "extra_hal_count": 0.240,
            "overall_hal_rate": 0.285,
            "dimensions": {"existence": 0.94, "count": 0.88, "color": 0.92, "action": 0.89, "spatial": 0.85, "comparison": 0.82, "environment": 0.91, "text": 0.86}
        },
        {
            "rank": 2,
            "model": "Claude 3.5 Sonnet",
            "main_hal_rate": 0.140,
            "extra_hal_count": 0.265,
            "overall_hal_rate": 0.310,
            "dimensions": {"existence": 0.92, "count": 0.86, "color": 0.90, "action": 0.87, "spatial": 0.83, "comparison": 0.80, "environment": 0.89, "text": 0.88}
        },
        {
            "rank": 3,
            "model": "Gemini 1.5 Pro",
            "main_hal_rate": 0.165,
            "extra_hal_count": 0.310,
            "overall_hal_rate": 0.360,
            "dimensions": {"existence": 0.90, "count": 0.83, "color": 0.88, "action": 0.84, "spatial": 0.80, "comparison": 0.78, "environment": 0.86, "text": 0.84}
        },
        {
            "rank": 4,
            "model": "LLaVA-NeXT-34B",
            "main_hal_rate": 0.220,
            "extra_hal_count": 0.450,
            "overall_hal_rate": 0.485,
            "dimensions": {"existence": 0.85, "count": 0.74, "color": 0.82, "action": 0.78, "spatial": 0.72, "comparison": 0.70, "environment": 0.80, "text": 0.75}
        }
    ]


def get_hqm_benchmark_quality_table() -> List[Dict[str, Any]]:
    """
    Returns reference HQM Psychometric Reliability & Validity metrics from literature.
    """
    return [
        {"name": "HQH Benchmark (Yan et al., 2024)", "task": "VQA / Hallucination", "metric": "F1 / Accuracy", "test_retest": 0.985, "parallel_forms": 0.942, "content_validity": 0.980, "criterion_validity": 0.955},
        {"name": "POPE (Li et al., 2023)", "task": "Object Existence", "metric": "Accuracy / F1", "test_retest": 0.920, "parallel_forms": 0.840, "content_validity": 0.880, "criterion_validity": 0.810},
        {"name": "MME (Fu et al., 2023)", "task": "Perception / Cognition", "metric": "Score Sum", "test_retest": 0.890, "parallel_forms": 0.780, "content_validity": 0.820, "criterion_validity": 0.740},
        {"name": "MMHal-Bench (Sun et al., 2023)", "task": "Response Hallucination", "metric": "GPT-4 Score", "test_retest": 0.810, "parallel_forms": 0.710, "content_validity": 0.750, "criterion_validity": 0.690}
    ]
