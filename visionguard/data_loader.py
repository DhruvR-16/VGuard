"""
Data Loader for VisionGuard Benchmark Datasets.
Loads annotated vision-language claim evaluation datasets, raw image paths,
and multi-annotator ground truth from data/annotations/claims.jsonl and data/visionguard_claims_v0.1.jsonl.
"""

import os
import json
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "visionguard_claims_v0.1.jsonl")
CLAIMS_FILE = os.path.join(BASE_DIR, "data", "annotations", "claims.jsonl")
SPLITS_FILE = os.path.join(BASE_DIR, "data", "annotations", "splits.json")


def get_sample_dataset() -> List[Dict[str, Any]]:
    """
    Loads the image-backed VisionGuard-Claims benchmark dataset.
    """
    if os.path.exists(DATA_FILE):
        dataset = []
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if "image_path" in item:
                        # Ensure absolute path or relative to project root
                        full_img_path = os.path.join(BASE_DIR, item["image_path"])
                        if os.path.exists(full_img_path):
                            item["full_image_path"] = full_img_path
                    dataset.append(item)
        return dataset

    return []


def get_atomic_claims_dataset(split: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads independent atomic claim annotations with multi-annotator ground truth.
    Optionally filters by split ('train', 'val', 'test').
    """
    if not os.path.exists(CLAIMS_FILE):
        return []

    split_ids = None
    if split and os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
            split_ids = set(splits_data.get(split, []))

    claims = []
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if split_ids is not None and item.get("claim_id") not in split_ids:
                    continue
                if "image_path" in item:
                    full_img_path = os.path.join(BASE_DIR, item["image_path"])
                    if os.path.exists(full_img_path):
                        item["full_image_path"] = full_img_path
                claims.append(item)

    return claims


def get_dataset_splits() -> Dict[str, List[str]]:
    """
    Returns train / val / test claim IDs.
    """
    if os.path.exists(SPLITS_FILE):
        with open(SPLITS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"train": [], "val": [], "test": []}


def get_leaderboard_data() -> List[Dict[str, Any]]:
    """
    Returns reference evaluation leaderboard data across 8 perceptual dimensions from literature.
    """
    return [
        {
            "rank": 1,
            "model": "GPT-4o",
            "source": "Literature Reference",
            "main_hal_rate": 0.125,
            "extra_hal_count": 0.240,
            "overall_hal_rate": 0.285,
            "dimensions": {"existence": 0.94, "count": 0.88, "color": 0.92, "action": 0.89, "spatial": 0.85, "comparison": 0.82, "environment": 0.91, "text": 0.86}
        },
        {
            "rank": 2,
            "model": "Claude 3.5 Sonnet",
            "source": "Literature Reference",
            "main_hal_rate": 0.140,
            "extra_hal_count": 0.265,
            "overall_hal_rate": 0.310,
            "dimensions": {"existence": 0.92, "count": 0.86, "color": 0.90, "action": 0.87, "spatial": 0.83, "comparison": 0.80, "environment": 0.89, "text": 0.88}
        },
        {
            "rank": 3,
            "model": "Gemini 1.5 Pro",
            "source": "Literature Reference",
            "main_hal_rate": 0.165,
            "extra_hal_count": 0.310,
            "overall_hal_rate": 0.360,
            "dimensions": {"existence": 0.90, "count": 0.83, "color": 0.88, "action": 0.84, "spatial": 0.80, "comparison": 0.78, "environment": 0.86, "text": 0.84}
        },
        {
            "rank": 4,
            "model": "LLaVA-NeXT-34B",
            "source": "Literature Reference",
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
        {"name": "HQH Benchmark (Yan et al., 2024)", "source": "Literature Reference", "task": "VQA / Hallucination", "metric": "F1 / Accuracy", "test_retest": 0.985, "parallel_forms": 0.942, "content_validity": 0.980, "criterion_validity": 0.955},
        {"name": "POPE (Li et al., 2023)", "source": "Literature Reference", "task": "Object Existence", "metric": "Accuracy / F1", "test_retest": 0.920, "parallel_forms": 0.840, "content_validity": 0.880, "criterion_validity": 0.810},
        {"name": "MME (Fu et al., 2023)", "source": "Literature Reference", "task": "Perception / Cognition", "metric": "Score Sum", "test_retest": 0.890, "parallel_forms": 0.780, "content_validity": 0.820, "criterion_validity": 0.740},
        {"name": "MMHal-Bench (Sun et al., 2023)", "source": "Literature Reference", "task": "Response Hallucination", "metric": "GPT-4 Score", "test_retest": 0.810, "parallel_forms": 0.710, "content_validity": 0.750, "criterion_validity": 0.690}
    ]
