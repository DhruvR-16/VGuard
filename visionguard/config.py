"""
VisionGuard Configuration & Reproducibility Manager.
Defines system modes (production/development/test), reproducible seeds,
confidence thresholds, output directories, and model backend specifications.
"""

import os
import random
from typing import Dict, Any, List
import numpy as np
import torch

# System Execution Mode
VISIONGUARD_MODE = os.getenv("VISIONGUARD_MODE", "production").lower()  # 'production', 'development', 'test'
VISIONGUARD_DEVICE = os.getenv("VISIONGUARD_DEVICE", "auto").lower()
STRICT_REAL = os.getenv("VISIONGUARD_STRICT_REAL", "False").lower() in ("true", "1")

# Confidence & Decision Thresholds
SUPPORT_THRESHOLD = float(os.getenv("VISIONGUARD_SUPPORT_THRESHOLD", "0.65"))
CONTRADICT_THRESHOLD = float(os.getenv("VISIONGUARD_CONTRADICT_THRESHOLD", "0.55"))
UNCERTAINTY_THRESHOLD = 0.45
SPATIAL_ADJACENCY_THRESHOLD = 0.45
CONTAINMENT_THRESHOLD = 0.65
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# Output Directories for Experiment Artifacts & Logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPERIMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "experiments")
LOGS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "logs")

for directory in [OUTPUT_DIR, EXPERIMENT_OUTPUT_DIR, LOGS_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Backend Configuration
MODEL_CONFIGS = {
    "grounding": {
        "model_id": "yolov8s-worldv2.pt",
        "conf_threshold": 0.12,
        "iou_threshold": 0.45
    },
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "temperature": 0.07
    },
    "vqa": {
        "model_id": "dandelin/vilt-b32-finetuned-vqa",
        "top_k": 5
    }
}


def seed_everything(seed: int = 42) -> None:
    """
    Sets global deterministic seeds for Python, NumPy, and PyTorch to guarantee reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> str:
    """
    Resolves active compute device.
    """
    if VISIONGUARD_DEVICE in ["cuda", "mps", "cpu"]:
        return VISIONGUARD_DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
