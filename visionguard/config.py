"""
VisionGuard configuration file.
Defines thresholds, fusion weights, claim categories, and verifier options.
"""

from typing import Dict

# Claim Categories as specified in Section 5 of VisionGuard Plan & paper
CLAIM_TYPES = ["object", "attribute", "relation", "scene"]

# Default Fusion Weights per Claim Type
DEFAULT_FUSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "object": {
        "vlm_score": 0.35,
        "similarity_score": 0.15,
        "object_detector_score": 0.50,
    },
    "attribute": {
        "vlm_score": 0.40,
        "similarity_score": 0.40,
        "object_detector_score": 0.20,
    },
    "relation": {
        "vlm_score": 0.70,
        "similarity_score": 0.15,
        "object_detector_score": 0.15,
    },
    "scene": {
        "vlm_score": 0.45,
        "similarity_score": 0.45,
        "object_detector_score": 0.10,
    },
}

# Uncertainty Thresholds for Verdict Classification
SUPPORTED_THRESHOLD = 0.75
CONTRADICTED_THRESHOLD = 0.35
# Scores between 0.35 and 0.75 are UNCERTAIN

# Verifiability Threshold
VERIFIABILITY_KEYWORD_BLACK_LIST = [
    "belongs to", "owns", "bought", "thinks", "feels", "wishes", 
    "yesterday", "tomorrow", "history of", "intent", "mind"
]

# Benchmark Quality Metrics Default Config
HQM_CONFIG = {
    "test_retest_seeds": [42, 1337],
    "parallel_prompt_formats": ["original", "negotiated_yn", "rephrased_vqa"],
    "noise_threshold": 0.10
}
