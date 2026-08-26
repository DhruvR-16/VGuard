"""
Data Loader Module for VisionGuard datasets and pre-computed leaderboard samples.
"""

from typing import List, Dict, Any


def get_sample_dataset() -> List[Dict[str, Any]]:
    """
    Returns sample dataset for testing, benchmarking, and demonstration.
    """
    return [
        {
            "id": "sample_001",
            "image_id": "vg_1001",
            "response": "A young boy wearing a red shirt is riding a bicycle near a park.",
            "is_hallucinated": False,
            "ground_truth": "A boy wearing a red shirt riding a bicycle near a park.",
            "visual_features": {
                "objects": {"boy": 0.95, "bicycle": 0.92, "shirt": 0.94, "park": 0.88},
                "attributes": {"red shirt": 0.91, "young boy": 0.89},
                "relations": {"riding bicycle": 0.93, "near park": 0.88},
                "scene": "park"
            }
        },
        {
            "id": "sample_002",
            "image_id": "vg_1002",
            "response": "A brown dog is sitting beside a red bicycle in Paris.",
            "is_hallucinated": True,
            "ground_truth": "A brown dog sitting near a blue bicycle in a yard.",
            "visual_features": {
                "objects": {"dog": 0.92, "bicycle": 0.89},
                "attributes": {"brown dog": 0.90, "red bicycle": 0.25},
                "relations": {"sitting beside": 0.86},
                "scene": "yard"
            }
        },
        {
            "id": "sample_003",
            "image_id": "vg_1003",
            "response": "There are three cats eating green vegetables on a white desk.",
            "is_hallucinated": False,
            "ground_truth": "Cat and green vegetables on white desk.",
            "visual_features": {
                "objects": {"cat": 0.94, "green vegetables": 0.92, "white desk": 0.90},
                "attributes": {"green vegetables": 0.95, "white desk": 0.91},
                "relations": {"eating": 0.88},
                "scene": "indoor"
            }
        },
        {
            "id": "sample_004",
            "image_id": "vg_1004",
            "response": "An alien is driving a flying car above a snowy field.",
            "is_hallucinated": True,
            "ground_truth": "A car driving on a snowy road.",
            "visual_features": {
                "objects": {"car": 0.85, "snowy field": 0.88},
                "attributes": {"snowy field": 0.89},
                "relations": {"driving": 0.82},
                "scene": "snow"
            }
        }
    ]


def get_leaderboard_data() -> List[Dict[str, Any]]:
    """
    Returns benchmark leaderboard evaluations on HQH across 8 perceptual dimensions (arXiv:2406.17115v3).
    """
    return [
        {
            "rank": 1,
            "model": "GPT-4o",
            "main_hal_rate": 0.330,
            "extra_hal_count": 0.220,
            "overall_hal_rate": 0.370,
            "dimensions": {
                "existence": 0.309, "count": 0.250, "color": 0.226, "action": 0.280,
                "spatial": 0.392, "comparison": 0.310, "environment": 0.290, "text": 0.315
            }
        },
        {
            "rank": 2,
            "model": "LLaVA-OneVision-7B",
            "main_hal_rate": 0.355,
            "extra_hal_count": 0.122,
            "overall_hal_rate": 0.389,
            "dimensions": {
                "existence": 0.304, "count": 0.275, "color": 0.310, "action": 0.320,
                "spatial": 0.403, "comparison": 0.335, "environment": 0.300, "text": 0.340
            }
        },
        {
            "rank": 3,
            "model": "Qwen2-VL-7B",
            "main_hal_rate": 0.320,
            "extra_hal_count": 0.246,
            "overall_hal_rate": 0.400,
            "dimensions": {
                "existence": 0.329, "count": 0.260, "color": 0.227, "action": 0.290,
                "spatial": 0.363, "comparison": 0.305, "environment": 0.285, "text": 0.310
            }
        },
        {
            "rank": 4,
            "model": "Gemini-1.5-Pro",
            "main_hal_rate": 0.441,
            "extra_hal_count": 0.231,
            "overall_hal_rate": 0.480,
            "dimensions": {
                "existence": 0.406, "count": 0.320, "color": 0.334, "action": 0.370,
                "spatial": 0.512, "comparison": 0.410, "environment": 0.380, "text": 0.390
            }
        },
        {
            "rank": 5,
            "model": "LLaVA-1.5-13B",
            "main_hal_rate": 0.498,
            "extra_hal_count": 0.248,
            "overall_hal_rate": 0.550,
            "dimensions": {
                "existence": 0.509, "count": 0.410, "color": 0.367, "action": 0.420,
                "spatial": 0.558, "comparison": 0.480, "environment": 0.450, "text": 0.470
            }
        },
        {
            "rank": 6,
            "model": "InstructBLIP-Vicuna-7B",
            "main_hal_rate": 0.559,
            "extra_hal_count": 0.301,
            "overall_hal_rate": 0.610,
            "dimensions": {
                "existence": 0.502, "count": 0.480, "color": 0.379, "action": 0.450,
                "spatial": 0.678, "comparison": 0.530, "environment": 0.510, "text": 0.540
            }
        },
        {
            "rank": 7,
            "model": "MiniGPT4-Vicuna-7B",
            "main_hal_rate": 0.765,
            "extra_hal_count": 0.728,
            "overall_hal_rate": 0.832,
            "dimensions": {
                "existence": 0.694, "count": 0.650, "color": 0.635, "action": 0.690,
                "spatial": 0.866, "comparison": 0.740, "environment": 0.710, "text": 0.750
            }
        }
    ]


def get_hqm_benchmark_quality_table() -> List[Dict[str, Any]]:
    """
    Returns HQM Benchmark Quality Measurement results table comparing POPE, AMBER, OpenCHAIR, MMHal, GAVIE, and HQH.
    """
    return [
        {"name": "POPE", "task": "Yes-or-No", "metric": "Accuracy", "test_retest": 0.9996, "parallel_forms": 0.3563, "content_validity": 0.84, "criterion_validity": 0.9634},
        {"name": "AMBER-d", "task": "Yes-or-No", "metric": "Accuracy", "test_retest": 0.9986, "parallel_forms": 0.3636, "content_validity": 0.90, "criterion_validity": 0.9321},
        {"name": "AMBER-g", "task": "Captioning", "metric": "CHAIR", "test_retest": 0.9378, "parallel_forms": 0.5333, "content_validity": 0.74, "criterion_validity": 0.8774},
        {"name": "OpenCHAIR", "task": "Captioning", "metric": "OCH", "test_retest": 0.9896, "parallel_forms": 0.5510, "content_validity": 0.76, "criterion_validity": 0.6818},
        {"name": "MMHal", "task": "Free-form", "metric": "Hal Score", "test_retest": 0.8829, "parallel_forms": 0.8833, "content_validity": 0.79, "criterion_validity": 0.7518},
        {"name": "GAVIE", "task": "Free-form", "metric": "Hal Score", "test_retest": 0.9091, "parallel_forms": 0.9136, "content_validity": 0.88, "criterion_validity": 0.7938},
        {"name": "HQH (Ours)", "task": "Free-form", "metric": "Decomposed Hal Rate", "test_retest": 0.9977, "parallel_forms": 0.9856, "content_validity": 0.98, "criterion_validity": 0.9556},
    ]
