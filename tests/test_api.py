"""
Integration tests for FastAPI endpoints in VisionGuard.
"""

import pytest
from fastapi.testclient import TestClient
from visionguard.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint():
    payload = {
        "response_text": "A young boy wearing a red shirt is riding a bicycle near a park.",
        "visual_features": {
            "objects": {"boy": 0.95, "bicycle": 0.92, "shirt": 0.94},
            "color_distribution": {"red": 0.35},
            "scene": {"type": "outdoor park", "is_outdoor": True}
        }
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    assert len(data["claims"]) >= 1
    assert "repaired_response" in data


def test_repair_endpoint():
    payload = {
        "response_text": "A brown dog is sitting beside a bicycle. The dog is an astronaut.",
        "visual_features": {"objects": {"dog": 0.92, "bicycle": 0.90}}
    }
    response = client.post("/api/repair", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "repaired_response" in data


def test_adversarial_endpoint():
    payload = {
        "claims": ["A man is riding a bicycle.", "The car is red."]
    }
    response = client.post("/api/adversarial/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "adversarial_cases" in data
    assert len(data["adversarial_cases"]) >= 1


def test_experiments_endpoint():
    response = client.get("/api/experiments/run")
    assert response.status_code == 200
    data = response.json()
    assert "experiment_1_claim_classification" in data
    assert "experiment_2_baseline_comparison" in data
    assert "experiment_3_ablation_study" in data
    assert "experiment_4_robustness" in data
    assert "experiment_5_human_alignment" in data
    assert "experiment_6_guardrail_effectiveness" in data


def test_leaderboard_and_hqm():
    res_lb = client.get("/api/leaderboard")
    assert res_lb.status_code == 200
    assert "leaderboard" in res_lb.json()

    res_hqm = client.get("/api/hqm")
    assert res_hqm.status_code == 200
    assert "benchmarks" in res_hqm.json()
