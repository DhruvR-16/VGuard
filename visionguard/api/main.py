"""
FastAPI Server for VisionGuard (CAM-V).
Exposes RESTful endpoints for claim-aware multi-evidence verification, response repair,
adversarial generation, 6 evaluation experiments, and real-time hallucination guardrails.
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from visionguard import __version__
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.eval.benchmark_runner import BenchmarkRunner
from visionguard.eval.ablation import AblationStudyRunner
from visionguard.eval.adversarial import AdversarialHallucinationGenerator
from visionguard.eval.experiment_claim_classification import ClaimClassificationExperiment
from visionguard.eval.experiment_robustness import RobustnessExperiment
from visionguard.eval.experiment_human_alignment import HumanAlignmentExperiment
from visionguard.eval.experiment_guardrail import GuardrailEffectivenessExperiment
from visionguard.data_loader import (
    get_sample_dataset,
    get_leaderboard_data,
    get_hqm_benchmark_quality_table
)

app = FastAPI(
    title="VisionGuard CAM-V API",
    description="Claim-Aware Multi-Evidence Detection, Response Repair & Psychometric Quality Evaluation for LVLMs",
    version=__version__
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = CAMVOrchestrator(use_mock=True)
benchmark_runner = BenchmarkRunner(use_mock=True)
ablation_runner = AblationStudyRunner(use_mock=True)
adversarial_gen = AdversarialHallucinationGenerator()
claim_class_exp = ClaimClassificationExperiment(use_mock=True)
robustness_exp = RobustnessExperiment(use_mock=True)
human_align_exp = HumanAlignmentExperiment()
guardrail_exp = GuardrailEffectivenessExperiment(use_mock=True)


class AnalyzeRequest(BaseModel):
    response_text: str
    image_id: Optional[str] = "img_001"
    visual_features: Optional[Dict[str, Any]] = None
    fusion_mode: Optional[str] = "claim_aware"
    enable_repair: Optional[bool] = True


class RepairRequest(BaseModel):
    response_text: str
    visual_features: Optional[Dict[str, Any]] = None


class AdversarialRequest(BaseModel):
    claims: List[str]


class GuardrailRequest(BaseModel):
    prompt: str
    response_text: str
    enable_task_identifier: bool = True
    enable_claim_filter: bool = True


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": __version__, "service": "VisionGuard CAM-V Engine"}


@app.post("/api/analyze")
def analyze_response(req: AnalyzeRequest):
    """
    Runs full CAM-V claim decomposition, multi-evidence fusion, and verdict assignment.
    """
    if not req.response_text or not req.response_text.strip():
        raise HTTPException(status_code=400, detail="response_text cannot be empty.")

    res = orchestrator.analyze(
        response_text=req.response_text,
        visual_features=req.visual_features,
        image_id=req.image_id,
        enable_repair=req.enable_repair
    )
    return res.model_dump()


@app.post("/api/upload_and_analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    response_text: Optional[str] = Form(None),
    fusion_mode: Optional[str] = Form("claim_aware"),
    enable_repair: Optional[bool] = Form(True)
):
    """
    Automated Endpoint: Upload any raw image file (JPEG, PNG, WebP) and optional caption.
    Auto-extracts visual features and executes CAM-V verification + response repair.
    """
    try:
        contents = await file.read()
        res = orchestrator.analyze(
            response_text=response_text,
            image_input=contents,
            image_id=file.filename,
            enable_repair=enable_repair
        )
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded image: {str(e)}")


@app.post("/api/repair")
def repair_response(req: RepairRequest):
    """
    Performs constrained response repair and re-verification over factual assertions.
    """
    res = orchestrator.analyze(
        response_text=req.response_text,
        visual_features=req.visual_features,
        enable_repair=True
    )
    return {
        "original_response": req.response_text,
        "repaired_response": res.repaired_response.model_dump() if res.repaired_response else None,
        "overall_verdict": res.overall_verdict.value,
        "hallucination_probability": res.hallucination_probability
    }


@app.post("/api/adversarial/generate")
def generate_adversarial_suite(req: AdversarialRequest):
    """
    Generates paired adversarial test cases across 7 perturbation operators.
    """
    cases = adversarial_gen.generate_adversarial_suite(req.claims)
    return {"adversarial_cases": [c.model_dump() for c in cases]}


@app.get("/api/experiments/run")
def run_all_experiments():
    """
    Runs all 6 core evaluation experiments.
    """
    samples = get_sample_dataset()
    return {
        "experiment_1_claim_classification": claim_class_exp.run_experiment(samples),
        "experiment_2_baseline_comparison": benchmark_runner.run_benchmark(samples),
        "experiment_3_ablation_study": ablation_runner.run_ablation(samples),
        "experiment_4_robustness": robustness_exp.run_experiment(samples),
        "experiment_5_human_alignment": human_align_exp.run_experiment(300),
        "experiment_6_guardrail_effectiveness": guardrail_exp.run_experiment(samples)
    }


@app.get("/api/leaderboard")
def get_leaderboard():
    """
    Returns HQH 8-dimension model evaluation leaderboard.
    """
    return {"leaderboard": get_leaderboard_data()}


@app.get("/api/hqm")
def get_hqm_quality():
    """
    Returns HQM Benchmark Quality Measurement results.
    """
    return {"benchmarks": get_hqm_benchmark_quality_table()}


@app.get("/api/benchmark/run")
def run_benchmark_evaluation():
    """
    Runs benchmark comparison across CAM-V and 4 baselines.
    """
    samples = get_sample_dataset()
    results = benchmark_runner.run_benchmark(samples)
    return results


@app.get("/api/ablation/run")
def run_ablation_experiment():
    """
    Runs Section 14 Ablation Study across 6 configurations (A0 to A5).
    """
    samples = get_sample_dataset()
    results = ablation_runner.run_ablation(samples)
    return results


@app.post("/api/guardrail/apply")
def apply_guardrail(req: GuardrailRequest):
    """
    Applies real-time hallucination mitigation and constrained response repair.
    """
    decorated_prompt = f"[vqa] {req.prompt}" if req.enable_task_identifier else req.prompt
    res = orchestrator.analyze(req.response_text, enable_repair=True)

    repaired_text = res.repaired_response.repaired_text if res.repaired_response else req.response_text
    removed_claims = res.repaired_response.removed_claims if res.repaired_response else []

    return {
        "original_prompt": req.prompt,
        "decorated_prompt": decorated_prompt,
        "original_response": req.response_text,
        "mitigated_response": repaired_text,
        "removed_claims": removed_claims,
        "hallucination_probability": res.hallucination_probability,
        "verdict": res.overall_verdict.value,
        "decomposed_metrics": res.decomposed_metrics
    }


# Mount Static Files for Web Dashboard
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
