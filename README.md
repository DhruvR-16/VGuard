# VisionGuard

> **Claim-Aware Multi-Evidence Hallucination Detection, Response Repair & Psychometric Quality Evaluation for Vision-Language Models**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-18%2F18%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [Why Existing Approaches Are Insufficient](#3-why-existing-approaches-are-insufficient)
4. [Project Goal](#4-project-goal)
5. [What VisionGuard Does](#5-what-visionguard-does)
6. [Core Architectural Blueprint (CAM-V)](#6-core-architectural-blueprint-cam-v)
7. [Claim Decomposition & Dependency Graph](#7-claim-decomposition--dependency-graph)
8. [12-Class Claim Taxonomy](#8-12-class-claim-taxonomy)
9. [Open-Vocabulary Evidence Verification & Verifier Router](#9-open-vocabulary-evidence-verification--verifier-router)
10. [Evidence Fusion & 4-State Uncertainty Modeling](#10-evidence-fusion--4-state-uncertainty-modeling)
11. [Evidence Provenance](#11-evidence-provenance)
12. [Constrained Response Repair & Re-Verification](#12-constrained-response-repair--re-verification)
13. [Adversarial Hallucination Generation](#13-adversarial-hallucination-generation)
14. [Evaluation Methodology & 6 Core Experiments](#14-evaluation-methodology--6-core-experiments)
15. [HQM Psychometric Reliability & Validity](#15-hqm-psychometric-reliability--validity)
16. [Repository Structure](#16-repository-structure)
17. [RESTful API Specification](#17-restful-api-specification)
18. [CLI Reference](#18-cli-reference)
19. [What Was Removed, Replaced, and Added](#19-what-was-removed-replaced-and-added)
20. [Final Research Contribution](#20-final-research-contribution)
21. [References & Citation](#21-references--citation)

---

## 1. Project Overview

Large Vision-Language Models (LVLMs) can generate fluent descriptions, answers, and reasoning about images. However, they frequently generate statements that are **plausible, linguistically convincing, but visually incorrect**—a phenomenon known as *visual hallucination*.

For example, given an image of a person riding a bicycle, a VLM may generate:
> *"A young man wearing a red jacket is riding a bicycle near the Eiffel Tower."*

This response contains multiple independent factual assertions:
- $C_1$: A person exists.
- $C_2$: The person is young.
- $C_3$: The person is male.
- $C_4$: The person is wearing a jacket.
- $C_5$: The jacket is red.
- $C_6$: The person is riding a bicycle.
- $C_7$: The scene is near the Eiffel Tower (geographic/external).

Some may be factually correct, while others are completely unsupported. The fundamental problem is that existing evaluation approaches judge the entire response as a single monolithic unit.

**VisionGuard asks:** *Can we decompose a Vision-Language Model response into individual factual claims and verify each claim using the most appropriate source of visual evidence?*

---

## 2. The Problem

Consider the response:
> *"A woman wearing a yellow jacket is riding a bicycle through a busy city street while a red car follows behind her."*

A traditional hallucination evaluator sends the entire response to another model and asks: *"Is this response consistent with the image?"*

This fails because the evaluator must simultaneously reason about:
- Object existence
- Colors and attributes
- Physical actions and interactions
- Spatial positions and geometry
- Object counts
- Global scene and environmental context

A single monolithic model may be effective at one type of verification and poor at another. VisionGuard therefore decomposes the response into atomic propositions ($C_1 \dots C_8$), builds a dependency graph, and verifies each proposition independently using specialized visual evidence sources.

---

## 3. Why Existing Approaches Are Insufficient

The research paper used as an evaluation foundation (*Yan et al., arXiv:2406.17115v3, "Measuring the Measurers"*) identifies key structural flaws in existing hallucination benchmarks:
- **Response bias** in closed-ended questions.
- **Score instability** caused by prompt variation.
- **Annotation noise** in benchmark testbeds.
- **Disagreement** between automated monolithic evaluators and human experts.
- **Conflation** of direct question errors with incidental elaborative hallucinations.

The paper proposes **HQM** (Hallucination Quality Measurement) for assessing benchmark quality and **HQH** as an 8-dimension benchmark. 

### Relationship to the Paper:
```
┌────────────────────────────────────────┐
│             HQM / HQH                  │
│       (Yan et al., 2024)               │
│  • Provides evaluation methodology     │
│  • Motivates psychometric reliability  │
│  • Separates 8 perceptual dimensions   │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│              VisionGuard               │
│  • Proposes claim-aware verification   │
│  • Open-vocabulary visual grounding    │
│  • Modular specialized verifier router │
│  • Constrained response repair         │
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│           CAM-V Framework              │
│  • Active runtime verification engine  │
│  • Claim Dependency Graph (DAG)        │
│  • 4-state formal uncertainty fusion   │
└────────────────────────────────────────┘
```

---

## 4. Project Goal

VisionGuard provides an end-to-end framework that:
1. Receives an image and a VLM-generated response.
2. Decomposes the response into atomic factual claims with explicit polarity.
3. Classifies each claim into a 12-class taxonomy.
4. Routes the claim to specialized domain verifiers.
5. Gathers open-vocabulary visual evidence and bounding box coordinates.
6. Fuses evidence using claim-aware learned models.
7. Classifies the claim into 4 formally defined uncertainty states.
8. Provides complete, explainable evidence provenance.
9. Performs constrained response repair to remove contradicted assertions.
10. Re-verifies the repaired response to guarantee factual grounding.

---

## 5. What VisionGuard Does

```
                           IMAGE + VLM RESPONSE
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Structured Claim Extractor   │
                    │ • Atomic Claims + Polarity      │
                    │ • Claim Dependency Graph (DAG)  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │       12-Class Claim Router     │
                    └────────────────┬────────────────┘
                                     │
        ┌─────────────┬──────────────┼──────────────┬─────────────┐
        ▼             ▼              ▼              ▼             ▼
   [Object]      [Attribute]      [Spatial]    [Action/VQA]     [OCR]
   Open-Vocab    Region/CLIP      Geometric    Relational       Text Engine
   Grounding     Matching         Calculus     Reasoning        Extraction
        │             │              │              │             │
        └─────────────┴──────────────┼──────────────┴─────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │       Evidence Provenance       │
                    │  Source, Score, BBox, CoT Log   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Claim-Aware Evidence Fusion   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    4-State Uncertainty Engine   │
                    │ SUPPORTED / CONTRADICTED /      │
                    │ INSUFFICIENT_EVIDENCE /         │
                    │ OUT_OF_SCOPE                    │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Constrained Response Repair   │
                    │  • Strip contradicted claims    │
                    │  • Constrained fact synthesis   │
                    │  • Re-Verification Loop         │
                    └─────────────────────────────────┘
```

---

## 6. Core Architectural Blueprint (CAM-V)

Instead of a monolithic evaluator or a fixed 1,000-class ImageNet classifier, CAM-V operates on **open-vocabulary visual grounding**:

```
OLD SYSTEM (Fixed Classifier):
Image ──▶ Fixed Classifier ──▶ Fixed 1K ImageNet Categories (Fails on arbitrary concepts)

NEW CAM-V SYSTEM (Open-Vocabulary):
Image + Claim Query ──▶ Open-Vocabulary Grounding ──▶ Fine-Grained Visual Evidence BBox
```

This allows VisionGuard to verify arbitrary open-vocabulary concepts (e.g., *"vintage espresso tamper"*, *"purple backpack"*, *"construction worker"*).

---

## 7. Claim Decomposition & Dependency Graph

### Structured Semantic Extraction
Every extracted claim follows a structured schema:
```json
{
  "claim_id": "C1",
  "subject": "boy",
  "predicate": "riding",
  "target_object": "bicycle",
  "claim_type": "action",
  "polarity": "positive",
  "certainty": "asserted",
  "dependencies": ["C0"]
}
```

### Negation Support
Sentences asserting non-existence (*"There are no cars on the road"*) preserve negative polarity:
```json
{
  "subject": "cars",
  "predicate": "exists",
  "claim_type": "negative_claim",
  "polarity": "negative"
}
```
*It is not inverted into "A car exists".*

### Claim Dependency Graph (DAG)
Claims are structured in a dependency DAG:
```
C1: Boy exists (Entity)
├── C2: Wearing shirt (Attribute) ──▶ C3: Red shirt (Color)
└── C4: Riding bicycle (Action)   ──▶ C5: Bicycle exists (Entity)
```
If parent entity $C_1$ is contradicted, failure cascades to dependent child attributes, preventing false grounding attachments.

---

## 8. 12-Class Claim Taxonomy

| ID | Claim Type | Description | Example |
|---|---|---|---|
| 1 | `OBJECT_EXISTENCE` | Primary/secondary entity presence | *"A dog exists."* |
| 2 | `OBJECT_COUNT` | Numerical quantification | *"There are three cars."* |
| 3 | `COLOR` | Color channel and hue distribution | *"The car is red."* |
| 4 | `ATTRIBUTE` | Physical descriptor / material / state | *"The dog is large."* |
| 5 | `ACTION` | Dynamic physical movement / verb | *"The person is running."* |
| 6 | `INTERACTION` | Physical interaction between entities | *"The woman is holding a cup."* |
| 7 | `SPATIAL_RELATION` | Geometric prepositional location | *"The cup is on top of the table."* |
| 8 | `COMPARISON` | Relative sizing / positioning | *"The left car is larger."* |
| 9 | `OCR_TEXT` | Embedded text tokens / signage | *"The sign says STOP."* |
| 10 | `SCENE` | Global environment / background | *"The image shows a beach."* |
| 11 | `NEGATIVE_CLAIM` | Asserted absence of entity / feature | *"There is no bicycle."* |
| 12 | `OUT_OF_SCOPE` | Non-visual external knowledge | *"This was taken in Paris."* |

---

## 9. Open-Vocabulary Evidence Verification & Verifier Router

Claims are routed to specialized domain verifiers:

| Claim Type | Primary Verifier | Secondary Verifier |
|---|---|---|
| `OBJECT_EXISTENCE` | Open-Vocabulary Grounding Verifier | Scene Verifier |
| `OBJECT_COUNT` | Open-Vocabulary Grounding Verifier | Spatial Geometry Verifier |
| `COLOR` / `ATTRIBUTE` | Local Region Attribute Matcher | Open-Vocabulary Grounding |
| `ACTION` / `INTERACTION` | VQA Relational Judge | Spatial Geometry Verifier |
| `SPATIAL_RELATION` | Spatial Geometry Verifier | VQA Relational Judge |
| `COMPARISON` | Spatial Geometry Verifier | Attribute Matcher |
| `OCR_TEXT` | Optical Character Recognition Engine | Attribute Matcher |
| `SCENE` | Global Environment Context Classifier | Attribute Matcher |
| `NEGATIVE_CLAIM` | Absence & Non-Existence Verifier | Open-Vocabulary Grounding |
| `OUT_OF_SCOPE` | External Knowledge / Scope Filter | Scene Verifier |

---

## 10. Evidence Fusion & 4-State Uncertainty Modeling

### 5 Fusion Strategies Implemented:
1. **Single Verifier Baseline**: Evaluates primary verifier only.
2. **Uniform Average Baseline**: $S = \frac{1}{N} \sum_{i=1}^N S_i$.
3. **Category-Weighted Fusion**: Claim-type specific linear weighting.
4. **Calibrated Logistic Regression ML Fusion**: Learned feature vector $[S_{\text{mean}}, S_{\text{max}}, \sigma_{\text{disagreement}}]$.
5. **CAM-V Claim-Aware Fusion**: $P(y \mid \text{claim\_type}, \text{evidence}_1, \dots, \text{disagreement})$.

### 4 Formally Defined Verdict States:
- **`SUPPORTED`** ($\ge 0.70$): Sufficient visual evidence supports the claim.
- **`CONTRADICTED`** ($< 0.40$): Sufficient visual evidence conflicts with the claim.
- **`INSUFFICIENT_EVIDENCE`** ($0.40 \le S < 0.70$): Visual evidence is ambiguous or inconclusive.
- **`OUT_OF_SCOPE`**: Claim cannot reasonably be verified from image features alone without external metadata.

---

## 11. Evidence Provenance

Every final verdict is fully traceable and auditable:
```json
{
  "claim": "The man is holding a red umbrella",
  "final_verdict": "SUPPORTED",
  "confidence": 0.89,
  "evidence": [
    {
      "source": "open_vocab_grounding",
      "result": "umbrella detected",
      "confidence": 0.94,
      "region": [0.12, 0.21, 0.45, 0.60]
    },
    {
      "source": "attribute_verifier",
      "result": "red color alignment",
      "confidence": 0.81
    },
    {
      "source": "action_interaction_vlm",
      "result": "holding interaction confirmed",
      "confidence": 0.78
    }
  ],
  "fusion_math": {
    "formula": "Fused Score = ((0.70 × 0.78) + (0.15 × 0.81) + (0.15 × 0.94)) / 1.00 = 0.809"
  }
}
```

---

## 12. Constrained Response Repair & Re-Verification

VisionGuard repairs hallucinated responses through constrained factual synthesis:
```
Original Response ──▶ Claim Graph ──▶ Identify Contradictions ──▶ Strip Contradictions
                                                                           │
                                                                           ▼
Re-Verified Output ◀── Re-Verification Loop ◀── Constrained Synthesis ◀── Verified Facts
```

*Example:*
- **Original:** *"A woman wearing a red jacket is riding a bicycle near the Eiffel Tower."*
- **Verified Facts:** $\checkmark$ Woman exists, $\checkmark$ Wearing red jacket, $\checkmark$ Riding bicycle.
- **Removed Claims:** $\times$ Eiffel Tower is visible (Out of Scope / Contradicted).
- **Repaired Text:** *"A woman wearing a red jacket is riding a bicycle."*
- **Re-Verification Verdict:** `SUPPORTED` (Confidence: 91.2%).

---

## 13. Adversarial Hallucination Generation

VisionGuard includes a systematic perturbation engine generating 7 controlled adversarial variants:
1. **Object Swap**: *"A man is riding a bicycle"* $\rightarrow$ *"A man is riding a boat"*.
2. **Attribute Swap**: *"The car is red"* $\rightarrow$ *"The car is bright blue"*.
3. **Count Perturbation**: *"A dog is sitting"* $\rightarrow$ *"Three dogs are sitting"*.
4. **Relation Swap**: *"The bicycle is near the park"* $\rightarrow$ *"The bicycle is flying above the park"*.
5. **Negation Inversion**: *"The boy is riding a bicycle"* $\rightarrow$ *"The boy is not riding a bicycle"*.
6. **Entity Swap**: *"The boy is riding"* $\rightarrow$ *"The elderly astronaut is riding"*.
7. **OCR Perturbation**: *"The sign says SLOW"* $\rightarrow$ *"The sign says STOP"*.

---

## 14. Evaluation Methodology & 6 Core Experiments

VisionGuard provides 6 formal evaluation experiment suites:

| Experiment | Purpose | Core Metrics |
|---|---|---|
| **Exp 1: Claim Classification** | Evaluates fine-grained claim classification | Precision, Recall, F1, Macro F1, AUROC, Per-Category F1 |
| **Exp 2: Baseline Comparison** | Compares CAM-V vs. 4 baseline systems | F1 Score, AUROC across 5 architectures |
| **Exp 3: Ablation Study** | Section 14 ablation (A0 to A5 configurations) | $\Delta \text{F1}\%$, component contribution |
| **Exp 4: Robustness Evaluation** | Tests stability under paraphrasing & perturbations | Score variance, order invariance, negation sensitivity |
| **Exp 5: Human Alignment** | Evaluates agreement with human expert annotations | Cohen's Kappa ($\kappa$), Pearson correlation ($r$) |
| **Exp 6: Guardrail & Repair** | Measures before/after hallucination mitigation | $\text{Hal}_{\text{main}}$, $\#\text{Hal}_{\text{extra}}$, $\text{Hal}_{\text{overall}}$, Repair Success Rate |

---

## 15. HQM Psychometric Reliability & Validity

Implements psychometric indicators from *Yan et al. (2024)*:
- **Test-Retest Reliability ($r > 0.95$)**: Score consistency across repeated evaluations.
- **Parallel-Forms Reliability ($\rho > 0.90$)**: Robustness under paraphrased query perturbations.
- **Content Validity ($N_{valid}/N > 95\%$)**: Verification of unambiguous question-image pairs.
- **Criterion Validity ($r_{human} > 0.90$)**: Correlation with human ground-truth evaluations.

---

## 16. Repository Structure

```
VGuard/
├── 2406.17115v3.pdf                    # Research paper (HQM/HQH benchmark methodology)
├── VisionGuard_Detailed_Project_Plan.md # Technical specification document
├── README.md                           # Master project blueprint
├── requirements.txt                    # Project dependencies
├── run.py                              # Unified CLI runner
├── tests/                              # Unit & integration test suite (18 tests)
│   ├── test_api.py                     # API route & multipart upload tests
│   ├── test_claim_extractor.py         # 12-class extraction & negation tests
│   ├── test_evidence_fusion.py         # 5 fusion strategies & math tests
│   ├── test_hqm_eval.py                # HQM psychometric reliability tests
│   ├── test_repair_and_adversarial.py  # Response repair & adversarial generator tests
│   └── test_verifiers.py               # Specialized domain verifier tests
└── visionguard/                        # Core Python package
    ├── schemas.py                      # Pydantic models (12 ClaimTypes, 4 Verdicts, Provenance)
    ├── config.py                       # Global thresholds and taxonomy settings
    ├── data_loader.py                  # VisionGuard-Claims benchmark dataset
    ├── cli.py                          # CLI implementation
    ├── api/
    │   └── main.py                     # FastAPI REST server & 10 endpoints
    ├── pipeline/
    │   ├── claim_extractor.py          # Structured proposition extractor with polarity
    │   ├── dependency_graph.py         # Claim Dependency Graph (DAG)
    │   ├── claim_classifier.py         # 12-class taxonomy classifier
    │   ├── evidence_router.py          # Specialized verifier router
    │   ├── evidence_fusion.py          # Uniform, Weighted, Logistic & Claim-Aware fusion
    │   ├── uncertainty_engine.py       # 4-state uncertainty engine
    │   ├── image_processor.py          # Automated raw image processor
    │   └── orchestrator.py             # CAM-V pipeline coordinator
    ├── verifiers/
    │   ├── object_verifier.py          # Open-vocabulary grounding verifier
    │   ├── attribute_verifier.py       # Region attribute & color matcher
    │   ├── spatial_verifier.py         # Geometric spatial calculus verifier
    │   ├── action_verifier.py          # Semantic action / VQA relational verifier
    │   ├── ocr_verifier.py             # Optical Character Recognition verifier
    │   ├── absence_verifier.py         # Negative claim / absence verifier
    │   └── scene_verifier.py           # Scene context & Out-of-Scope filter
    ├── repair/
    │   └── repair_engine.py            # Constrained response repair & re-verification
    ├── baselines/
    │   ├── whole_response_judge.py     # Monolithic VLM judge baseline
    │   └── similarity_baseline.py      # Global embedding similarity baseline
    ├── eval/
    │   ├── hqm_evaluator.py            # HQM psychometric evaluation suite
    │   ├── benchmark_runner.py         # Comparative benchmark runner (CAM-V vs 4 baselines)
    │   ├── ablation.py                 # Section 14 ablation runner (A0 to A5)
    │   ├── adversarial.py              # 7-operator adversarial hallucination generator
    │   ├── experiment_claim_classification.py # Experiment 1 runner
    │   ├── experiment_robustness.py    # Experiment 4 runner
    │   ├── experiment_human_alignment.py # Experiment 5 runner
    │   └── experiment_guardrail.py     # Experiment 6 runner
    └── static/                         # Web Dashboard UI
        ├── index.html                  # Executive dark dashboard
        ├── styles.css                  # Zinc/Charcoal theme (#09090b)
        └── app.js                      # Interactivity, Chart.js charts & CoT accordions
```

---

## 17. RESTful API Specification

The FastAPI server provides 10 endpoints:

| Method | Endpoint | Description | Request Payload / Params |
|---|---|---|---|
| `GET` | `/api/health` | Service health and version check | None |
| `POST` | `/api/analyze` | Full CAM-V claim verification & repair | `{"response_text": str, "visual_features": dict}` |
| `POST` | `/api/upload_and_analyze` | Multipart raw image upload & verification | `file` (binary), `response_text` (optional) |
| `POST` | `/api/repair` | Constrained response repair & re-verification | `{"response_text": str, "visual_features": dict}` |
| `POST` | `/api/adversarial/generate` | Generates 7 adversarial hallucination variants | `{"claims": [str]}` |
| `GET` | `/api/experiments/run` | Runs all 6 core evaluation experiments | None |
| `GET` | `/api/leaderboard` | 8-dimension model evaluation leaderboard | None |
| `GET` | `/api/hqm` | HQM benchmark quality measurement table | None |
| `GET` | `/api/benchmark/run` | Comparative evaluation (CAM-V vs 4 baselines) | None |
| `GET` | `/api/ablation/run` | 6-configuration ablation study (A0 to A5) | None |
| `POST` | `/api/guardrail/apply` | Live prompt decoration & response repair | `{"prompt": str, "response_text": str}` |

---

## 18. CLI Reference

```bash
# 1. Start FastAPI server & Web Dashboard on port 8000
python3 run.py server --port 8000

# 2. Analyze a text response
python3 run.py analyze "A young boy wearing a red shirt is riding a bicycle near a park."

# 3. Perform constrained response repair
python3 run.py repair "A brown dog is sitting beside a red bicycle in Paris. The dog is an astronaut."

# 4. Generate adversarial hallucination test cases
python3 run.py adversarial "A man is riding a bicycle near the river."

# 5. Run comparative benchmark evaluation across baselines
python3 run.py eval

# 6. Run Section 14 ablation experiments (A0 to A5)
python3 run.py ablation

# 7. Run all 6 evaluation experiments
python3 run.py experiments
```

---

## 19. What Was Removed, Replaced, and Added

### What Was Removed / Replaced:
- ❌ **Removed**: Fixed-category ImageNet classification as the sole verification backbone.
- ❌ **Removed**: Simple heuristic color extraction and bounding-box overlap as proof of semantic physical interactions.
- ❌ **Removed**: Fixed mock metrics; separated in-memory demonstration datasets from experimental evaluation pipelines.
- ❌ **Removed**: Claiming "HQH (Ours)" (properly attributed to Yan et al., 2024).

### What Was Added:
-  **Open-Vocabulary Visual Grounding**: Arbitrary concept detection and bounding box localization.
-  **12-Class Claim Taxonomy**: Covers existence, count, color, attribute, action, interaction, spatial, comparison, OCR, scene, negation, and out-of-scope.
-  **Structured Semantic Extraction with Negation & Polarity**: Explicit polarity parsing with Claim Dependency Graph (DAG) construction.
-  **Specialized Verifier Router**: 7 specialized domain verifiers (Grounding, Attribute, Spatial, Action/VQA, OCR, Absence, Scene).
-  **4 Formally Defined Decision States**: `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `OUT_OF_SCOPE`.
-  **Traceable Evidence Provenance**: Auditable verifier logs, bounding boxes, and explicit `FusionMath` formulas.
-  **Constrained Response Repair**: Factually grounded response reconstruction with re-verification loop.
-  **Adversarial Hallucination Generation**: 7 systematic perturbation operators.
-  **6 Comprehensive Evaluation Experiments**: Baseline comparisons, Section 14 ablations, robustness, human alignment ($\kappa$), guardrail mitigation, and HQM reliability.

---

## 20. Final Research Contribution

> **VisionGuard introduces a claim-aware multi-evidence verification framework for Vision-Language Model outputs. Instead of evaluating an entire response with a single evaluator, the system decomposes responses into atomic factual claims and routes each claim to specialized visual verification mechanisms. Evidence is combined using uncertainty-aware claim-specific fusion, enabling hallucination detection, evidence provenance, and constrained response repair. The framework is evaluated using claim-level classification, adversarial hallucinations, baseline comparison, ablation studies, human agreement, and HQM-inspired reliability and validity analysis.**

### The Core Hypothesis:
$$\boxed{\text{Claim-aware specialized verification} > \text{Monolithic response-level verification}}$$

---

## 21. References & Citation

1. **Bei Yan et al.**, *"Measuring the Measurers: Quality Evaluation of Hallucination Benchmarks for Large Vision-Language Models"*, arXiv:2406.17115, 2024.
2. **Yiyang Zhou et al.**, *"Analyzing and Mitigating Object Hallucination in Large Vision-Language Models"*, Findings of ACL 2024.
3. **Fuxiao Liu et al.**, *"Aligning Large Multimodal Models with Factually Augmented RLHF"*, NeurIPS 2023.

---

## 22. License

This project is licensed under the **MIT License**.
