# 🛡️ VGuard

> **Claim-Aware Multi-Evidence Hallucination Detection, Response Repair & Psychometric Quality Evaluation for Large Vision-Language Models (LVLMs)**



> [!NOTE]
> **Active Research & Development**: VGuard is under active development with continuous updates, architectural refinements, and feature additions. APIs, verifier modules, and evaluation benchmarks are being actively iterated upon.

---

## 📌 Executive Summary

**VGuard** (Claim-Aware Multi-Evidence Verification / CAM-V) is an open-source, open-vocabulary framework designed to detect, explain, and mitigate visual hallucinations in Large Vision-Language Models (LVLMs).

Unlike conventional evaluation approaches that treat generated multimodal text as monolithic blocks, VGuard decomposes responses into an **atomic Claim Dependency Graph (DAG)**. Each claim is classified according to a **12-Class Fine-Grained Taxonomy** and routed to **specialized visual domain verifiers** (Open-Vocabulary Grounding, Spatial Calculus, Attribute/Color Matchers, Relational Action Judges, OCR, Absence Filters, and Scene Context). Evidence is aggregated via **4-State Formal Uncertainty Modeling**, producing auditable evidence provenance and enabling **closed-loop constrained response repair with re-verification**.

```
   ┌───────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
   │    VLM Response &     │ ───▶ │ Claim Decomposition DAG │ ───▶ │ 12-Class Claim Router   │
   │      Input Image      │      │  (Atomic Propositions)  │      │  (Specialized Routing)  │
   └───────────────────────┘      └─────────────────────────┘      └────────────┬────────────┘
                                                                                │
   ┌───────────────────────┐      ┌─────────────────────────┐                   ▼
   │  Repaired Grounded    │ ◀─── │ Constrained Synthesis & │ ◀─── ┌─────────────────────────┐
   │  Output (Re-Verified) │      │ 4-State Uncertainty     │      │ Multi-Evidence Verifier │
   │                       │      │ (SUPPORTED/CONTRADICTED)│      │ Ensemble (7 Verifiers)  │
   └───────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 📑 Table of Contents

1. [Key Capabilities](#-key-capabilities)
2. [The Multimodal Hallucination Problem](#-the-multimodal-hallucination-problem)
3. [System Architecture & CAM-V Blueprint](#-system-architecture--cam-v-blueprint)
4. [12-Class Claim Taxonomy](#-12-class-claim-taxonomy)
5. [Specialized Verifier Ensemble & Routing Matrix](#-specialized-verifier-ensemble--routing-matrix)
6. [Claim Dependency Graph (DAG) & Cascading Falsification](#-claim-dependency-graph-dag--cascading-falsification)
7. [Evidence Fusion & 4-State Uncertainty Engine](#-evidence-fusion--4-state-uncertainty-engine)
8. [Auditable Evidence Provenance](#-auditable-evidence-provenance)
9. [Constrained Response Repair & Re-Verification Loop](#-constrained-response-repair--re-verification-loop)
10. [Adversarial Hallucination Generator (7 Operators)](#-adversarial-hallucination-generator-7-operators)
11. [Scientific Evaluation Suite (6 Core Experiments)](#-scientific-evaluation-suite-6-core-experiments)
12. [HQM Psychometric Reliability & Validity](#-hqm-psychometric-reliability--validity)
13. [Installation & Setup](#-installation--setup)
14. [Quick Start & Usage](#-quick-start--usage)
15. [CLI Reference](#-cli-reference)
16. [Python SDK & Programmatic API](#-python-sdk--programmatic-api)
17. [RESTful API Specification](#-restful-api-specification)
18. [Interactive Web Dashboard](#-interactive-web-dashboard)
19. [Repository Structure](#-repository-structure)
20. [References](#-references)
21. [License](#-license)

---

## 🌟 Key Capabilities

- 🔍 **Open-Vocabulary Visual Grounding**: Unbounded semantic concept verification via YOLO-World v2 and zero-shot visual backends without ImageNet/fixed-class restrictions.
- 🌳 **Hierarchical Claim Dependency Graph (DAG)**: Models prerequisite entity-attribute-action dependencies; cascading falsification prevents false grounding attachments when root entities are absent.
- 🎯 **12-Class Fine-Grained Claim Taxonomy**: Explicitly isolates object existence, counts, colors, physical attributes, dynamic actions, inter-entity interactions, spatial relations, comparative scales, OCR text, scene context, negative assertions, and out-of-scope knowledge.
- ⚖️ **Directional Evidence & 4-State Uncertainty Engine**: Measures independent support, contradiction, and uncertainty dimensions to produce calibrated verdicts: `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, and `OUT_OF_SCOPE`.
- 🛠️ **Closed-Loop Constrained Response Repair**: Automatically isolates and prunes contradicted assertions, reconstructs grammatically coherent grounded text, and validates factual consistency via an immediate re-verification pass.
- ⚡ **7-Operator Adversarial Hallucination Engine**: Systematically synthesizes controlled adversarial perturbations (Object Swap, Attribute Swap, Count Mismatch, Spatial Relation Inversion, Negation Swap, Entity Swap, OCR Distortion) for robustness stress testing.
- 📊 **HQM Psychometric Evaluation & 6 Benchmark Suites**: Evaluated against test-retest reliability ($r > 0.95$), parallel-forms reliability ($\rho > 0.90$), content validity, criterion validity, baseline comparisons against 4 architectures, and Section 14 ablations (A0–A5).
- 🚀 **Production-Ready Stack**: Includes a high-throughput FastAPI REST backend (10 endpoints), an executive dark-mode Web Dashboard UI, and a unified CLI.

---

## ⚠️ The Multimodal Hallucination Problem

Large Vision-Language Models (LVLMs) frequently generate descriptions that are fluent and linguistically persuasive, yet factually inconsistent with visual inputs.

Consider the following response generated for an image:
> *"A young woman wearing a red jacket is riding a bicycle near the Eiffel Tower while two dogs play on the grass."*

A traditional **monolithic evaluator** (e.g., GPT-4V-as-a-Judge or CLIPScore) evaluates this response as a single monolithic block:

```
Traditional Monolithic Approach:
Image + Full Text ──▶ Monolithic Judge ──▶ "Score: 0.65" (Opaque, Uncalibrated, Unactionable)
```

### Why Monolithic Evaluation Fails:
1. **Conflated Verification Modalities**: The model must simultaneously reason about object presence (woman, bicycle, dogs), attributes (young, red), dynamic actions (riding, playing), counts (two dogs), spatial relations (near, on), and world knowledge (Eiffel Tower).
2. **Hallucination Blind Spots**: A monolithic judge may rate a sentence as "plausible" because $80\%$ of the statement is true, completely ignoring the hallucination of an out-of-scope landmark or incorrect count.
3. **Lack of Explainability & Actionability**: Monolithic scores do not provide bounding box coordinates, visual evidence logs, or mechanisms to fix the hallucination.

### The VGuard Solution:
VGuard breaks down the complex response into atomic, verifiable propositions ($C_1 \dots C_8$), constructs a dependency graph, and verifies each proposition independently using specialized visual evidence tools.

---

## 📐 System Architecture & CAM-V Blueprint

VGuard operates through a coordinated 6-stage pipeline:

```
                                  IMAGE + VLM RESPONSE
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │   Structured Claim Extractor    │
                           │ • Atomic Propositions + Polarity│
                           │ • Claim Dependency Graph (DAG)  │
                           └────────────────┬────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │      12-Class Claim Router      │
                           └────────────────┬────────────────┘
                                            │
               ┌─────────────┬──────────────┼──────────────┬─────────────┐
               ▼             ▼              ▼              ▼             ▼
          [Grounding]   [Attribute]     [Spatial]     [Action/VQA]     [OCR]
          YOLO-World    Region/CLIP     Geometric     Relational       Tesseract/
          Open-Vocab    Matcher         Calculus      Judge            Vision OCR
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
                           │  • Category-Weighted / ML Model │
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
                           │  • Prune contradicted claims    │
                           │  • Reconstruct grounded facts   │
                           │  • Re-Verification Safety Loop  │
                           └─────────────────────────────────┘
```

---

## 🏷️ 12-Class Claim Taxonomy

VGuard categorizes every extracted proposition into a structured 12-class taxonomy:

| ID | Claim Type | Description | Example Proposition |
|---|---|---|---|
| 1 | `OBJECT_EXISTENCE` | Primary or secondary physical entity presence | *"A bicycle exists in the foreground."* |
| 2 | `OBJECT_COUNT` | Numerical quantification of discrete entities | *"There are three chairs at the table."* |
| 3 | `COLOR` | Color channel and hue distribution of an entity | *"The car is bright red."* |
| 4 | `ATTRIBUTE` | Physical descriptor, material, shape, or state | *"The dog is large and fluffy."* |
| 5 | `ACTION` | Dynamic physical movement or activity | *"The person is running along the path."* |
| 6 | `INTERACTION` | Physical interaction between two or more entities | *"The child is holding an ice cream cone."* |
| 7 | `SPATIAL_RELATION` | Geometric and topological prepositional location | *"The laptop is on top of the desk."* |
| 8 | `COMPARISON` | Relative sizing, positioning, or property scaling | *"The red car is larger than the blue truck."* |
| 9 | `OCR_TEXT` | Embedded text tokens, signage, or typography | *"The road sign says STOP."* |
| 10 | `SCENE` | Global environment, weather, and setting context | *"The scene is an outdoor mountain landscape."* |
| 11 | `NEGATIVE_CLAIM` | Asserted absence of an entity or attribute | *"There are no pedestrians on the sidewalk."* |
| 12 | `OUT_OF_SCOPE` | Non-visual, historical, or external knowledge | *"This event took place during 1945 in Berlin."* |

---

## 🔀 Specialized Verifier Ensemble & Routing Matrix

Extracted claims are dispatched to specialized domain verifiers:

| Claim Type | Primary Verifier | Secondary Verifier | Verification Mechanism |
|---|---|---|---|
| `OBJECT_EXISTENCE` | Open-Vocabulary Grounding | Scene Verifier | Zero-shot YOLO-World / Open-Vocab BBox grounding |
| `OBJECT_COUNT` | Count Verifier | Spatial Geometry | Bounding box clustering + Non-Maximum Suppression (NMS) |
| `COLOR` / `ATTRIBUTE` | Region Attribute Matcher | Open-Vocabulary Grounding | Cropped region color histogram & feature embeddings |
| `ACTION` / `INTERACTION` | Relational Action Judge | Spatial Geometry | Spatial overlap calculus + semantic relational verification |
| `SPATIAL_RELATION` | Spatial Geometry Verifier | Relational Judge | 2D/3D coordinate calculus (IoU, centroid deltas, containment) |
| `COMPARISON` | Spatial Geometry Verifier | Attribute Matcher | Relative bounding box area and dimension comparative calculus |
| `OCR_TEXT` | OCR Text Engine | Attribute Matcher | Optical Character Recognition + Levenshtein fuzzy string distance |
| `SCENE` | Global Scene Classifier | Attribute Matcher | Whole-image scene embeddings & environmental classification |
| `NEGATIVE_CLAIM` | Absence Verifier | Open-Vocabulary Grounding | Inverse threshold verification across high-confidence regions |
| `OUT_OF_SCOPE` | Scope & External Filter | Scene Verifier | Multimodal verifiability gating & knowledge boundary check |

---

## 🌲 Claim Dependency Graph (DAG) & Cascading Falsification

VGuard constructs a Directed Acyclic Graph (DAG) representing the linguistic dependencies of the response.

```
C1: "A woman exists" (Entity Root)
├── C2: "Wearing a jacket" (Attribute) ──▶ C3: "The jacket is red" (Color)
└── C4: "Riding a bicycle" (Action)    ──▶ C5: "A bicycle exists" (Entity Root)
```

### Cascading Falsification Principle:
If root entity $C_1$ (*"A woman exists"*) is determined to be **`CONTRADICTED`** (i.e., no person is in the scene), VGuard automatically propagates the contradiction down to dependent child claims ($C_2, C_3, C_4$). This prevents false grounding attachments where a background object is mistakenly attributed to a non-existent subject.

---

## ⚖️ Evidence Fusion & 4-State Uncertainty Engine

VGuard computes three directional scores for every claim:
- **Support Score ($S$)** $\in [0, 1]$
- **Contradiction Score ($C$)** $\in [0, 1]$
- **Uncertainty Score ($U$)** $\in [0, 1]$

### 5 Fusion Modes Implemented:
1. **Single Verifier Baseline**: Utilizes primary verifier evidence directly.
2. **Uniform Average Baseline**: $S_{\text{fused}} = \frac{1}{N} \sum_{i=1}^N S_i$.
3. **Category-Weighted Fusion**: Claim-type specific linear weighting matrix.
4. **Calibrated ML Fusion**: Logistic regression trained on feature vector $[S_{\text{mean}}, S_{\text{max}}, C_{\text{max}}, \sigma_{\text{disagreement}}]$.
5. **Claim-Aware CAM-V Fusion**: $P(y \mid \text{claim\_type}, \text{evidence}_1 \dots \text{evidence}_k, \Delta_{\text{conflict}})$.

### Formal 4-State Verdict Criteria:
```
┌───────────────────────────┬────────────────────────────────────────────────────────┐
│ State                     │ Formal Decision Condition                              │
├───────────────────────────┼────────────────────────────────────────────────────────┤
│ SUPPORTED                 │ S_fused ≥ 0.70 AND C_fused < 0.40                      │
│ CONTRADICTED              │ C_fused ≥ 0.50 OR S_fused < 0.35                       │
│ INSUFFICIENT_EVIDENCE     │ 0.35 ≤ S_fused < 0.70 AND C_fused < 0.50               │
│ OUT_OF_SCOPE              │ Claim requires external world metadata / non-visual    │
└───────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 🔍 Auditable Evidence Provenance

Every decision includes full mathematical traceability and bounding box provenance:

```json
{
  "claim_id": "C3",
  "claim_text": "The woman is wearing a red jacket",
  "claim_type": "color",
  "final_verdict": "SUPPORTED",
  "confidence": 0.892,
  "support_score": 0.892,
  "contradiction_score": 0.045,
  "uncertainty_score": 0.063,
  "evidence": [
    {
      "source": "open_vocab_grounding",
      "result": "jacket detected",
      "confidence": 0.94,
      "bounding_box": [0.15, 0.22, 0.48, 0.65]
    },
    {
      "source": "attribute_verifier",
      "result": "red color alignment confirmed",
      "confidence": 0.86,
      "details": {"dominant_hue": "red", "saturation": 0.82}
    }
  ],
  "fusion_math": {
    "formula": "Fused = ((0.65 × 0.86) + (0.35 × 0.94)) / 1.00 = 0.888",
    "weights_used": {"attribute_verifier": 0.65, "open_vocab_grounding": 0.35}
  }
}
```

---

## 🛠️ Constrained Response Repair & Re-Verification Loop

When hallucinations are detected, VGuard repairs the response:

```
Original Response ──▶ Claim Graph ──▶ Identify Contradictions ──▶ Strip Contradictions
                                                                            │
                                                                            ▼
Re-Verified Output ◀── Re-Verification Pass ◀── Constrained Synthesis ◀── Verified Facts
```

### Real-World Example:
- **Original Response**: *"A brown dog is sitting beside a red bicycle. The dog is an astronaut wearing a space helmet."*
- **Verified Claims**:
  - $\checkmark$ `C1`: Dog exists (`SUPPORTED`, $94.2\%$)
  - $\checkmark$ `C2`: Dog is brown (`SUPPORTED`, $89.0\%$)
  - $\checkmark$ `C3`: Bicycle exists (`SUPPORTED`, $91.5\%$)
  - $\checkmark$ `C4`: Bicycle is red (`SUPPORTED`, $88.2\%$)
- **Contradicted Claims**:
  - $\times$ `C5`: Dog is an astronaut (`CONTRADICTED`, $96.1\%$)
  - $\times$ `C6`: Space helmet present (`CONTRADICTED`, $93.4\%$)
- **Repaired Response**: *"A brown dog is sitting beside a red bicycle."*
- **Re-Verification Verdict**: `SUPPORTED` (Overall Confidence: $92.4\%$).

---

## ⚡ Adversarial Hallucination Generator (7 Operators)

VGuard includes a systematic perturbation engine for automated stress testing:

1. **Object Swap**: Substitutes detected objects with plausible distractors (*"person on a bicycle"* $\rightarrow$ *"person on a motorcycle"*).
2. **Attribute Swap**: Modifies colors, sizes, or materials (*"red jacket"* $\rightarrow$ *"neon green jacket"*).
3. **Count Perturbation**: Alters numerical quantifiers (*"two birds"* $\rightarrow$ *"five birds"*).
4. **Spatial Relation Inversion**: Inverts prepositional relationships (*"on top of the desk"* $\rightarrow$ *"underneath the desk"*).
5. **Negation Inversion**: Flips polarity of assertions (*"no cars are visible"* $\rightarrow$ *"cars are visible"*).
6. **Entity Swap**: Replaces subject entities with rare or impossible categories (*"a young boy is running"* $\rightarrow$ *"an astronaut is running"*).
7. **OCR Perturbation**: Modifies embedded text characters (*"STOP sign"* $\rightarrow$ *"SLOW sign"*).

---

## 📊 Scientific Evaluation Suite (6 Core Experiments)

VGuard provides 6 automated, reproducible experiment runners:

| Experiment | Focus Area | Evaluated Metrics |
|---|---|---|
| **Exp 1: Claim Classification** | Granular claim-level accuracy across 12 types | Precision, Recall, Macro F1, AUROC, Per-Category F1 |
| **Exp 2: Baseline Comparison** | CAM-V vs. 4 baseline architectures | F1 Score, AUROC, Brier Calibration Score |
| **Exp 3: Section 14 Ablation** | Component contribution (A0 to A5 setups) | $\Delta \text{F1}\%$, Grounding Gain, DAG Propagation Gain |
| **Exp 4: Robustness Evaluation** | Stability under linguistic perturbations | Score Variance, Order Invariance, Negation Sensitivity |
| **Exp 5: Human Alignment** | Agreement with expert human annotations | Cohen's Kappa ($\kappa$), Pearson Correlation ($r$) |
| **Exp 6: Guardrail Effectiveness** | Before/after hallucination mitigation | $\text{Hal}_{\text{main}}$, $\#\text{Hal}_{\text{extra}}$, Latency (ms) |

---

## 🧠 HQM Psychometric Reliability & Validity

VGuard integrates the Hallucination Quality Measurement (HQM) framework (*Yan et al., arXiv:2406.17115v3*):

- **Test-Retest Reliability ($r > 0.95$)**: Measures evaluation consistency across repeated inference passes.
- **Parallel-Forms Reliability ($\rho > 0.90$)**: Evaluates score stability under semantic paraphrasing.
- **Content Validity ($N_{\text{valid}} / N > 95\%$)**: Guarantees unambiguous image-claim correspondence.
- **Criterion Validity ($r_{\text{human}} > 0.90$)**: Empirically validates correlation against human expert judgments.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.1+
- Tesseract OCR (optional, for OCR claim verification)
- macOS / Linux / Windows (WSL2 recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/DhruvR-16/VGuard.git
cd VGuard
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: Verify Tesseract OCR Installation
```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr
```

### 5. Environment Configuration
Copy the sample environment file:
```bash
cp .env.example .env
```

---

## 🚀 Quick Start & Usage

### 1. Launch the FastAPI Server & Web Dashboard
```bash
python3 run.py server --port 8000
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 2. Verify a Response via CLI
```bash
python3 run.py analyze "A young boy wearing a red shirt is riding a bicycle." --image data/images/000001.jpg
```

### 3. Repair a Hallucinated Caption
```bash
python3 run.py repair "A brown dog is sitting beside a red bicycle. The dog is an astronaut." --image data/images/000002.jpg
```

### 4. Run the Evaluation Experiment Suite
```bash
# Run comparative baseline benchmarks
python3 run.py eval --experiment benchmark

# Run Section 14 ablation studies
python3 run.py eval --experiment ablation

# Run robustness & order invariance tests
python3 run.py eval --experiment robustness

# Run all 6 evaluation experiments
python3 run.py eval --experiment all
```

---

## ⌨️ CLI Reference

The `run.py` script provides a unified CLI for all operations:

```bash
usage: run.py [-h] {server,analyze,repair,eval,ablation,experiments,adversarial} ...

Available commands:
  server         Launch FastAPI REST server and interactive Web Dashboard
  analyze        Analyze and verify factual claims in a VLM response
  repair         Perform constrained response repair on a hallucinated response
  eval           Execute benchmark or scientific evaluation experiments
  ablation       Run Section 14 ablation experiment suite (A0 to A5)
  experiments    Run all 6 scientific evaluation experiments
  adversarial    Generate 7-operator adversarial hallucination test suite
```

### Command Flags:
- `server`: `--host` (default: 127.0.0.1), `--port` (default: 8000), `--reload`
- `analyze`: `text` (required), `--image` (path), `--mock` (run with mock backends)
- `repair`: `text` (required), `--image` (path), `--mock`
- `eval`: `--experiment {benchmark,ablation,robustness,human_alignment,guardrail,claim_classification,all}`, `--split {train,val,test}`, `--mock`
- `adversarial`: `claim` (required base factual proposition)

---

## 🐍 Python SDK & Programmatic API

You can use VGuard directly within your Python applications:

```python
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.schemas import Verdict

# 1. Initialize the VGuard CAM-V Orchestrator
orchestrator = CAMVOrchestrator(use_mock=False)

# 2. Analyze a VLM generated response against an image
response_text = "A young boy wearing a red shirt is riding a bicycle near a park."
image_path = "data/images/000001.jpg"

result = orchestrator.analyze(
    response_text=response_text,
    image_input=image_path,
    enable_repair=True
)

# 3. Inspect Overall Verdict & Confidence
print(f"Overall Verdict: {result.overall_verdict.value}")
print(f"Confidence: {result.overall_confidence:.1%}")
print(f"Hallucination Probability: {result.hallucination_probability:.1%}")

# 4. Iterate over Atomic Claims
for item in result.claims:
    claim = item.claim
    print(f"\nClaim [{claim.claim_id}] ({claim.type.value}): '{claim.text}'")
    print(f"  Verdict: {item.verdict.value} (Fused Score: {item.fused_score:.3f})")
    if item.evidence:
        for ev in item.evidence:
            print(f"    - {ev.source}: {ev.result} (score: {ev.support_score:.2f})")

# 5. Retrieve Constrained Repaired Response
if result.repaired_response:
    print(f"\nRepaired Response: '{result.repaired_response.repaired_text}'")
    print(f"Removed Claims: {result.repaired_response.removed_claims}")
```

---

## 📡 RESTful API Specification

The FastAPI server provides 10 REST endpoints:

| Method | Endpoint | Description | Request Format |
|---|---|---|---|
| `GET` | `/api/health` | Service health, status, and version | None |
| `POST` | `/api/analyze` | Full CAM-V claim verification & repair | JSON: `{"response_text": str, "visual_features": dict}` |
| `POST` | `/api/upload_and_analyze` | Multipart raw image upload & verification | Multipart: `file` (binary), `response_text` (str) |
| `POST` | `/api/repair` | Constrained response repair & re-verification | JSON: `{"response_text": str, "visual_features": dict}` |
| `POST` | `/api/adversarial/generate` | Generates 7 adversarial variants per claim | JSON: `{"claims": [str]}` |
| `GET` | `/api/experiments/run` | Executes all 6 scientific evaluation experiments | None |
| `GET` | `/api/leaderboard` | 8-dimension model evaluation leaderboard | None |
| `GET` | `/api/hqm` | HQM benchmark psychometric quality table | None |
| `GET` | `/api/benchmark/run` | Comparative evaluation (CAM-V vs 4 baselines)| None |
| `GET` | `/api/ablation/run` | 6-configuration ablation study (A0 to A5) | None |
| `POST` | `/api/guardrail/apply` | Live prompt decoration & guardrail repair | JSON: `{"prompt": str, "response_text": str}` |

### Example cURL Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "response_text": "A woman wearing a red jacket is riding a bicycle.",
       "visual_features": {
         "objects": {"woman": 0.95, "bicycle": 0.92, "jacket": 0.90},
         "color_distribution": {"red": 0.40}
       }
     }'
```

---

## 🖥️ Interactive Web Dashboard

VGuard includes a dark-mode Web Dashboard featuring:

- **CAM-V Playground**: Drag-and-drop image upload, real-time claim decomposition, bounding box visualizer, and CoT reasoning logs.
- **HQM Framework**: Psychometric reliability and validity measurement tables.
- **8D Leaderboard**: Multimodal model rankings across 8 perceptual dimensions.
- **Baselines & Ablation**: Interactive Chart.js comparative performance visualizations.
- **Live Guardrail**: Real-time response repair simulation with latency tracking.

To access the dashboard, run `python3 run.py server` and open `http://127.0.0.1:8000`.

---

## 📁 Repository Structure

```
VGuard/
├── README.md                           # Master documentation & project specification
├── requirements.txt                    # Python package dependencies
├── run.py                              # Unified CLI entry point
├── .env.example                        # Example environment configuration
├── 2406.17115v3.pdf                    # Research paper (HQM/HQH methodology foundation)
├── data/                               # Datasets & benchmark annotations
│   ├── annotations/                    # Ground truth annotations
│   ├── images/                         # Sample evaluation images
│   └── visionguard_claims_v0.1.jsonl   # Claim-level benchmark dataset
├── tests/                              # Unit & integration test suite
│   ├── test_api.py                     # FastAPI endpoint tests
│   ├── test_claim_extractor.py         # 12-class extraction & DAG tests
│   ├── test_count_verifier.py          # Object count & NMS clustering tests
│   ├── test_evidence_fusion.py         # Directional fusion & uncertainty tests
│   ├── test_hqm_eval.py                # HQM psychometric reliability tests
│   ├── test_integration_real_images.py # End-to-end real image inference tests
│   ├── test_repair_and_adversarial.py  # Response repair & perturbation tests
│   └── test_verifiers.py               # Specialized domain verifier tests
└── visionguard/                        # Core Python package
    ├── schemas.py                      # Pydantic schemas (12 ClaimTypes, 4 Verdicts, Evidence)
    ├── config.py                       # Global thresholds and configuration
    ├── cli.py                          # CLI implementation
    ├── data_loader.py                  # Benchmark data loader & fixtures
    ├── api/
    │   └── main.py                     # FastAPI REST application
    ├── pipeline/
    │   ├── claim_extractor.py          # Structured claim & polarity parser
    │   ├── dependency_graph.py         # Claim Dependency Graph (DAG) engine
    │   ├── claim_classifier.py         # 12-class taxonomy classifier
    │   ├── evidence_router.py          # Verifier dispatch router
    │   ├── evidence_fusion.py          # Directional evidence fusion engine
    │   ├── uncertainty_engine.py       # 4-state formal uncertainty engine
    │   ├── image_processor.py          # Raw image & feature preprocessing
    │   └── orchestrator.py             # CAM-V pipeline coordinator
    ├── verifiers/
    │   ├── object_verifier.py          # Open-vocabulary YOLO-World grounding
    │   ├── count_verifier.py           # Object count & NMS verifier
    │   ├── attribute_verifier.py       # Region attribute & color matcher
    │   ├── spatial_verifier.py         # 2D/3D coordinate calculus verifier
    │   ├── action_verifier.py          # Relational action / VQA verifier
    │   ├── ocr_verifier.py             # Optical Character Recognition engine
    │   ├── absence_verifier.py         # Negative assertion / absence verifier
    │   └── scene_verifier.py           # Global scene context & scope filter
    ├── repair/
    │   └── repair_engine.py            # Constrained response repair & re-verification
    ├── baselines/
    │   ├── whole_response_judge.py     # Monolithic VLM judge baseline
    │   ├── claim_level_single_judge.py # Single judge baseline
    │   └── similarity_baseline.py      # Embedding similarity baseline
    ├── eval/
    │   ├── hqm_evaluator.py            # HQM psychometric evaluation suite
    │   ├── benchmark_runner.py         # Comparative benchmark runner
    │   ├── ablation.py                 # Section 14 ablation study runner (A0 to A5)
    │   ├── adversarial.py              # 7-operator adversarial generator
    │   ├── experiment_claim_classification.py
    │   ├── experiment_robustness.py
    │   ├── experiment_human_alignment.py
    │   └── experiment_guardrail.py
    └── static/                         # Executive Web Dashboard
        ├── index.html                  # Dashboard HTML layout
        ├── styles.css                  # Dark mode styling (#09090b)
        └── app.js                      # Reactive frontend & Chart.js charts
```

---

## 📚 References

1. **Bei Yan et al.**, *"Measuring the Measurers: Quality Evaluation of Hallucination Benchmarks for Large Vision-Language Models"*, arXiv:2406.17115, 2024.
2. **Yiyang Zhou et al.**, *"Analyzing and Mitigating Object Hallucination in Large Vision-Language Models"*, Findings of ACL 2024.
3. **Fuxiao Liu et al.**, *"Aligning Large Multimodal Models with Factually Augmented RLHF"*, NeurIPS 2023.
4. **Tianrui Guan et al.**, *"HallusionBench: An Advanced Diagnostic Suite for Entangled Language & Vision Hallucinations"*, CVPR 2024.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
