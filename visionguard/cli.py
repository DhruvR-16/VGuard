"""
Command Line Interface (CLI) for VisionGuard.
Supports server launch, claim analysis, response repair, benchmarking, ablation,
scientific evaluation experiments, and adversarial suite generation.
"""

import argparse
import json
import uvicorn
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.eval.benchmark_runner import BenchmarkRunner
from visionguard.eval.ablation import AblationStudyRunner
from visionguard.eval.adversarial import AdversarialHallucinationGenerator
from visionguard.eval.experiment_claim_classification import ClaimClassificationExperiment
from visionguard.eval.experiment_robustness import RobustnessExperiment
from visionguard.eval.experiment_human_alignment import HumanAlignmentExperiment
from visionguard.eval.experiment_guardrail import GuardrailEffectivenessExperiment
from visionguard.data_loader import get_sample_dataset, get_atomic_claims_dataset


def main():
    parser = argparse.ArgumentParser(
        description="VisionGuard (CAM-V): Claim-Aware Multi-Evidence Verification & Response Repair"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Server Subcommand
    server_parser = subparsers.add_parser("server", help="Launch FastAPI server and Web Dashboard")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    server_parser.add_argument("--port", type=int, default=8000, help="Port number")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # 2. Analyze Subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze and verify a VLM response")
    analyze_parser.add_argument("text", type=str, help="VLM generated text response")
    analyze_parser.add_argument("--image", type=str, default=None, help="Path to image file")
    analyze_parser.add_argument("--mock", action="store_true", help="Run with mock models")

    # 3. Repair Subcommand
    repair_parser = subparsers.add_parser("repair", help="Repair hallucinated response")
    repair_parser.add_argument("text", type=str, help="VLM generated text response")
    repair_parser.add_argument("--image", type=str, default=None, help="Path to image file")
    repair_parser.add_argument("--mock", action="store_true", help="Run with mock models")

    # 4. Eval Subcommand
    eval_parser = subparsers.add_parser("eval", help="Run benchmark or specific scientific experiment")
    eval_parser.add_argument(
        "--experiment",
        type=str,
        default="benchmark",
        choices=["benchmark", "ablation", "robustness", "human_alignment", "guardrail", "claim_classification", "all"],
        help="Experiment to execute"
    )
    eval_parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"], help="Dataset split")
    eval_parser.add_argument("--mock", action="store_true", help="Run with mock models")

    # 5. Ablation Subcommand
    ablation_parser = subparsers.add_parser("ablation", help="Run Section 14 ablation experiments")
    ablation_parser.add_argument("--mock", action="store_true", help="Run with mock models")

    # 6. Experiments Subcommand
    exp_parser = subparsers.add_parser("experiments", help="Run all 6 evaluation experiments")
    exp_parser.add_argument("--mock", action="store_true", help="Run with mock models")

    # 7. Adversarial Subcommand
    adv_parser = subparsers.add_parser("adversarial", help="Generate adversarial hallucination cases")
    adv_parser.add_argument("claim", type=str, help="Base factual claim")

    args = parser.parse_args()

    if args.command == "server":
        print(f"Starting VisionGuard Server on http://{args.host}:{args.port}")
        uvicorn.run("visionguard.api.main:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "analyze":
        orch = CAMVOrchestrator(use_mock=args.mock)
        res = orch.analyze(response_text=args.text, image_input=args.image)
        print("\n=== VisionGuard CAM-V Analysis Result ===")
        print(f"Overall Verdict: {res.overall_verdict.value}")
        print(f"Confidence: {res.overall_confidence:.1%}")
        print(f"Hallucination Probability: {res.hallucination_probability:.1%}")
        print("\nDecomposed Claims:")
        for c in res.claims:
            print(f"  [{c.claim.claim_id}] {c.claim.type.value.upper()}: '{c.claim.text}' -> {c.verdict.value} (Fused Score: {c.fused_score:.3f})")
        if res.repaired_response:
            print(f"\nRepaired Response: '{res.repaired_response.repaired_text}'")

    elif args.command == "repair":
        orch = CAMVOrchestrator(use_mock=args.mock)
        res = orch.analyze(response_text=args.text, image_input=args.image, enable_repair=True)
        print("\n=== VisionGuard Constrained Response Repair ===")
        print(f"Original Text: {args.text}")
        if res.repaired_response:
            print(f"Verified Facts: {res.repaired_response.verified_facts}")
            print(f"Removed Claims: {res.repaired_response.removed_claims}")
            print(f"Repaired Output: {res.repaired_response.repaired_text}")
        else:
            print("Status: Ungrounded (No image was provided for visual verification and repair).")

    elif args.command == "eval":
        dataset = get_atomic_claims_dataset(args.split) or get_sample_dataset()
        exp = args.experiment

        if exp == "benchmark":
            runner = BenchmarkRunner(use_mock=args.mock)
            results = runner.run_benchmark(dataset)
            print("\n=== VisionGuard Comparative Benchmark Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "ablation":
            runner = AblationStudyRunner(use_mock=args.mock)
            results = runner.run_ablation(dataset)
            print("\n=== VisionGuard Section 14 Ablation Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "robustness":
            results = RobustnessExperiment(use_mock=args.mock).run_experiment(get_sample_dataset())
            print("\n=== VisionGuard Robustness & Stability Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "human_alignment":
            results = HumanAlignmentExperiment(use_mock=args.mock).run_experiment(dataset)
            print("\n=== VisionGuard Human Alignment & Inter-Annotator Agreement Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "guardrail":
            results = GuardrailEffectivenessExperiment(use_mock=args.mock).run_experiment(get_sample_dataset())
            print("\n=== VisionGuard Response Repair Guardrail Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "claim_classification":
            results = ClaimClassificationExperiment(use_mock=args.mock).run_experiment(get_sample_dataset())
            print("\n=== VisionGuard Claim Classification Experiment Results ===")
            print(json.dumps(results, indent=2))

        elif exp == "all":
            bm = BenchmarkRunner(use_mock=args.mock).run_benchmark(dataset)
            ab = AblationStudyRunner(use_mock=args.mock).run_ablation(dataset)
            rob = RobustnessExperiment(use_mock=args.mock).run_experiment(get_sample_dataset())
            ha = HumanAlignmentExperiment(use_mock=args.mock).run_experiment(dataset)
            gr = GuardrailEffectivenessExperiment(use_mock=args.mock).run_experiment(get_sample_dataset())

            print("\n=== VisionGuard Complete 5-Experiment Suite Results ===")
            print(json.dumps({
                "benchmark": bm,
                "ablation": ab,
                "robustness": rob,
                "human_alignment": ha,
                "guardrail": gr
            }, indent=2))

    elif args.command == "ablation":
        runner = AblationStudyRunner(use_mock=args.mock)
        results = runner.run_ablation(get_atomic_claims_dataset() or get_sample_dataset())
        print("\n=== VisionGuard Section 14 Ablation Results ===")
        print(json.dumps(results, indent=2))

    elif args.command == "experiments":
        samples = get_sample_dataset()
        atomic_claims = get_atomic_claims_dataset()

        exp1 = ClaimClassificationExperiment(use_mock=args.mock).run_experiment(samples)
        exp4 = RobustnessExperiment(use_mock=args.mock).run_experiment(samples)
        exp5 = HumanAlignmentExperiment(use_mock=args.mock).run_experiment(atomic_claims)
        exp6 = GuardrailEffectivenessExperiment(use_mock=args.mock).run_experiment(samples)

        print("\n=== VisionGuard 6-Experiment Suite Results ===")
        print(json.dumps({
            "exp1_claim_classification": exp1,
            "exp4_robustness": exp4,
            "exp5_human_alignment": exp5,
            "exp6_guardrail_effectiveness": exp6
        }, indent=2))

    elif args.command == "adversarial":
        gen = AdversarialHallucinationGenerator()
        cases = gen.generate_adversarial_suite([args.claim])
        print(f"\n=== Adversarial Cases for: '{args.claim}' ===")
        for c in cases:
            print(f"  [{c.perturbation_type}]: '{c.perturbed_claim}' -> {c.ground_truth.value}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
