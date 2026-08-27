"""
Experiment 4: Real Robustness & Stability Evaluator for VisionGuard.
Calculates real score variance and stability across prompt paraphrases,
entity order permutations (Order Invariance), and negation transformations (Negation Sensitivity)
grounded directly on raw benchmark images without manual feature injection.
"""

from typing import List, Dict, Any, Tuple
import os
import numpy as np
from visionguard.pipeline.orchestrator import CAMVOrchestrator
from visionguard.schemas import Verdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RobustnessExperiment:
    """
    Evaluates system robustness, order invariance, and negation sensitivity dynamically over raw images.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.orchestrator = CAMVOrchestrator(use_mock=use_mock)

    def run_experiment(self, sample_dataset: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        def resolve_img(rel):
            p = os.path.join(BASE_DIR, rel)
            return p if os.path.exists(p) else rel

        # 1. Prompt Paraphrase Stability Testbed (Raw Images)
        paraphrase_testbed = [
            (
                "A young boy wearing a red shirt is riding a bicycle near a park.",
                "A boy dressed in a red shirt rides a bicycle beside a park.",
                resolve_img("data/images/000001.jpg")
            ),
            (
                "A brown dog is sitting beside a red bicycle.",
                "Beside a red bicycle, a brown dog is sitting.",
                resolve_img("data/images/000002.jpg")
            ),
            (
                "A blue car is parked on the snowy road.",
                "On the snowy road, a blue car is parked.",
                resolve_img("data/images/000003.jpg")
            ),
            (
                "A cup is on top of the desk beside a laptop.",
                "Next to a laptop on the desk sits a cup.",
                resolve_img("data/images/000009.jpg")
            ),
            (
                "An espresso tamper is resting on the marble counter next to a grinder.",
                "Next to a grinder on the marble counter rests an espresso tamper.",
                resolve_img("data/images/000012.jpg")
            )
        ]

        deltas = []
        para_verdict_matches = 0

        for orig, para, img_path in paraphrase_testbed:
            res_orig = self.orchestrator.analyze(orig, image_input=img_path)
            res_para = self.orchestrator.analyze(para, image_input=img_path)

            delta = abs(res_orig.overall_confidence - res_para.overall_confidence)
            deltas.append(delta)

            if res_orig.overall_verdict == res_para.overall_verdict:
                para_verdict_matches += 1

        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        stability_score = round(float(1.0 - avg_delta), 4)
        para_verdict_invariance = round(float(para_verdict_matches / len(paraphrase_testbed)), 4)

        # 2. Dynamic Order Invariance Evaluation (Raw Images)
        order_testbed = [
            (
                "A dog and a bicycle.",
                "A bicycle and a dog.",
                resolve_img("data/images/000002.jpg")
            ),
            (
                "A red apple and a yellow banana.",
                "A yellow banana and a red apple.",
                resolve_img("data/images/000017.jpg")
            ),
            (
                "A cup and a laptop.",
                "A laptop and a cup.",
                resolve_img("data/images/000009.jpg")
            ),
            (
                "A violinist and a cello.",
                "A cello and a violinist.",
                resolve_img("data/images/000010.jpg")
            ),
            (
                "A tamper and a grinder.",
                "A grinder and a tamper.",
                resolve_img("data/images/000012.jpg")
            )
        ]

        order_matches = 0
        order_deltas = []
        for s1, s2, img_path in order_testbed:
            r1 = self.orchestrator.analyze(s1, image_input=img_path)
            r2 = self.orchestrator.analyze(s2, image_input=img_path)
            if r1.overall_verdict == r2.overall_verdict:
                order_matches += 1
            order_deltas.append(abs(r1.overall_confidence - r2.overall_confidence))

        order_invariance_rate = round(float(order_matches / len(order_testbed)), 4)
        order_score = round(float(1.0 - np.mean(order_deltas)), 4)

        # 3. Dynamic Negation Sensitivity Evaluation (Raw Images)
        negation_testbed = [
            (
                "A car is present on the road.",
                "There are no cars on the road.",
                resolve_img("data/images/000003.jpg")
            ),
            (
                "A dog is present in the yard.",
                "There are no dogs in the yard.",
                resolve_img("data/images/000002.jpg")
            ),
            (
                "There are cars on the empty road.",
                "There are no cars on the empty road.",
                resolve_img("data/images/000006.jpg")
            ),
            (
                "A cup is on the desk.",
                "There is no cup on the desk.",
                resolve_img("data/images/000009.jpg")
            ),
            (
                "A helicopter is in the sky.",
                "There is no helicopter in the sky.",
                resolve_img("data/images/000014.jpg")
            )
        ]

        negation_flips = 0
        for pos_stmt, neg_stmt, img_path in negation_testbed:
            r_pos = self.orchestrator.analyze(pos_stmt, image_input=img_path)
            r_neg = self.orchestrator.analyze(neg_stmt, image_input=img_path)

            # Negation sensitivity: verdicts or confidence must appropriately flip or differentiate
            if r_pos.overall_verdict != r_neg.overall_verdict or abs(r_pos.overall_confidence - r_neg.overall_confidence) >= 0.20:
                negation_flips += 1

        negation_sensitivity_rate = round(float(negation_flips / len(negation_testbed)), 4)

        return {
            "num_test_pairs": len(paraphrase_testbed),
            "average_score_deviation": round(avg_delta, 4),
            "prompt_paraphrase_stability": stability_score,
            "verdict_invariance_rate": para_verdict_invariance,
            "order_invariance_score": order_score,
            "order_verdict_consistency": order_invariance_rate,
            "negation_sensitivity_score": negation_sensitivity_rate,
            "conclusion": f"CAM-V achieves {stability_score:.1%} paraphrase stability, {order_score:.1%} order invariance, and {negation_sensitivity_rate:.1%} negation sensitivity on raw benchmark images."
        }
