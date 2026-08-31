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
            if not rel:
                return None
            if os.path.isabs(rel) and os.path.exists(rel):
                return rel
            p_real = os.path.join(BASE_DIR, "data", "images_real", os.path.basename(rel))
            if os.path.exists(p_real):
                return p_real
            p = os.path.join(BASE_DIR, rel)
            return p if os.path.exists(p) else rel

        # Dynamic construction from provided dataset samples if available
        if sample_dataset and len(sample_dataset) >= 5:
            paraphrase_testbed = []
            order_testbed = []
            negation_testbed = []

            for s in sample_dataset[:10]:
                txt = s.get("claim") or s.get("response_text") or ""
                img = s.get("full_image_path") or s.get("image_path")
                if not txt or not img:
                    continue
                # Paraphrase: active/passive or descriptive prefix
                para_txt = f"Indeed, {txt[0].lower() + txt[1:]}" if len(txt) > 2 else txt
                paraphrase_testbed.append((txt, para_txt, resolve_img(img)))

                # Order: if multi-word or compound
                words = txt.split()
                if len(words) >= 4:
                    order_txt = f"{' '.join(words[2:])}, {' '.join(words[:2])}"
                else:
                    order_txt = txt
                order_testbed.append((txt, order_txt, resolve_img(img)))

                # Negation
                neg_txt = f"There is no {txt.lower()}" if not txt.lower().startswith("there") else f"It is false that {txt}"
                negation_testbed.append((txt, neg_txt, resolve_img(img)))

            if len(paraphrase_testbed) < 3:
                paraphrase_testbed = None
        else:
            paraphrase_testbed = None

        if not paraphrase_testbed:
            # Standard Curated Photographic Testbed
            paraphrase_testbed = [
                (
                    "A yellow car is visible on the street.",
                    "On the city street, a yellow car is visible.",
                    resolve_img("data/images_real/real_000001.jpg") or resolve_img("data/images/000001.jpg")
                ),
                (
                    "A brown dog is present on the grass.",
                    "Present on the green grass is a brown dog.",
                    resolve_img("data/images_real/real_000002.jpg") or resolve_img("data/images/000002.jpg")
                ),
                (
                    "A white mug is resting on the desk.",
                    "Resting on top of the desk is a white mug.",
                    resolve_img("data/images_real/real_000003.jpg") or resolve_img("data/images/000003.jpg")
                ),
                (
                    "A bowl containing fruit is on the table.",
                    "On the dining table sits a bowl containing fruit.",
                    resolve_img("data/images_real/real_000004.jpg") or resolve_img("data/images/000009.jpg")
                ),
                (
                    "A soccer player in a blue jersey is on the pitch.",
                    "On the soccer pitch is an athlete wearing a blue jersey.",
                    resolve_img("data/images_real/real_000005.jpg") or resolve_img("data/images/000010.jpg")
                )
            ]

            order_testbed = [
                (
                    "A yellow car and commercial buildings.",
                    "Commercial buildings and a yellow car.",
                    resolve_img("data/images_real/real_000001.jpg") or resolve_img("data/images/000001.jpg")
                ),
                (
                    "A brown dog and a wooden bowl.",
                    "A wooden bowl and a brown dog.",
                    resolve_img("data/images_real/real_000002.jpg") or resolve_img("data/images/000002.jpg")
                ),
                (
                    "A laptop and a white coffee mug.",
                    "A white coffee mug and a laptop.",
                    resolve_img("data/images_real/real_000003.jpg") or resolve_img("data/images/000003.jpg")
                ),
                (
                    "Fresh apples and yellow bananas.",
                    "Yellow bananas and fresh apples.",
                    resolve_img("data/images_real/real_000004.jpg") or resolve_img("data/images/000017.jpg")
                ),
                (
                    "A blue van and a red truck.",
                    "A red truck and a blue van.",
                    resolve_img("data/images_real/real_000008.jpg") or resolve_img("data/images/000012.jpg")
                )
            ]

            negation_testbed = [
                (
                    "A yellow car is visible on the street.",
                    "There are no cars visible on the street.",
                    resolve_img("data/images_real/real_000001.jpg") or resolve_img("data/images/000001.jpg")
                ),
                (
                    "A brown dog is present on the grass.",
                    "There are no dogs present on the grass.",
                    resolve_img("data/images_real/real_000002.jpg") or resolve_img("data/images/000002.jpg")
                ),
                (
                    "A white mug is resting on the desk.",
                    "There is no mug resting on the desk.",
                    resolve_img("data/images_real/real_000003.jpg") or resolve_img("data/images/000003.jpg")
                ),
                (
                    "A bowl containing fruit is on the table.",
                    "There is no bowl or fruit on the table.",
                    resolve_img("data/images_real/real_000004.jpg") or resolve_img("data/images/000009.jpg")
                ),
                (
                    "A blue van and a red truck are on the road.",
                    "There are no vehicles on the road.",
                    resolve_img("data/images_real/real_000008.jpg") or resolve_img("data/images/000014.jpg")
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

        # 2. Dynamic Order Invariance Evaluation
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

        # 3. Dynamic Negation Sensitivity Evaluation
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
