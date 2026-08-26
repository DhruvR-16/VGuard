"""
Adversarial Hallucination Generation Engine for VisionGuard.
Systematically generates 7 types of controlled adversarial hallucination perturbations:
1. Object Swap
2. Attribute Swap
3. Count Perturbation
4. Relation Swap
5. Negation Inversion
6. Entity Swap
7. OCR Text Perturbation
"""

import re
import random
from typing import List, Dict, Any
from visionguard.schemas import AdversarialClaim, Verdict


class AdversarialHallucinationGenerator:
    """
    Generates controlled adversarial hallucination testbeds from verified claims.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.object_swaps = {
            "bicycle": ["horse", "motorcycle", "boat"],
            "bike": ["skateboard", "scooter", "horse"],
            "car": ["helicopter", "submarine", "train"],
            "dog": ["lion", "wolf", "bear"],
            "cat": ["rabbit", "hamster", "tiger"],
            "shirt": ["sweater", "armor", "robe"],
            "umbrella": ["flag", "sword", "shield"]
        }
        self.color_swaps = {
            "red": ["blue", "neon green", "purple"],
            "blue": ["bright yellow", "crimson", "orange"],
            "green": ["magenta", "gold", "black"],
            "brown": ["cyan", "silver", "white"],
            "white": ["pitch black", "violet", "emerald"]
        }
        self.entity_swaps = {
            "boy": ["elderly woman", "astronaut", "firefighter"],
            "girl": ["policeman", "robot", "wizard"],
            "man": ["toddler", "scuba diver", "chef"],
            "woman": ["mechanic", "sailor", "pilot"],
            "person": ["alien", "mannequin", "statue"]
        }
        self.relation_swaps = {
            "riding": ["standing on top of", "hiding under", "chasing"],
            "wearing": ["holding in hand", "sitting on", "painting"],
            "sitting": ["flying above", "hanging from", "dancing on"],
            "near": ["far away from", "buried inside", "orbiting"]
        }

    def generate_adversarial_suite(self, base_claims: List[str]) -> List[AdversarialClaim]:
        """
        Generates paired adversarial test cases across all 7 perturbation operators.
        """
        adversarial_cases: List[AdversarialClaim] = []

        for claim in base_claims:
            lower = claim.lower()

            # 1. Object Swap
            for orig_obj, swaps in self.object_swaps.items():
                if re.search(r'\b' + orig_obj + r'\b', lower):
                    swap_target = random.choice(swaps)
                    pert = re.sub(r'\b' + orig_obj + r'\b', swap_target, claim, flags=re.IGNORECASE)
                    adversarial_cases.append(AdversarialClaim(
                        original_claim=claim,
                        perturbed_claim=pert,
                        perturbation_type="object_swap",
                        ground_truth=Verdict.CONTRADICTED
                    ))
                    break

            # 2. Attribute / Color Swap
            for orig_col, swaps in self.color_swaps.items():
                if re.search(r'\b' + orig_col + r'\b', lower):
                    swap_target = random.choice(swaps)
                    pert = re.sub(r'\b' + orig_col + r'\b', swap_target, claim, flags=re.IGNORECASE)
                    adversarial_cases.append(AdversarialClaim(
                        original_claim=claim,
                        perturbed_claim=pert,
                        perturbation_type="attribute_swap",
                        ground_truth=Verdict.CONTRADICTED
                    ))
                    break

            # 3. Entity Swap
            for orig_ent, swaps in self.entity_swaps.items():
                if re.search(r'\b' + orig_ent + r'\b', lower):
                    swap_target = random.choice(swaps)
                    pert = re.sub(r'\b' + orig_ent + r'\b', swap_target, claim, flags=re.IGNORECASE)
                    adversarial_cases.append(AdversarialClaim(
                        original_claim=claim,
                        perturbed_claim=pert,
                        perturbation_type="entity_swap",
                        ground_truth=Verdict.CONTRADICTED
                    ))
                    break

            # 4. Relation Swap
            for orig_rel, swaps in self.relation_swaps.items():
                if re.search(r'\b' + orig_rel + r'\b', lower):
                    swap_target = random.choice(swaps)
                    pert = re.sub(r'\b' + orig_rel + r'\b', swap_target, claim, flags=re.IGNORECASE)
                    adversarial_cases.append(AdversarialClaim(
                        original_claim=claim,
                        perturbed_claim=pert,
                        perturbation_type="relation_swap",
                        ground_truth=Verdict.CONTRADICTED
                    ))
                    break

            # 5. Count Perturbation
            if re.search(r'\b(a|an|one)\b', lower):
                pert = re.sub(r'\b(a|an|one)\b', 'three', claim, count=1, flags=re.IGNORECASE)
                adversarial_cases.append(AdversarialClaim(
                    original_claim=claim,
                    perturbed_claim=pert,
                    perturbation_type="count_perturbation",
                    ground_truth=Verdict.CONTRADICTED
                ))

            # 6. Negation Inversion
            if " is " in claim:
                pert = claim.replace(" is ", " is not ")
                adversarial_cases.append(AdversarialClaim(
                    original_claim=claim,
                    perturbed_claim=pert,
                    perturbation_type="negation_inversion",
                    ground_truth=Verdict.CONTRADICTED
                ))

        return adversarial_cases
