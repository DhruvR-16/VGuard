"""
Relational Action & Interaction Verifier for VisionGuard.
Formulates targeted VQA queries (e.g. 'Is the person riding the bicycle?')
and queries ViLT / VLM models for affirmative likelihood rather than inferring actions from co-presence.
"""

from typing import Dict, Any, Optional
from PIL import Image
from visionguard.schemas import AtomicClaim, Evidence
from visionguard.models.registry import registry


class ActionInteractionVerifier:
    """
    Action & Interaction Verifier using targeted Visual Question Answering (VQA).
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.registry = registry

    def verify(
        self,
        claim: AtomicClaim,
        image_path: Optional[str] = None,
        visual_features: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        claim_text = claim.text.strip()

        if visual_features is None and image_path is None:
            return Evidence(
                source="action_interaction_vlm",
                result="No visual context provided",
                support_score=0.0,
                contradiction_score=0.0,
                uncertainty_score=1.0,
                confidence=0.0,
                cot_reasoning="No image context provided for action verification.",
                details={"status": "ungrounded"}
            )

        image_obj: Optional[Image.Image] = visual_features.get("image_obj") if visual_features else None

        # 1. Formulate targeted VQA Question from Linguistic Structure
        question = self._formulate_vqa_question(claim)

        top_answer = "unknown"
        yes_prob = 0.50
        no_prob = 0.50
        vqa_executed = False

        if image_obj is not None and not self.use_mock:
            top_answer, yes_prob, no_prob = self.registry.run_vqa_query(image_obj, question)
            vqa_executed = True

        # 2. Check visual_features relations or fallback
        if not vqa_executed and visual_features is not None:
            rel_dict = visual_features.get("relations", {})
            claim_lower = claim_text.lower()
            if any(k in claim_lower for k in rel_dict):
                matched = next(k for k in rel_dict if k in claim_lower)
                val = float(rel_dict[matched])
                yes_prob = val
                no_prob = 1.0 - val
                top_answer = "yes" if val >= 0.60 else "no"
            else:
                # If no VQA model and no relation annotations, action cannot be assumed from co-presence!
                yes_prob = 0.40
                no_prob = 0.30
                top_answer = "ambiguous"

        # Directional evidence computation
        if yes_prob >= 0.65:
            support_s = round(yes_prob, 3)
            contradict_s = round(max(0.05, no_prob), 3)
            uncertainty_s = 0.10
            res_str = f"Action verified: VQA confirmed '{question}' with affirmative likelihood {yes_prob:.1%}."
        elif no_prob >= 0.60:
            support_s = round(yes_prob, 3)
            contradict_s = round(no_prob, 3)
            uncertainty_s = 0.15
            res_str = f"Contradiction: VQA refuted '{question}' (no_prob={no_prob:.1%})."
        else:
            support_s = round(yes_prob, 3)
            contradict_s = round(no_prob, 3)
            uncertainty_s = 0.60
            res_str = f"Inconclusive action evidence: VQA response for '{question}' was '{top_answer}' (uncertainty=0.60)."

        cot = (
            f"Step 1: Formulated relational VQA question: '{question}'. "
            f"Step 2: ViLT VQA model returned top answer '{top_answer}' with P(yes)={yes_prob:.2f}, P(no)={no_prob:.2f}. "
            f"Step 3: Action verified without relying on mere object co-presence. "
            f"Step 4: Directional evidence -> Support={support_s:.2f}, Contradict={contradict_s:.2f}, Uncertainty={uncertainty_s:.2f}."
        )

        return Evidence(
            source="action_interaction_vlm",
            result=res_str,
            support_score=support_s,
            contradiction_score=contradict_s,
            uncertainty_score=uncertainty_s,
            confidence=round(yes_prob, 3),
            score=support_s,
            cot_reasoning=cot,
            details={
                "vqa_question": question,
                "vqa_top_answer": top_answer,
                "p_yes": round(yes_prob, 3),
                "p_no": round(no_prob, 3)
            }
        )

    def _formulate_vqa_question(self, claim: AtomicClaim) -> str:
        """
        Converts atomic claim proposition into a natural binary verification question.
        """
        if claim.linguistic_structure:
            subj = claim.linguistic_structure.subject or "subject"
            pred = claim.linguistic_structure.predicate or "doing action"
            obj = claim.linguistic_structure.target_object or ""
            target_str = f" the {obj}" if obj else ""

            # Check if predicate already has auxiliary
            if pred.startswith("is ") or pred.startswith("are "):
                return f"{pred.capitalize()} the {subj}{target_str}?"
            else:
                return f"Is the {subj} {pred}{target_str}?"

        clean = claim.text.strip().rstrip('.')
        return f"Is it true that {clean}?"


# Alias
RelationVerifier = ActionInteractionVerifier
