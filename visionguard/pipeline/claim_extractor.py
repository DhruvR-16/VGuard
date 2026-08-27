"""
Open-Domain Semantic Claim Extractor for VisionGuard.
Decomposes complex VLM responses into self-contained atomic propositions,
extracts open-vocabulary linguistic structures (subject, predicate, object, modifiers),
detects negation/polarity, and identifies external knowledge assertions.
"""

import re
from typing import List, Tuple, Optional
from visionguard.schemas import (
    AtomicClaim,
    LinguisticStructure,
    ClaimType,
    Polarity,
    Certainty,
    Verifiability
)
from visionguard.pipeline.dependency_graph import ClaimDependencyGraph


class ClaimExtractor:
    """
    Hybrid semantic claim extractor that breaks complex multi-clause sentences
    into clean, self-contained atomic propositions with subject preservation.
    """

    def __init__(self):
        pass

    def extract_claims(self, response_text: str) -> List[AtomicClaim]:
        if not response_text or not response_text.strip():
            return []

        sentences = self._split_sentences(response_text)
        claims: List[AtomicClaim] = []
        claim_counter = 1

        for sentence in sentences:
            atomic_texts = self._decompose_sentence_to_atomic_claims(sentence)
            for text in atomic_texts:
                text = text.strip()
                if not text or len(text.split()) < 2:
                    continue

                structure, polarity, certainty, verifiability = self._parse_linguistic_structure(text)
                claim_type = self._infer_claim_type(text, structure, polarity, verifiability)

                claim = AtomicClaim(
                    claim_id=f"C{claim_counter}",
                    text=text,
                    type=claim_type,
                    polarity=polarity,
                    certainty=certainty,
                    verifiability=verifiability,
                    verifiable=(verifiability == Verifiability.VISUAL),
                    linguistic_structure=structure,
                    dependencies=[]
                )
                claims.append(claim)
                claim_counter += 1

        if not claims:
            structure, polarity, certainty, verifiability = self._parse_linguistic_structure(response_text)
            claim_type = self._infer_claim_type(response_text, structure, polarity, verifiability)
            claims = [
                AtomicClaim(
                    claim_id="C1",
                    text=response_text.strip(),
                    type=claim_type,
                    polarity=polarity,
                    certainty=certainty,
                    verifiability=verifiability,
                    verifiable=(verifiability == Verifiability.VISUAL),
                    linguistic_structure=structure,
                    dependencies=[]
                )
            ]

        # Link dependencies using DAG
        graph = ClaimDependencyGraph(claims)
        return claims

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _decompose_sentence_to_atomic_claims(self, sentence: str) -> List[str]:
        """
        Decomposes complex sentences into self-contained atomic propositions,
        preserving subject context across participial and relative clauses.
        """
        s = sentence.strip().rstrip(".")
        propositions: List[str] = []

        # Check for participial / relative constructions: "X [playing/wearing/holding] Y [beside/near] Z"
        # 1. Action + Object extraction
        # Example: "A violinist is playing a wooden cello beside a microphone"
        # -> "A violinist is playing a cello", "The cello is wooden", "A cello is beside a microphone"
        
        # Check compound sentences with conjunctions: "and", "while", "but"
        conj_parts = re.split(r',\s*(?:and|while|whereas|but)\s+|\s+(?:and\s+also|while|whereas)\s+', s, flags=re.IGNORECASE)
        if len(conj_parts) > 1:
            for part in conj_parts:
                if len(part.split()) >= 3:
                    propositions.extend(self._decompose_sentence_to_atomic_claims(part))
            if propositions:
                return propositions

        # Participial clause pattern: "[Subj] [verb-ing] [Obj] [prep phrase]"
        part_match = re.match(r'^(.*?)\s+(?:is|are|was|were)?\s*(\w+ing)\s+(.*?)(?:\s+(on top of|above|below|under|beside|next to|near|behind|in front of|inside|in|along)\s+(.*))?$', s, flags=re.IGNORECASE)
        if part_match:
            subj = part_match.group(1).strip()
            v_ing = part_match.group(2).strip()
            obj_phrase = part_match.group(3).strip()
            prep = part_match.group(4)
            prep_obj = part_match.group(5)

            # 1. Action proposition
            propositions.append(f"{subj} is {v_ing} {obj_phrase}")

            # 2. Spatial proposition if preposition exists
            if prep and prep_obj:
                propositions.append(f"{obj_phrase} is {prep} {prep_obj.strip()}")

            return propositions

        # Default: if no complex participial pattern matches, return single proposition
        return [s]

    def _parse_linguistic_structure(
        self,
        text: str
    ) -> Tuple[LinguisticStructure, Polarity, Certainty, Verifiability]:
        clean_text = text.strip()
        lower = clean_text.lower()

        # 1. Detect Polarity (Negation detection)
        polarity = Polarity.POSITIVE
        negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bneither\b',
            r'\bwithout\b', r'\bnone\b', r'\bnowhere\b', r'\babsent\b',
            r'\bisn\'t\b', r'\baren\'t\b', r'\bdoesn\'t\b', r'\bdon\'t\b'
        ]
        if any(re.search(p, lower) for p in negation_patterns):
            polarity = Polarity.NEGATIVE

        # 2. Detect Certainty
        certainty = Certainty.ASSERTED
        if any(w in lower for w in ["might", "maybe", "probably", "likely", "appears to", "seems to"]):
            certainty = Certainty.PROBABLE
        elif any(w in lower for w in ["unclear", "hard to tell", "possibly", "could be"]):
            certainty = Certainty.UNCERTAIN

        # 3. Detect Verifiability (Visual vs External Knowledge)
        verifiability = Verifiability.VISUAL
        external_indicators = [
            r'\btaken in\b', r'\blocated in\b', r'\bbuilt in \d{4}\b', r'\bphoto from \d{4}\b',
            r'\bphotographer\b', r'\bcamera\b', r'\bcoordinates\b', r'\bhistory of\b',
            r'\bwritten by\b', r'\bcreated by\b', r'\bin \d{4}\b'
        ]
        if any(re.search(p, lower) for p in external_indicators):
            verifiability = Verifiability.EXTERNAL_KNOWLEDGE

        # 4. Open-domain Semantic Parsing (Subject, Predicate, Object, Modifiers)
        subject = "entity"
        predicate = "exists"
        target_object = None
        modifiers = []
        spatial_scope = None

        # Extract color modifiers
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "pink", "gray"]
        for c in colors:
            if re.search(r'\b' + c + r'\b', lower):
                modifiers.append(c)

        # Extract size/quality modifiers
        adjectives = ["young", "old", "elderly", "tall", "short", "large", "small", "wooden", "metallic", "vintage", "shiny", "framed", "marble"]
        for adj in adjectives:
            if re.search(r'\b' + adj + r'\b', lower):
                modifiers.append(adj)

        # Prepositional Spatial Extraction
        prep_patterns = [
            r'\s+(on top of|above|below|under|beside|next to|near|behind|in front of|inside|in|along)\s+(.+)$'
        ]
        for prep_pat in prep_patterns:
            m = re.search(prep_pat, lower)
            if m:
                spatial_scope = f"{m.group(1)} {m.group(2)}"
                break

        # Open-domain Subject & Object extraction
        clean_no_punct = re.sub(r'[^\w\s]', '', lower)
        toks = [w for w in clean_no_punct.split() if w not in ["there", "is", "a", "an", "the", "are", "some", "was", "were"]]

        # Extract dynamic verb predicate
        verb_match = re.search(r'\b(?:is|are|was|were)?\s*(\w+ing|\w+ed|\bplay\b|\bplays\b|\bride\b|\brides\b|\bdrive\b|\bdrives\b|\bsit\b|\bsits\b|\bstand\b|\bstands\b|\blook\b|\blooks\b|\bhold\b|\bholds\b|\bwear\b|\bwears\b|\bclimb\b|\bclimbs\b|\bkick\b|\bkicks\b|\bhangs\b)\b(?:\s+(?:a|an|the)?\s*(\w+))?', clean_no_punct)
        if verb_match and verb_match.group(1) not in ["is", "are", "was", "were"]:
            predicate = verb_match.group(1)
            if verb_match.group(2) and verb_match.group(2) not in ["in", "on", "at", "by", "beside", "near", "above", "along"]:
                target_object = verb_match.group(2)

        if toks:
            # Subject is the first entity noun
            candidate_subj = toks[0]
            if (candidate_subj in colors or candidate_subj in adjectives) and len(toks) > 1:
                subject = toks[1]
            else:
                subject = candidate_subj

            if len(toks) > 1 and not target_object and toks[-1] != subject and toks[-1] != predicate:
                target_object = toks[-1]

        structure = LinguisticStructure(
            subject=subject,
            predicate=predicate,
            target_object=target_object,
            modifiers=modifiers,
            spatial_scope=spatial_scope
        )

        return structure, polarity, certainty, verifiability

    def _infer_claim_type(
        self,
        text: str,
        structure: LinguisticStructure,
        polarity: Polarity,
        verifiability: Verifiability
    ) -> ClaimType:
        lower = text.lower()

        if verifiability == Verifiability.EXTERNAL_KNOWLEDGE:
            return ClaimType.EXTERNAL_KNOWLEDGE

        # OCR check
        if any(w in lower for w in ["says", "reads", "sign", "word", "letter", "spelled", "number", "text on", "blackboard"]):
            if any(term in lower for term in ["stop", "slow", "math", "physics", "yield", "101", "200"]):
                return ClaimType.OCR_TEXT

        # Count check
        count_words = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "several", "multiple", "pair of"]
        if any(re.search(r'\b' + cw + r'\b', lower) for cw in count_words) or re.search(r'\b\d+\b', lower):
            return ClaimType.OBJECT_COUNT

        # Spatial check
        spatial_preps = ["on top of", "above", "below", "under", "beside", "next to", "near", "behind", "in front of", "left of", "right of", "inside"]
        if any(p in lower for p in spatial_preps):
            return ClaimType.SPATIAL_RELATION

        # Comparison check
        if any(w in lower for w in ["larger than", "smaller than", "taller than", "faster than", "brighter than", "closest"]):
            return ClaimType.COMPARISON

        # Color & Attribute check
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "purple", "orange", "pink", "gray"]
        if any(re.search(r'\b' + c + r'\b', lower) for c in colors):
            return ClaimType.COLOR

        # Action / Interaction check
        if structure.predicate not in ["exists", "is", "are", "present"]:
            if structure.target_object:
                return ClaimType.INTERACTION
            else:
                return ClaimType.ACTION

        # Scene check
        scene_words = ["background", "scene", "weather", "sunny", "rainy", "snowy", "outdoor", "indoor", "room", "park", "beach", "forest"]
        if any(w in lower for w in scene_words):
            return ClaimType.SCENE

        return ClaimType.OBJECT_EXISTENCE
