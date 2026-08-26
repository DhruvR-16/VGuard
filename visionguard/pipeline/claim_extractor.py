"""
Structured Claim Extractor Module for VisionGuard.
Decomposes natural language VLM responses into atomic, independently verifiable propositions.
Extracts semantic structure (subject, predicate, object, modifiers, spatial scope),
preserves polarity/negation, classifies across the 12-class taxonomy, and constructs the Claim Dependency Graph.
"""

import re
from typing import List, Optional, Tuple
from visionguard.schemas import (
    AtomicClaim,
    ClaimType,
    Polarity,
    Certainty,
    LinguisticStructure
)
from visionguard.pipeline.dependency_graph import ClaimDependencyGraph


class ClaimExtractor:
    """
    Structured semantic claim extractor with negation tracking and dependency graph construction.
    """

    def __init__(self):
        pass

    def extract_claims(self, response_text: str) -> List[AtomicClaim]:
        """
        Decomposes a response paragraph into structured atomic claims.
        """
        if not response_text or not response_text.strip():
            return []

        raw_sentences = self._split_sentences(response_text)
        claims: List[AtomicClaim] = []
        claim_counter = 1

        for sentence in raw_sentences:
            sub_clauses = self._split_clauses(sentence)
            for clause in sub_clauses:
                clause = clause.strip()
                if not clause or len(clause.split()) < 2:
                    continue

                structure, polarity, certainty = self._parse_linguistic_structure(clause)
                claim_type = self._infer_claim_type(clause, structure, polarity)

                claim = AtomicClaim(
                    claim_id=f"C{claim_counter}",
                    text=clause,
                    type=claim_type,
                    polarity=polarity,
                    certainty=certainty,
                    verifiable=(claim_type != ClaimType.OUT_OF_SCOPE),
                    linguistic_structure=structure,
                    dependencies=[]
                )
                claims.append(claim)
                claim_counter += 1

        # If no sub-claims extracted, treat whole text as single atomic claim
        if not claims:
            structure, polarity, certainty = self._parse_linguistic_structure(response_text)
            claim_type = self._infer_claim_type(response_text, structure, polarity)
            claims = [
                AtomicClaim(
                    claim_id="C1",
                    text=response_text.strip(),
                    type=claim_type,
                    polarity=polarity,
                    certainty=certainty,
                    verifiable=(claim_type != ClaimType.OUT_OF_SCOPE),
                    linguistic_structure=structure,
                    dependencies=[]
                )
            ]

        # Build Dependency Graph & Link IDs
        graph = ClaimDependencyGraph(claims)
        return claims

    def _split_sentences(self, text: str) -> List[str]:
        # Split on standard punctuation while preserving tokens
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _split_clauses(self, sentence: str) -> List[str]:
        # Split coordinating conjunctions, relative clauses, and participial phrases
        patterns = [
            r'\s+and\s+',
            r'\s+while\s+',
            r'\s+whereas\s+',
            r'\s+as\s+well\s+as\s+',
            r',\s*(?:and|but|while|which|who)\s+',
            r'\s+(?:who\s+is|that\s+is|which\s+is)\s+',
            r'(?<=[a-zA-Z])\s+(?:wearing|holding|carrying|riding|driving|sitting near|standing in)\s+'
        ]
        clauses = [sentence]
        for pat in patterns:
            new_clauses = []
            for c in clauses:
                parts = re.split(pat, c, flags=re.IGNORECASE)
                if len(parts) > 1 and len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
                    new_clauses.extend(parts)
                else:
                    new_clauses.append(c)
            clauses = new_clauses

        return [c.strip() for c in clauses if c.strip()]

    def _parse_linguistic_structure(self, text: str) -> Tuple[LinguisticStructure, Polarity, Certainty]:
        """
        Extracts subject, predicate, object, modifiers, and detects negation/polarity.
        """
        clean_text = text.strip()
        lower = clean_text.lower()

        # 1. Detect Polarity (Negation detection)
        polarity = Polarity.POSITIVE
        negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bneither\b',
            r'\bwithout\b', r'\bnone\b', r'\bnowhere\b', r'\babsent\b'
        ]
        if any(re.search(p, lower) for p in negation_patterns):
            polarity = Polarity.NEGATIVE

        # 2. Detect Certainty
        certainty = Certainty.ASSERTED
        if any(w in lower for w in ["maybe", "might", "possibly", "could be"]):
            certainty = Certainty.UNCERTAIN
        elif any(w in lower for w in ["likely", "probably", "appears to", "seems to"]):
            certainty = Certainty.PROBABLE

        # 3. Extract Modifiers (colors, textures, descriptors)
        colors = ["red", "blue", "green", "yellow", "brown", "white", "black", "orange", "purple", "gray", "grey", "golden", "pink"]
        attributes = ["young", "old", "large", "small", "tall", "short", "snowy", "wooden", "metal", "glass", "leather", "fast", "slow"]
        modifiers = []
        for c in colors + attributes:
            if re.search(r'\b' + c + r'\b', lower):
                modifiers.append(c)

        # 4. Extract Spatial Scope
        spatial_scope = None
        spatial_match = re.search(r'\b(near|beside|in|on|at|under|above|behind|in front of|inside|outside|below)\s+([^,.]+)', clean_text, re.IGNORECASE)
        if spatial_match:
            spatial_scope = spatial_match.group(0).strip()

        # 5. Extract Subject, Predicate, and Target Object
        subject = "object"
        predicate = "is present"
        target_object = None

        action_verbs = [
            "riding", "wearing", "holding", "driving", "running", "walking",
            "sitting", "standing", "looking at", "eating", "flying", "carrying",
            "pushing", "pulling", "chasing", "watching"
        ]
        nouns = [
            "boy", "girl", "man", "woman", "person", "dog", "cat", "bicycle", "bike",
            "car", "vehicle", "shirt", "jacket", "coat", "park", "street", "table",
            "desk", "umbrella", "cup", "river", "road", "scene", "photo", "sign",
            "ball", "balloon", "tree", "building", "alien", "nematode"
        ]

        # Match nouns in text
        matched_nouns = [n for n in nouns if re.search(r'\b' + n + r'\b', lower)]
        if matched_nouns:
            subject = matched_nouns[0]
            if len(matched_nouns) > 1:
                target_object = matched_nouns[1]

        # Match action verbs
        for verb in action_verbs:
            if re.search(r'\b' + verb + r'\b', lower):
                predicate = verb
                break

        structure = LinguisticStructure(
            subject=subject,
            predicate=predicate,
            target_object=target_object,
            modifiers=modifiers,
            spatial_scope=spatial_scope
        )
        return structure, polarity, certainty

    def _infer_claim_type(self, text: str, structure: LinguisticStructure, polarity: Polarity) -> ClaimType:
        """
        Classifies claim into one of the 12 fine-grained taxonomy types.
        """
        lower = text.lower()

        # 1. Negative Claim
        if polarity == Polarity.NEGATIVE:
            return ClaimType.NEGATIVE_CLAIM

        # 2. Out of Scope (External Knowledge / Unverifiable Specific Location)
        unverifiable_cities = ["paris", "tokyo", "london", "new york", "beijing", "berlin", "rome", "san francisco"]
        if any(re.search(r'\b' + city + r'\b', lower) for city in unverifiable_cities):
            return ClaimType.OUT_OF_SCOPE

        # 3. OCR / Embedded Text
        if any(w in lower for w in ["says", "written", "text", "reads", "sign says", "labeled"]):
            return ClaimType.OCR_TEXT

        # 4. Object Count
        count_words = ["one", "two", "three", "four", "five", "multiple", "several", "pair of", "1", "2", "3", "4", "5"]
        if any(re.search(r'\b' + cw + r'\b', lower) for cw in count_words) and "is" not in lower:
            return ClaimType.OBJECT_COUNT

        # 5. Color
        colors = ["red", "blue", "green", "yellow", "brown", "white", "black", "orange", "purple", "pink"]
        if any(re.search(r'\b' + c + r'\b', lower) for c in colors):
            return ClaimType.COLOR

        # 6. Spatial Relation
        spatial_terms = ["on top of", "under", "above", "behind", "in front of", "beside", "next to", "inside", "near"]
        if any(re.search(r'\b' + st + r'\b', lower) for st in spatial_terms) and structure.target_object:
            return ClaimType.SPATIAL_RELATION

        # 7. Interaction / Action
        interaction_verbs = ["holding", "wearing", "carrying", "pushing", "pulling", "looking at"]
        if any(re.search(r'\b' + iv + r'\b', lower) for iv in interaction_verbs) and structure.target_object:
            return ClaimType.INTERACTION

        if structure.predicate in ["riding", "running", "walking", "sitting", "standing", "flying", "eating", "driving"]:
            return ClaimType.ACTION

        # 8. Comparison
        if any(w in lower for w in ["larger than", "smaller than", "taller", "shorter", "faster", "more than"]):
            return ClaimType.COMPARISON

        # 9. Scene
        if any(w in lower for w in ["scene is", "photo showing", "environment", "background", "indoor", "outdoor"]):
            return ClaimType.SCENE

        # 10. Attribute
        if structure.modifiers:
            return ClaimType.ATTRIBUTE

        # 11. Default: Object Existence
        return ClaimType.OBJECT_EXISTENCE
