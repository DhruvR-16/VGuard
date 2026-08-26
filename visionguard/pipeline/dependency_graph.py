"""
Claim Dependency Graph Module for VisionGuard.
Builds a Directed Acyclic Graph (DAG) over atomic claims linking attributes, actions,
and spatial relations to their parent entity claims to prevent cross-entity evidence leakage.
"""

from typing import List, Dict, Set, Optional
from visionguard.schemas import AtomicClaim, ClaimType, Verdict, ClaimVerificationResult


class ClaimDependencyGraph:
    """
    Manages dependency trees over atomic claims.
    Ensures attributes and actions are anchored to their verified parent entities.
    """

    def __init__(self, claims: Optional[List[AtomicClaim]] = None):
        self.claims: Dict[str, AtomicClaim] = {c.claim_id: c for c in (claims or [])}
        self.adj_list: Dict[str, List[str]] = {c.claim_id: [] for c in (claims or [])}
        self.parent_map: Dict[str, List[str]] = {c.claim_id: [] for c in (claims or [])}

        if claims:
            self.build_graph(claims)

    def build_graph(self, claims: List[AtomicClaim]) -> None:
        """
        Infers dependencies between entity existence claims and their attributes/actions.
        """
        self.claims = {c.claim_id: c for c in claims}
        self.adj_list = {c.claim_id: [] for c in claims}
        self.parent_map = {c.claim_id: [] for c in claims}

        entity_claims: Dict[str, str] = {} # subject_text -> claim_id

        # Pass 1: Identify entity existence claims
        for claim in claims:
            if claim.type == ClaimType.OBJECT_EXISTENCE and claim.linguistic_structure:
                subj = claim.linguistic_structure.subject.lower()
                entity_claims[subj] = claim.claim_id

        # Pass 2: Link attributes, actions, and relations to parent entities
        for claim in claims:
            if not claim.linguistic_structure:
                continue

            subj = claim.linguistic_structure.subject.lower()
            obj = claim.linguistic_structure.target_object.lower() if claim.linguistic_structure.target_object else None

            # If this is a dependent claim (color, attribute, action, relation)
            if claim.type in [ClaimType.COLOR, ClaimType.ATTRIBUTE, ClaimType.ACTION, ClaimType.INTERACTION, ClaimType.SPATIAL_RELATION]:
                # Link to subject parent entity if exists and distinct
                if subj in entity_claims and entity_claims[subj] != claim.claim_id:
                    parent_id = entity_claims[subj]
                    if parent_id not in claim.dependencies:
                        claim.dependencies.append(parent_id)
                    self.adj_list[parent_id].append(claim.claim_id)
                    self.parent_map[claim.claim_id].append(parent_id)

                # Link to target object parent entity if exists and distinct
                if obj and obj in entity_claims and entity_claims[obj] != claim.claim_id:
                    parent_id = entity_claims[obj]
                    if parent_id not in claim.dependencies:
                        claim.dependencies.append(parent_id)
                    self.adj_list[parent_id].append(claim.claim_id)
                    self.parent_map[claim.claim_id].append(parent_id)

    def get_parent_claims(self, claim_id: str) -> List[AtomicClaim]:
        """Returns the parent claims that this claim depends on."""
        parent_ids = self.parent_map.get(claim_id, [])
        return [self.claims[pid] for pid in parent_ids if pid in self.claims]

    def get_child_claims(self, claim_id: str) -> List[AtomicClaim]:
        """Returns the child claims that depend on this claim."""
        child_ids = self.adj_list.get(claim_id, [])
        return [self.claims[cid] for cid in child_ids if cid in self.claims]

    def propagate_contradictions(self, results: List[ClaimVerificationResult]) -> List[ClaimVerificationResult]:
        """
        If a parent entity is CONTRADICTED, dependent child attributes/actions cannot be confirmed.
        Cascades failure appropriately.
        """
        res_map = {r.claim.claim_id: r for r in results}
        contradicted_ids: Set[str] = {r.claim.claim_id for r in results if r.verdict == Verdict.CONTRADICTED}

        for r in results:
            parents = self.parent_map.get(r.claim.claim_id, [])
            if any(pid in contradicted_ids for pid in parents):
                if r.verdict == Verdict.SUPPORTED:
                    r.verdict = Verdict.INSUFFICIENT_EVIDENCE
                    r.explanation = f"Parent entity ({', '.join(parents)}) was contradicted; dependent attribute cannot be grounded."

        return results

    def get_verified_subgraph(self, results: List[ClaimVerificationResult]) -> List[ClaimVerificationResult]:
        """Returns only the claims that are SUPPORTED and have all parents SUPPORTED."""
        supported_ids = {r.claim.claim_id for r in results if r.verdict == Verdict.SUPPORTED}
        valid_claims = []

        for r in results:
            if r.verdict == Verdict.SUPPORTED:
                parents = self.parent_map.get(r.claim.claim_id, [])
                if all(pid in supported_ids for pid in parents):
                    valid_claims.append(r)

        return valid_claims
