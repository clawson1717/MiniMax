"""TrajectoryGraph: stores milestones and their acceptance/prune/rectify history."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import EvidenceItem, LayerResult, Milestone, VerificationStatus


@dataclass
class TrajectoryGraph:
    """Graph of milestones produced by an agent run, with verification bookkeeping."""

    milestones: dict[str, Milestone] = field(default_factory=dict)
    root_id: str | None = None
    accepted_ids: list[str] = field(default_factory=list)
    pruned_ids: list[str] = field(default_factory=list)
    rectified_ids: list[str] = field(default_factory=list)

    def add_milestone(self, m: Milestone) -> None:
        """Register a milestone and wire it into the parent/child topology."""
        self.milestones[m.id] = m
        if m.parent_id is None and self.root_id is None:
            self.root_id = m.id
        if m.parent_id is not None:
            parent = self.milestones.get(m.parent_id)
            if parent is not None and m.id not in parent.children_ids:
                parent.children_ids.append(m.id)

    def mark_accepted(self, id: str) -> None:
        """Mark a milestone as verified and accepted into the trajectory."""
        if id not in self.milestones:
            raise KeyError(f"unknown milestone: {id}")
        self.milestones[id].verification_status = VerificationStatus.PASS
        if id not in self.accepted_ids:
            self.accepted_ids.append(id)

    def mark_pruned(self, id: str) -> None:
        """Mark a milestone as irreparable and pruned from the trajectory."""
        if id not in self.milestones:
            raise KeyError(f"unknown milestone: {id}")
        self.milestones[id].verification_status = VerificationStatus.PRUNED
        if id not in self.pruned_ids:
            self.pruned_ids.append(id)

    def mark_rectified(self, id: str) -> None:
        """Mark a milestone as having been corrected by the rectifier."""
        if id not in self.milestones:
            raise KeyError(f"unknown milestone: {id}")
        self.milestones[id].verification_status = VerificationStatus.RECTIFIED
        if id not in self.rectified_ids:
            self.rectified_ids.append(id)

    def get_lineage(self, id: str) -> list[str]:
        """Return the path of milestone ids from root to the given milestone (inclusive)."""
        if id not in self.milestones:
            raise KeyError(f"unknown milestone: {id}")
        lineage: list[str] = []
        current: str | None = id
        visited: set[str] = set()
        while current is not None:
            if current in visited:
                break  # defensive: avoid cycles
            visited.add(current)
            lineage.append(current)
            parent = self.milestones[current].parent_id
            current = parent
        lineage.reverse()
        return lineage

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the trajectory graph."""
        return {
            "root_id": self.root_id,
            "accepted_ids": list(self.accepted_ids),
            "pruned_ids": list(self.pruned_ids),
            "rectified_ids": list(self.rectified_ids),
            "milestones": {
                mid: {
                    "id": m.id,
                    "step_index": m.step_index,
                    "content": m.content,
                    "evidence_chain": [
                        {
                            "source": e.source,
                            "claim": e.claim,
                            "confidence": e.confidence,
                            "metadata": dict(e.metadata),
                        }
                        for e in m.evidence_chain
                    ],
                    "verification_status": m.verification_status.value,
                    "layer_results": {
                        name: {
                            "layer_name": r.layer_name,
                            "passed": r.passed,
                            "reason": r.reason,
                            "confidence": r.confidence,
                        }
                        for name, r in m.layer_results.items()
                    },
                    "parent_id": m.parent_id,
                    "children_ids": list(m.children_ids),
                }
                for mid, m in self.milestones.items()
            },
        }
