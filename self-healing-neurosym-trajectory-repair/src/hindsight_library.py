"""HeRL-inspired hindsight experience library for SNTR."""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional
import structlog

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_EMBEDDING_DEPS = True
except ImportError:
    HAS_EMBEDDING_DEPS = False

from .data_structures import HindsightExperience, TrajectoryPatch

logger = structlog.get_logger()


class HindsightExperienceLibrary:
    """
    HeRL-inspired experience replay for SNTR.
    
    Stores repaired reasoning failures with their Isabelle proofs.
    Uses embedding similarity to retrieve similar past repairs,
    enabling warm-start of proof search.
    
    Capacity-limited: evicts low-score experiences when full.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._experiences: list[HindsightExperience] = []
        
        if HAS_EMBEDDING_DEPS:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            d = 384  # embedding dimension for all-MiniLM-L6-v2
            self._index = faiss.IndexFlatL2(d)
        else:
            self._encoder = None
            self._index = None

    def add(self, patch: TrajectoryPatch) -> str:
        """Add a repaired trajectory to the experience library."""
        experience_id = str(uuid.uuid4())
        
        embedding = []
        if self._encoder:
            try:
                embedding = self._encoder.encode(
                    [patch.original_step + " " + patch.hol_theorem]
                )[0].tolist()
            except Exception:
                pass

        experience = HindsightExperience(
            id=experience_id,
            failed_claim=patch.original_step,
            hol_term=patch.hol_theorem,
            proof=patch.proof_trace,
            corrected_reasoning=patch.corrected_step,
            embedding=embedding,
        )

        self._experiences.append(experience)

        if self._index is not None and embedding:
            self._index.add(np.array([embedding]).astype("float32"))

        # Evict low-score experiences if over capacity
        if len(self._experiences) > self.capacity:
            self._evict_low_score()

        logger.info(
            "sntr.experience.add",
            id=experience_id,
            total=len(self._experiences),
        )
        
        return experience_id

    def retrieve(self, query: str, k: int = 3) -> list[HindsightExperience]:
        """Retrieve k most similar past experiences to the given query."""
        if not self._experiences or self._index is None:
            return []

        try:
            query_embedding = self._encoder.encode([query])[0]
            distances, indices = self._index.search(
                np.array([query_embedding]).astype("float32"), k
            )
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self._experiences):
                    results.append(self._experiences[idx])
            
            logger.info("sntr.experience.retrieve", query=query[:50], hits=len(results))
            return results
            
        except Exception as e:
            logger.warning("sntr.experience.retrieve_failed", error=str(e))
            return []

    def update_scores(self, experience_id: str, success: bool) -> None:
        """Update success/failure counts after a retrieval attempt."""
        for exp in self._experiences:
            if exp.id == experience_id:
                if success:
                    exp.success_count += 1
                else:
                    exp.failure_count += 1
                break

    def _evict_low_score(self) -> None:
        """Evict experiences with lowest retrieval scores when at capacity."""
        sorted_experiences = sorted(
            self._experiences,
            key=lambda e: (e.retrieval_score(), e.success_count),
        )
        # Remove bottom 10%
        evict_count = max(1, len(sorted_experiences) // 10)
        for exp in sorted_experiences[:evict_count]:
            self._experiences.remove(exp)
            logger.info("sntr.experience.evict", id=exp.id)

    def stats(self) -> dict:
        """Return library statistics."""
        total = len(self._experiences)
        if total == 0:
            return {"total": 0, "avg_success_rate": 0.0, "total_uses": 0}

        total_successes = sum(e.success_count for e in self._experiences)
        total_uses = sum(e.success_count + e.failure_count for e in self._experiences)

        return {
            "total": total,
            "avg_success_rate": total_successes / total if total > 0 else 0.0,
            "total_uses": total_uses,
            "capacity": self.capacity,
        }
