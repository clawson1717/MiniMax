"""Disagreement Monitor — DiscoUQ-style disagreement feature extraction for multi-agent ensembles.

Implements: arXiv:2603.20975 — Structured Disagreement Analysis for Uncertainty Quantification

Three methods:
  - DiscoUQ-LLM:    logistic regression on LLM-extracted linguistic features
  - DiscoUQ-Embed:  logistic regression on embedding geometry features
  - DiscoUQ-Learn:  neural network combining all features

Key insight: Weak disagreement tier (where simple vote counting fails) gets the biggest improvement.
AUROC 0.802 on 5-agent Qwen3.5-27B system, ECE 0.036 vs 0.098 for naive voting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np


@dataclass
class DisagreementSignal:
    """Signal emitted by the Disagreement Monitor after processing agent responses."""

    # Linguistic features (DiscoUQ-LLM)
    evidence_overlap: float  # 0-1: fraction of shared evidence across agents
    argument_strength: float  # 0-1: average strength of each agent's argument
    divergence_depth: int  # How many reasoning steps before agents diverged

    # Embedding geometry features (DiscoUQ-Embed)
    dispersion: float  # Spread of agent response embeddings (std dev)
    cohesion: float  # Intra-cluster closeness (avg pairwise cosine sim)

    # Combined calibration score
    confidence_score: float  # 0-1: calibrated confidence (from logistic regression)
    disagreement_tier: str  # "low" | "medium" | "weak" (where voting fails)


@dataclass
class AgentResponse:
    """A single agent's response to a prompt."""

    agent_id: str
    reasoning_steps: list[str]  # Intermediate reasoning steps
    final_answer: str
    embedding: Optional[np.ndarray] = None  # Response embedding (for geometry features)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while it its i me my we our they them their he "
    "him his she her you your this that these those".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization, stripping non-alpha noise."""
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    """Extract keyword set (tokens minus stop words)."""
    return {t for t in _tokenize(text) if t not in _STOP_WORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _build_tfidf_embeddings(texts: list[str], dim: int = 256) -> list[np.ndarray]:
    """Build simple TF-IDF-like bag-of-words embeddings (no external deps).

    Uses hashed features so embedding dimension is fixed and we don't need
    a vocabulary pass. This is NOT production quality — it's a lightweight
    stand-in for when no real embeddings are supplied.
    """
    embeddings: list[np.ndarray] = []
    # Collect document frequencies for IDF
    doc_tokens = [_tokenize(t) for t in texts]
    n_docs = len(texts)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for unique_token in set(tokens):
            df[unique_token] += 1

    for tokens in doc_tokens:
        vec = np.zeros(dim, dtype=np.float64)
        tf: Counter[str] = Counter(tokens)
        for token, count in tf.items():
            idf = math.log((n_docs + 1) / (df.get(token, 0) + 1)) + 1.0
            # Hash token to a bucket
            idx = hash(token) % dim
            vec[idx] += count * idf
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        embeddings.append(vec)
    return embeddings


# --------------------------------------------------------------------------
# Default feature weights
# --------------------------------------------------------------------------

_DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "evidence_overlap": 2.0,
    "argument_strength": 1.5,
    "divergence_depth": -0.3,
    "dispersion": -2.5,
    "cohesion": 2.0,
}


# --------------------------------------------------------------------------
# DisagreementMonitor — the real implementation
# --------------------------------------------------------------------------


class DisagreementMonitor:
    """Monitors multi-agent disagreement using DiscoUQ-style feature extraction.

    Usage:
        monitor = DisagreementMonitor(config)
        responses = [AgentResponse(...), ...]
        signal = monitor.compute_signal(responses)
    """

    def __init__(self, config: "DisagreementConfig") -> None:
        self.config = config
        # Merge user-supplied weights with defaults
        self._weights = dict(_DEFAULT_FEATURE_WEIGHTS)
        if config.feature_weights:
            self._weights.update(config.feature_weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_signal(
        self, responses: list[AgentResponse], mode: Optional[str] = None
    ) -> DisagreementSignal:
        """Compute disagreement signal from multi-agent responses.

        Args:
            responses: List of agent responses
            mode: Override mode ("llm" | "embed" | "learn"), else uses config

        Returns:
            DisagreementSignal with linguistic + geometry features + calibrated score
        """
        mode = mode or self.config.mode

        # Always extract both feature sets (cheap); mode controls calibration.
        evidence_overlap, argument_strength, divergence_depth = (
            self._extract_linguistic_features(responses)
        )
        dispersion, cohesion = self._extract_embedding_features(responses)

        features = {
            "evidence_overlap": evidence_overlap,
            "argument_strength": argument_strength,
            "divergence_depth": float(divergence_depth),
            "dispersion": dispersion,
            "cohesion": cohesion,
        }

        confidence = self._calibrate_confidence(features, mode)
        tier = self._classify_tier(confidence)

        return DisagreementSignal(
            evidence_overlap=evidence_overlap,
            argument_strength=argument_strength,
            divergence_depth=divergence_depth,
            dispersion=dispersion,
            cohesion=cohesion,
            confidence_score=confidence,
            disagreement_tier=tier,
        )

    # ------------------------------------------------------------------
    # Linguistic features (DiscoUQ-LLM)
    # ------------------------------------------------------------------

    def _extract_linguistic_features(
        self, responses: list[AgentResponse]
    ) -> tuple[float, float, int]:
        """Extract evidence_overlap, argument_strength, divergence_depth.

        Evidence overlap: Jaccard similarity of keyword evidence sets per agent pair,
                          averaged across all pairs.

        Argument strength: Heuristic based on reasoning step count and
                           vocabulary diversity (no real LLM needed).

        Divergence depth: First step index where agent final answers diverge.

        Uses keyword overlap heuristics (no external LLM call).
        """
        if len(responses) < 2:
            # Single agent — no disagreement possible
            return (1.0, self._single_agent_strength(responses), 0)

        # --- evidence overlap ---
        keyword_sets = []
        for resp in responses:
            combined_text = " ".join(resp.reasoning_steps) + " " + resp.final_answer
            keyword_sets.append(_keywords(combined_text))

        pair_jaccards = [
            _jaccard(keyword_sets[i], keyword_sets[j])
            for i, j in combinations(range(len(responses)), 2)
        ]
        evidence_overlap = float(np.mean(pair_jaccards)) if pair_jaccards else 1.0

        # --- argument strength ---
        strengths: list[float] = []
        for resp in responses:
            n_steps = len(resp.reasoning_steps)
            # Vocabulary diversity: unique tokens / total tokens in reasoning
            all_text = " ".join(resp.reasoning_steps)
            tokens = _tokenize(all_text)
            if tokens:
                vocab_diversity = len(set(tokens)) / len(tokens)
            else:
                vocab_diversity = 0.0
            # Heuristic: more steps + higher diversity ⇒ stronger argument
            # Normalize step count contribution (diminishing returns past 10)
            step_score = min(n_steps / 10.0, 1.0)
            strength = 0.5 * step_score + 0.5 * vocab_diversity
            strengths.append(min(strength, 1.0))
        argument_strength = float(np.mean(strengths))

        # --- divergence depth ---
        divergence_depth = self._compute_divergence_depth(responses)

        return (evidence_overlap, argument_strength, divergence_depth)

    def _single_agent_strength(self, responses: list[AgentResponse]) -> float:
        """Argument strength for a single agent (or empty list)."""
        if not responses:
            return 0.0
        resp = responses[0]
        tokens = _tokenize(" ".join(resp.reasoning_steps))
        if not tokens:
            return 0.0
        vocab_diversity = len(set(tokens)) / len(tokens)
        step_score = min(len(resp.reasoning_steps) / 10.0, 1.0)
        return min(0.5 * step_score + 0.5 * vocab_diversity, 1.0)

    @staticmethod
    def _compute_divergence_depth(responses: list[AgentResponse]) -> int:
        """First step index at which agents' final answers start to diverge.

        If all agents agree, returns the max step count.
        If they disagree from the start, returns 0.
        """
        if len(responses) < 2:
            return 0

        # Normalise final answers for comparison
        answers = [r.final_answer.strip().lower() for r in responses]
        if len(set(answers)) <= 1:
            # All agree — divergence depth = max reasoning length
            return max(len(r.reasoning_steps) for r in responses)

        # Walk step-by-step: find the first index where not all agents share
        # the same step text (rough proxy for divergence point).
        max_steps = max(len(r.reasoning_steps) for r in responses)
        for step_idx in range(max_steps):
            step_texts: set[str] = set()
            for r in responses:
                if step_idx < len(r.reasoning_steps):
                    step_texts.add(r.reasoning_steps[step_idx].strip().lower())
                else:
                    step_texts.add("")  # agent ran out of steps
            if len(step_texts) > 1:
                return step_idx
        return max_steps

    # ------------------------------------------------------------------
    # Embedding geometry features (DiscoUQ-Embed)
    # ------------------------------------------------------------------

    def _extract_embedding_features(
        self, responses: list[AgentResponse]
    ) -> tuple[float, float]:
        """Extract dispersion and cohesion from agent response embeddings.

        Dispersion: std dev of embedding vectors (measures spread).
        Cohesion: mean pairwise cosine similarity between agent embeddings.

        If embeddings are None, builds simple TF-IDF hashed embeddings.
        """
        if len(responses) < 2:
            return (0.0, 1.0)

        # Gather / generate embeddings
        embeddings: list[np.ndarray] = []
        needs_generation = any(r.embedding is None for r in responses)

        if needs_generation:
            texts = [
                " ".join(r.reasoning_steps) + " " + r.final_answer for r in responses
            ]
            embeddings = _build_tfidf_embeddings(texts)
        else:
            embeddings = [r.embedding for r in responses]  # type: ignore[misc]

        mat = np.array(embeddings, dtype=np.float64)

        # --- dispersion: mean std dev across agents per dimension ---
        # This measures how much agents vary on each embedding dimension
        dispersion = float(np.mean(np.std(mat, axis=0)))

        # --- cohesion: mean pairwise cosine similarity ---
        n = len(embeddings)
        sims: list[float] = []
        for i, j in combinations(range(n), 2):
            sims.append(_cosine_similarity(mat[i], mat[j]))
        cohesion = float(np.mean(sims)) if sims else 1.0

        return (dispersion, cohesion)

    # ------------------------------------------------------------------
    # Confidence calibration
    # ------------------------------------------------------------------

    def _calibrate_confidence(
        self, features: dict[str, float], mode: str
    ) -> float:
        """Apply sigmoid-based calibration model to features.

        mode="llm":   Use only linguistic features.
        mode="embed": Use only embedding features.
        mode="learn": Use all features combined (weighted sum).

        Returns:
            Calibrated confidence score 0-1.
        """
        linguistic_keys = {"evidence_overlap", "argument_strength", "divergence_depth"}
        embedding_keys = {"dispersion", "cohesion"}

        if mode == "llm":
            active_keys = linguistic_keys
        elif mode == "embed":
            active_keys = embedding_keys
        elif mode == "learn":
            active_keys = linguistic_keys | embedding_keys
        else:
            active_keys = linguistic_keys | embedding_keys  # default to all

        # Weighted linear combination → sigmoid
        z = 0.0
        for key in active_keys:
            w = self._weights.get(key, 0.0)
            z += w * features.get(key, 0.0)

        # Bias term to center the sigmoid around ~0.5 for middling inputs
        bias = self._weights.get("bias", -1.5)
        z += bias

        return _sigmoid(z)

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------

    def _classify_tier(self, disagreement_score: float) -> str:
        """Classify disagreement into tiers.

        - low:      score > 0.8     (agents strongly agree)
        - medium:   0.5 < score ≤ 0.8
        - weak:     score ≤ 0.5     (where vote-counting fails — RAPTOR's target)
        """
        if disagreement_score > 0.8:
            return "low"
        elif disagreement_score > 0.5:
            return "medium"
        return "weak"


# ----------------------------------------------------------------------
# Stub (kept for backward compatibility)
# ----------------------------------------------------------------------
class _StubDisagreementMonitor:
    """Stub used during Step 1 (scaffold only). Kept for backward compatibility."""

    def __init__(self, config: "DisagreementConfig") -> None:
        self.config = config

    def compute_signal(
        self, responses: list[AgentResponse], mode: Optional[str] = None
    ) -> DisagreementSignal:
        return DisagreementSignal(
            evidence_overlap=0.6,
            argument_strength=0.7,
            divergence_depth=3,
            dispersion=0.4,
            cohesion=0.5,
            confidence_score=0.65,
            disagreement_tier="medium",
        )

    def _classify_tier(self, disagreement_score: float) -> str:
        if disagreement_score > 0.8:
            return "low"
        elif disagreement_score > 0.5:
            return "medium"
        return "weak"
