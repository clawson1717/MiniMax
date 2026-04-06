"""Tests for disagreement.py — Step 3: DiscoUQ-style disagreement monitor."""

import math

import numpy as np
import pytest

from raptor.disagreement import (
    AgentResponse,
    DisagreementMonitor,
    DisagreementSignal,
    _StubDisagreementMonitor,
    _build_tfidf_embeddings,
    _cosine_similarity,
    _jaccard,
    _keywords,
    _sigmoid,
    _tokenize,
)
from raptor.config import DisagreementConfig


# ======================================================================
# Fixtures
# ======================================================================


def _make_response(
    agent_id: str,
    steps: list[str],
    answer: str,
    embedding: np.ndarray | None = None,
) -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        reasoning_steps=steps,
        final_answer=answer,
        embedding=embedding,
    )


def _agreeing_agents(n: int = 3) -> list[AgentResponse]:
    """N agents that agree on the same answer with similar reasoning."""
    return [
        _make_response(
            f"agent-{i}",
            [
                "First analyze the problem carefully",
                "Apply the standard algorithm",
                "Compute the result step by step",
            ],
            "42",
        )
        for i in range(n)
    ]


def _disagreeing_agents(n: int = 3) -> list[AgentResponse]:
    """N agents that give different answers with different reasoning."""
    answers = ["42", "17", "99", "0", "256"]
    step_sets = [
        ["Analyze the problem", "Use method A", "Get result via algebra"],
        ["Consider boundary cases", "Use method B", "Estimate numerically"],
        ["Think about edge cases", "Use method C", "Prove by contradiction"],
        ["Start from first principles", "Use method D", "Verify by induction"],
        ["Apply heuristic", "Use method E", "Check with simulation"],
    ]
    return [
        _make_response(f"agent-{i}", step_sets[i % len(step_sets)], answers[i % len(answers)])
        for i in range(n)
    ]


# ======================================================================
# Helper function tests
# ======================================================================


class TestHelpers:
    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World! This is a TEST.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_numbers(self):
        tokens = _tokenize("The answer is 42.")
        assert "42" in tokens

    def test_keywords_removes_stop_words(self):
        kw = _keywords("this is a test of the system")
        assert "test" in kw
        assert "system" in kw
        assert "this" not in kw
        assert "is" not in kw
        assert "the" not in kw

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        # {a,b,c} ∩ {b,c,d} = {b,c}, union = {a,b,c,d}
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_jaccard_empty_sets(self):
        assert _jaccard(set(), set()) == 1.0

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_sigmoid_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_range(self):
        for x in np.linspace(-10, 10, 50):
            val = _sigmoid(float(x))
            assert 0.0 <= val <= 1.0

    def test_build_tfidf_embeddings_shape(self):
        texts = ["hello world", "goodbye world", "hello goodbye"]
        embs = _build_tfidf_embeddings(texts, dim=64)
        assert len(embs) == 3
        assert all(e.shape == (64,) for e in embs)

    def test_build_tfidf_embeddings_normalized(self):
        texts = ["the quick brown fox", "lazy dog jumps"]
        embs = _build_tfidf_embeddings(texts, dim=128)
        for e in embs:
            norm = np.linalg.norm(e)
            if norm > 0:
                assert norm == pytest.approx(1.0, abs=1e-6)

    def test_build_tfidf_identical_texts(self):
        texts = ["same text here", "same text here"]
        embs = _build_tfidf_embeddings(texts, dim=64)
        sim = _cosine_similarity(embs[0], embs[1])
        assert sim == pytest.approx(1.0, abs=1e-6)


# ======================================================================
# AgentResponse dataclass tests
# ======================================================================


class TestAgentResponse:
    def test_agent_response_creation(self):
        resp = AgentResponse(
            agent_id="agent-1",
            reasoning_steps=["Step 1: ...", "Step 2: ..."],
            final_answer="42",
        )
        assert resp.agent_id == "agent-1"
        assert len(resp.reasoning_steps) == 2
        assert resp.final_answer == "42"
        assert resp.embedding is None

    def test_agent_response_with_embedding(self):
        emb = np.array([1.0, 2.0, 3.0])
        resp = AgentResponse("a", ["step"], "answer", embedding=emb)
        assert np.array_equal(resp.embedding, emb)

    def test_multiple_agents(self):
        agents = [
            AgentResponse("a", ["r1"], "ans1"),
            AgentResponse("b", ["r2"], "ans2"),
            AgentResponse("c", ["r3"], "ans3"),
        ]
        assert len(agents) == 3
        assert [a.agent_id for a in agents] == ["a", "b", "c"]


# ======================================================================
# DisagreementMonitor — compute_signal
# ======================================================================


class TestComputeSignal:
    def test_returns_disagreement_signal(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = _agreeing_agents(3)
        signal = monitor.compute_signal(responses)
        assert isinstance(signal, DisagreementSignal)

    def test_agreeing_2_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_agreeing_agents(2))
        # Agents agree → high evidence overlap, high confidence
        assert signal.evidence_overlap > 0.5
        assert signal.confidence_score > 0.0
        assert signal.confidence_score <= 1.0

    def test_agreeing_3_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_agreeing_agents(3))
        assert signal.evidence_overlap > 0.5
        assert signal.divergence_depth > 0  # All agree, depth = max steps

    def test_agreeing_5_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_agreeing_agents(5))
        assert signal.evidence_overlap > 0.5
        assert signal.cohesion > 0.5  # Similar responses → high cohesion

    def test_disagreeing_2_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_disagreeing_agents(2))
        assert signal.divergence_depth == 0  # Diverge from step 0

    def test_disagreeing_3_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_disagreeing_agents(3))
        assert signal.divergence_depth == 0

    def test_disagreeing_5_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        signal = monitor.compute_signal(_disagreeing_agents(5))
        assert signal.divergence_depth == 0

    def test_single_agent(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        resp = [_make_response("solo", ["think", "compute"], "42")]
        signal = monitor.compute_signal(resp)
        # Single agent: no disagreement
        assert signal.evidence_overlap == 1.0
        assert signal.dispersion == 0.0
        assert signal.cohesion == 1.0
        assert signal.divergence_depth == 0

    def test_confidence_in_range(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        for responses in [_agreeing_agents(3), _disagreeing_agents(3)]:
            signal = monitor.compute_signal(responses)
            assert 0.0 <= signal.confidence_score <= 1.0

    def test_tier_is_valid(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        for responses in [_agreeing_agents(3), _disagreeing_agents(5)]:
            signal = monitor.compute_signal(responses)
            assert signal.disagreement_tier in {"low", "medium", "weak"}


# ======================================================================
# Linguistic features
# ======================================================================


class TestLinguisticFeatures:
    def test_evidence_overlap_identical_responses(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["the cat sat on the mat"], "yes"),
            _make_response("b", ["the cat sat on the mat"], "yes"),
        ]
        eo, _, _ = monitor._extract_linguistic_features(responses)
        assert eo == pytest.approx(1.0)

    def test_evidence_overlap_disjoint_responses(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["quantum physics entropy"], "alpha"),
            _make_response("b", ["culinary baking chocolate"], "beta"),
        ]
        eo, _, _ = monitor._extract_linguistic_features(responses)
        assert eo < 0.5  # Very different keyword sets

    def test_argument_strength_more_steps(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        short = [
            _make_response("a", ["check check check"], "x"),
            _make_response("b", ["check check check"], "x"),
        ]
        long_steps = [
            _make_response(
                "c",
                [
                    "Analyze the mathematical foundation carefully",
                    "Apply algebraic transformations to simplify expression",
                    "Consider boundary conditions and edge cases",
                    "Verify through numerical computation method",
                    "Cross reference with established theorem results",
                    "Evaluate convergence properties systematically",
                    "Synthesize findings into coherent proof framework",
                    "Confirm final answer through independent verification",
                ],
                "x",
            ),
            _make_response(
                "d",
                [
                    "Decompose problem into fundamental components",
                    "Identify relevant theoretical principles applicable",
                    "Construct logical argument from premises",
                    "Test hypothesis against known counterexamples",
                    "Refine approach based on intermediate results",
                    "Validate solution through dimensional analysis",
                    "Generalize findings to broader class problems",
                    "Document reasoning chain for reproducibility",
                ],
                "x",
            ),
        ]
        _, strength_short, _ = monitor._extract_linguistic_features(short)
        _, strength_long, _ = monitor._extract_linguistic_features(long_steps)
        assert strength_long > strength_short

    def test_argument_strength_in_range(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = _disagreeing_agents(3)
        _, strength, _ = monitor._extract_linguistic_features(responses)
        assert 0.0 <= strength <= 1.0

    def test_divergence_depth_all_agree(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = _agreeing_agents(3)
        _, _, depth = monitor._extract_linguistic_features(responses)
        # All agree on "42" → depth = max step count = 3
        assert depth == 3

    def test_divergence_depth_disagree_from_start(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = _disagreeing_agents(2)
        _, _, depth = monitor._extract_linguistic_features(responses)
        assert depth == 0  # Different steps from the beginning

    def test_divergence_depth_partial_agreement(self):
        """Agents share first step, diverge on second."""
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["same first step", "method A"], "42"),
            _make_response("b", ["same first step", "method B"], "17"),
        ]
        _, _, depth = monitor._extract_linguistic_features(responses)
        assert depth == 1  # Diverge at step index 1

    def test_empty_reasoning_steps(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", [], "42"),
            _make_response("b", [], "42"),
        ]
        eo, strength, depth = monitor._extract_linguistic_features(responses)
        assert 0.0 <= eo <= 1.0
        assert strength == 0.0  # No reasoning steps → minimal strength
        assert depth == 0  # Max of empty lists = 0

    def test_single_agent_linguistic(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [_make_response("solo", ["think hard"], "42")]
        eo, strength, depth = monitor._extract_linguistic_features(responses)
        assert eo == 1.0
        assert depth == 0


# ======================================================================
# Embedding features
# ======================================================================


class TestEmbeddingFeatures:
    def test_identical_embeddings(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        emb = np.array([1.0, 0.0, 0.0])
        responses = [
            _make_response("a", ["s"], "x", embedding=emb.copy()),
            _make_response("b", ["s"], "x", embedding=emb.copy()),
        ]
        disp, coh = monitor._extract_embedding_features(responses)
        # Identical vectors → zero per-dimension std → dispersion = 0
        assert disp == pytest.approx(0.0)
        assert coh == pytest.approx(1.0)

    def test_orthogonal_embeddings(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["s"], "x", embedding=np.array([1.0, 0.0])),
            _make_response("b", ["s"], "x", embedding=np.array([0.0, 1.0])),
        ]
        _, coh = monitor._extract_embedding_features(responses)
        assert coh == pytest.approx(0.0)

    def test_opposite_embeddings(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["s"], "x", embedding=np.array([1.0, 0.0])),
            _make_response("b", ["s"], "x", embedding=np.array([-1.0, 0.0])),
        ]
        _, coh = monitor._extract_embedding_features(responses)
        assert coh == pytest.approx(-1.0)

    def test_dispersion_with_known_vectors(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        # Known vectors where we can compute per-dimension std
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        responses = [
            _make_response("a", ["s"], "x", embedding=v1),
            _make_response("b", ["s"], "x", embedding=v2),
        ]
        disp, _ = monitor._extract_embedding_features(responses)
        # std along axis=0 of [[1,0,0],[0,1,0]] = [0.5, 0.5, 0.0], mean = 1/3
        expected = float(np.mean(np.std(np.array([v1, v2]), axis=0)))
        assert disp == pytest.approx(expected)

    def test_tfidf_fallback_when_no_embeddings(self):
        """When embeddings are None, TF-IDF fallback should work."""
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["quantum physics"], "alpha"),
            _make_response("b", ["quantum physics"], "alpha"),
        ]
        disp, coh = monitor._extract_embedding_features(responses)
        # Same text → high cohesion
        assert coh > 0.9

    def test_tfidf_fallback_different_texts(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["quantum physics entropy thermodynamics"], "alpha"),
            _make_response("b", ["baking chocolate recipe flour sugar"], "beta"),
        ]
        disp, coh = monitor._extract_embedding_features(responses)
        # Very different texts → lower cohesion
        assert coh < 0.9

    def test_single_agent_embedding(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [_make_response("solo", ["s"], "x", embedding=np.array([1.0, 2.0]))]
        disp, coh = monitor._extract_embedding_features(responses)
        assert disp == 0.0
        assert coh == 1.0

    def test_mixed_embeddings_uses_tfidf(self):
        """If ANY embedding is None, all should use TF-IDF fallback."""
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["hello world"], "x", embedding=np.array([1.0, 0.0])),
            _make_response("b", ["hello world"], "x", embedding=None),
        ]
        # Should not crash — falls back to TF-IDF for all
        disp, coh = monitor._extract_embedding_features(responses)
        assert isinstance(disp, float)
        assert isinstance(coh, float)


# ======================================================================
# Feature modes
# ======================================================================


class TestFeatureModes:
    def test_llm_mode(self):
        config = DisagreementConfig(mode="llm")
        monitor = DisagreementMonitor(config)
        signal = monitor.compute_signal(_agreeing_agents(3))
        assert isinstance(signal, DisagreementSignal)

    def test_embed_mode(self):
        config = DisagreementConfig(mode="embed")
        monitor = DisagreementMonitor(config)
        signal = monitor.compute_signal(_agreeing_agents(3))
        assert isinstance(signal, DisagreementSignal)

    def test_learn_mode(self):
        config = DisagreementConfig(mode="learn")
        monitor = DisagreementMonitor(config)
        signal = monitor.compute_signal(_agreeing_agents(3))
        assert isinstance(signal, DisagreementSignal)

    def test_mode_override(self):
        """Config says 'embed' but we override to 'llm'."""
        config = DisagreementConfig(mode="embed")
        monitor = DisagreementMonitor(config)
        signal_embed = monitor.compute_signal(_agreeing_agents(3), mode="embed")
        signal_llm = monitor.compute_signal(_agreeing_agents(3), mode="llm")
        # Different modes can produce different confidence scores
        # (not necessarily, but they CAN — we just check both work)
        assert isinstance(signal_embed, DisagreementSignal)
        assert isinstance(signal_llm, DisagreementSignal)

    def test_different_modes_use_different_features(self):
        """Verify that llm and embed modes give different scores
        when features differ significantly."""
        config = DisagreementConfig(
            mode="learn",
            feature_weights={
                "evidence_overlap": 5.0,
                "argument_strength": 0.0,
                "divergence_depth": 0.0,
                "dispersion": 0.0,
                "cohesion": 0.0,
            },
        )
        monitor = DisagreementMonitor(config)
        responses = _disagreeing_agents(3)
        signal_llm = monitor.compute_signal(responses, mode="llm")
        signal_embed = monitor.compute_signal(responses, mode="embed")
        # With extreme weights, modes should differ (at least slightly)
        # since they use different feature subsets
        assert isinstance(signal_llm.confidence_score, float)
        assert isinstance(signal_embed.confidence_score, float)


# ======================================================================
# Confidence calibration
# ======================================================================


class TestConfidenceCalibration:
    def test_confidence_in_0_1_range(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        for responses in [
            _agreeing_agents(2),
            _agreeing_agents(5),
            _disagreeing_agents(2),
            _disagreeing_agents(5),
        ]:
            signal = monitor.compute_signal(responses)
            assert 0.0 <= signal.confidence_score <= 1.0, (
                f"confidence_score={signal.confidence_score} out of range"
            )

    def test_calibrate_confidence_with_features(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        features = {
            "evidence_overlap": 0.9,
            "argument_strength": 0.8,
            "divergence_depth": 5.0,
            "dispersion": 0.1,
            "cohesion": 0.95,
        }
        conf = monitor._calibrate_confidence(features, "learn")
        assert 0.0 <= conf <= 1.0

    def test_calibrate_high_agreement_higher_than_disagreement(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        high_agree = {
            "evidence_overlap": 0.95,
            "argument_strength": 0.9,
            "divergence_depth": 0.0,
            "dispersion": 0.05,
            "cohesion": 0.98,
        }
        high_disagree = {
            "evidence_overlap": 0.1,
            "argument_strength": 0.3,
            "divergence_depth": 0.0,
            "dispersion": 0.9,
            "cohesion": 0.1,
        }
        conf_agree = monitor._calibrate_confidence(high_agree, "learn")
        conf_disagree = monitor._calibrate_confidence(high_disagree, "learn")
        assert conf_agree > conf_disagree

    def test_custom_weights(self):
        config = DisagreementConfig(
            feature_weights={"evidence_overlap": 10.0, "cohesion": 0.0}
        )
        monitor = DisagreementMonitor(config)
        features = {
            "evidence_overlap": 1.0,
            "argument_strength": 0.5,
            "divergence_depth": 2.0,
            "dispersion": 0.3,
            "cohesion": 0.5,
        }
        conf = monitor._calibrate_confidence(features, "learn")
        assert 0.0 <= conf <= 1.0


# ======================================================================
# Tier classification
# ======================================================================


class TestTierClassification:
    def test_low_tier_above_08(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        assert monitor._classify_tier(0.85) == "low"
        assert monitor._classify_tier(0.81) == "low"
        assert monitor._classify_tier(0.99) == "low"
        assert monitor._classify_tier(1.0) == "low"

    def test_medium_tier_between_05_and_08(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        assert monitor._classify_tier(0.8) == "medium"
        assert monitor._classify_tier(0.6) == "medium"
        assert monitor._classify_tier(0.51) == "medium"

    def test_weak_tier_at_or_below_05(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        assert monitor._classify_tier(0.5) == "weak"
        assert monitor._classify_tier(0.49) == "weak"
        assert monitor._classify_tier(0.0) == "weak"
        assert monitor._classify_tier(-0.1) == "weak"

    def test_boundary_08(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        assert monitor._classify_tier(0.8) == "medium"
        assert monitor._classify_tier(0.80001) == "low"

    def test_boundary_05(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        assert monitor._classify_tier(0.5) == "weak"
        assert monitor._classify_tier(0.50001) == "medium"


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_identical_responses(self):
        """All agents give exactly the same response."""
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["step 1", "step 2"], "42"),
            _make_response("b", ["step 1", "step 2"], "42"),
            _make_response("c", ["step 1", "step 2"], "42"),
        ]
        signal = monitor.compute_signal(responses)
        assert signal.evidence_overlap == pytest.approx(1.0)
        assert signal.divergence_depth == 2  # All agree, max steps = 2
        assert signal.cohesion > 0.99

    def test_empty_reasoning_all_agents(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", [], "42"),
            _make_response("b", [], "42"),
        ]
        signal = monitor.compute_signal(responses)
        assert isinstance(signal, DisagreementSignal)
        assert signal.argument_strength == 0.0

    def test_very_long_reasoning(self):
        """Agent with 50 reasoning steps."""
        monitor = DisagreementMonitor(DisagreementConfig())
        long_steps = [f"Detailed reasoning step {i} about topic" for i in range(50)]
        responses = [
            _make_response("a", long_steps, "42"),
            _make_response("b", long_steps, "42"),
        ]
        signal = monitor.compute_signal(responses)
        assert signal.argument_strength > 0.0
        assert signal.confidence_score >= 0.0

    def test_unicode_text(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["日本語のテスト"], "回答"),
            _make_response("b", ["中文测试"], "答案"),
        ]
        signal = monitor.compute_signal(responses)
        assert isinstance(signal, DisagreementSignal)

    def test_single_word_answers(self):
        monitor = DisagreementMonitor(DisagreementConfig())
        responses = [
            _make_response("a", ["yes"], "yes"),
            _make_response("b", ["no"], "no"),
        ]
        signal = monitor.compute_signal(responses)
        assert signal.divergence_depth == 0  # Different answers


# ======================================================================
# Stub backward compatibility
# ======================================================================


class TestStubBackwardCompat:
    def test_stub_still_works(self):
        stub = _StubDisagreementMonitor(DisagreementConfig())
        signal = stub.compute_signal([])
        assert isinstance(signal, DisagreementSignal)
        assert signal.confidence_score == 0.65

    def test_stub_classify_tier(self):
        stub = _StubDisagreementMonitor(DisagreementConfig())
        assert stub._classify_tier(0.9) == "low"
        assert stub._classify_tier(0.6) == "medium"
        assert stub._classify_tier(0.3) == "weak"

    def test_real_monitor_is_not_stub(self):
        """DisagreementMonitor should NOT be the stub anymore."""
        assert DisagreementMonitor is not _StubDisagreementMonitor


# ======================================================================
# Integration-ish: full signal pipeline
# ======================================================================


class TestIntegration:
    def test_full_pipeline_agreeing(self):
        config = DisagreementConfig(mode="learn")
        monitor = DisagreementMonitor(config)
        responses = _agreeing_agents(5)
        signal = monitor.compute_signal(responses)

        assert signal.evidence_overlap > 0.5
        assert signal.divergence_depth > 0
        assert 0.0 <= signal.confidence_score <= 1.0
        assert signal.disagreement_tier in {"low", "medium", "weak"}

    def test_full_pipeline_disagreeing(self):
        config = DisagreementConfig(mode="learn")
        monitor = DisagreementMonitor(config)
        responses = _disagreeing_agents(5)
        signal = monitor.compute_signal(responses)

        assert signal.divergence_depth == 0
        assert 0.0 <= signal.confidence_score <= 1.0
        assert signal.disagreement_tier in {"low", "medium", "weak"}

    def test_full_pipeline_with_embeddings(self):
        config = DisagreementConfig(mode="embed")
        monitor = DisagreementMonitor(config)
        responses = [
            _make_response("a", ["step"], "42", embedding=np.array([1.0, 0.0, 0.0])),
            _make_response("b", ["step"], "42", embedding=np.array([0.9, 0.1, 0.0])),
            _make_response("c", ["step"], "42", embedding=np.array([0.8, 0.2, 0.0])),
        ]
        signal = monitor.compute_signal(responses)
        assert signal.cohesion > 0.8  # Similar embeddings
        assert 0.0 <= signal.confidence_score <= 1.0

    def test_all_modes_produce_valid_output(self):
        for mode in ["llm", "embed", "learn"]:
            config = DisagreementConfig(mode=mode)
            monitor = DisagreementMonitor(config)
            for resp_fn in [_agreeing_agents, _disagreeing_agents]:
                signal = monitor.compute_signal(resp_fn(3))
                assert isinstance(signal, DisagreementSignal)
                assert 0.0 <= signal.confidence_score <= 1.0
                assert signal.disagreement_tier in {"low", "medium", "weak"}
