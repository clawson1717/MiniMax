"""Tests for CycleConsistencyProber using MockLLMClient."""

from mirror.llm import MockLLMClient
from mirror.models import CoTSample
from mirror.prober import CycleConsistencyProber


class TestCycleConsistencyProber:
    """Tests for the cycle consistency prober."""

    def _make_consistent_sample(self) -> CoTSample:
        return CoTSample(
            input_prompt="What is 23 * 47?",
            cot_explanation="First I multiply 20*47=940, then 3*47=141, total=1081",
            answer="1081",
            model_id="gpt-4",
        )

    def _make_inconsistent_sample(self) -> CoTSample:
        return CoTSample(
            input_prompt="What is 23 * 47? This is wrong and inconsistent.",
            cot_explanation="I just guessed. This is wrong and inconsistent.",
            answer="999",
            model_id="gpt-4",
        )

    def test_probe_returns_probe_result(self) -> None:
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_consistent_sample()
        result = prober.probe(sample)

        assert result.sample is sample
        assert "forward" in result.scores
        assert "reverse" in result.scores
        assert "cross_model" in result.scores

    def test_consistent_sample_scores_high(self) -> None:
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_consistent_sample()
        result = prober.probe(sample)

        assert result.forward_consistent is True
        assert result.reverse_consistent is True
        assert result.cross_model_consistent is True
        assert result.scores["forward"] >= 0.7
        assert result.scores["reverse"] >= 0.7
        assert result.scores["cross_model"] >= 0.7
        assert result.overall_score >= 0.7
        assert result.is_faithful is True

    def test_inconsistent_sample_scores_low(self) -> None:
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_inconsistent_sample()
        result = prober.probe(sample)

        assert result.forward_consistent is False
        assert result.reverse_consistent is False
        assert result.cross_model_consistent is False
        assert result.scores["forward"] <= 0.5
        assert result.scores["reverse"] <= 0.5
        assert result.scores["cross_model"] <= 0.5
        assert result.is_faithful is False

    def test_cross_model_with_separate_client(self) -> None:
        primary = MockLLMClient(default_model="primary-model")
        cross = MockLLMClient(default_model="cross-model")
        prober = CycleConsistencyProber(llm=primary, cross_model_llm=cross)
        sample = self._make_consistent_sample()
        result = prober.probe(sample)

        # Primary should have forward + reverse calls
        assert len(primary.call_log) == 2
        # Cross-model client should have 1 call
        assert len(cross.call_log) == 1
        assert cross.call_log[0]["model"] == "cross-model"

    def test_details_populated(self) -> None:
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_consistent_sample()
        result = prober.probe(sample)

        assert "forward" in result.details
        assert "reverse" in result.details
        assert "cross_model" in result.details

        # Each detail should have the expected keys
        assert "raw_response" in result.details["forward"]
        assert "raw_response" in result.details["reverse"]
        assert "raw_response" in result.details["cross_model"]

    def test_score_clamping(self) -> None:
        """Scores should always be between 0.0 and 1.0."""
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_consistent_sample()
        result = prober.probe(sample)

        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_call_log_recorded(self) -> None:
        llm = MockLLMClient()
        prober = CycleConsistencyProber(llm=llm)
        sample = self._make_consistent_sample()
        prober.probe(sample)

        # Should have exactly 3 calls: forward, reverse, cross-model
        assert len(llm.call_log) == 3
