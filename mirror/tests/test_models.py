"""Tests for MIRROR data models."""

from mirror.models import CoTSample, FaithfulnessReport, ProbeResult


class TestCoTSample:
    """Tests for CoTSample dataclass."""

    def test_creation(self) -> None:
        sample = CoTSample(
            input_prompt="What is 2 + 2?",
            cot_explanation="2 plus 2 equals 4",
            answer="4",
            model_id="gpt-4",
        )
        assert sample.input_prompt == "What is 2 + 2?"
        assert sample.cot_explanation == "2 plus 2 equals 4"
        assert sample.answer == "4"
        assert sample.model_id == "gpt-4"

    def test_default_model_id(self) -> None:
        sample = CoTSample(
            input_prompt="test",
            cot_explanation="test",
            answer="test",
        )
        assert sample.model_id == "unknown"

    def test_frozen(self) -> None:
        sample = CoTSample(
            input_prompt="test",
            cot_explanation="test",
            answer="test",
        )
        try:
            sample.answer = "changed"  # type: ignore[misc]
            assert False, "Should not allow mutation"
        except AttributeError:
            pass  # expected — frozen dataclass


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def _make_sample(self) -> CoTSample:
        return CoTSample(
            input_prompt="What is 23 * 47?",
            cot_explanation="20*47=940, 3*47=141, total=1081",
            answer="1081",
            model_id="gpt-4",
        )

    def test_overall_score_empty(self) -> None:
        result = ProbeResult(sample=self._make_sample())
        assert result.overall_score == 0.0

    def test_overall_score_computed(self) -> None:
        result = ProbeResult(
            sample=self._make_sample(),
            scores={"forward": 0.9, "reverse": 0.8, "cross_model": 1.0},
        )
        assert abs(result.overall_score - 0.9) < 0.01

    def test_is_faithful_true(self) -> None:
        result = ProbeResult(
            sample=self._make_sample(),
            scores={"forward": 0.9, "reverse": 0.8, "cross_model": 1.0},
        )
        assert result.is_faithful is True

    def test_is_faithful_false(self) -> None:
        result = ProbeResult(
            sample=self._make_sample(),
            scores={"forward": 0.2, "reverse": 0.3, "cross_model": 0.1},
        )
        assert result.is_faithful is False

    def test_consistency_flags(self) -> None:
        result = ProbeResult(
            sample=self._make_sample(),
            forward_consistent=True,
            reverse_consistent=False,
            cross_model_consistent=True,
        )
        assert result.forward_consistent is True
        assert result.reverse_consistent is False
        assert result.cross_model_consistent is True


class TestFaithfulnessReport:
    """Tests for FaithfulnessReport dataclass."""

    def _make_result(self, forward: float, reverse: float, cross: float) -> ProbeResult:
        sample = CoTSample(
            input_prompt="test",
            cot_explanation="test",
            answer="test",
        )
        return ProbeResult(
            sample=sample,
            scores={"forward": forward, "reverse": reverse, "cross_model": cross},
        )

    def test_empty_report(self) -> None:
        report = FaithfulnessReport()
        assert report.overall_score == 0.0
        assert report.aggregate_scores == {}
        assert report.num_faithful == 0
        assert report.faithfulness_rate == 0.0

    def test_aggregate_scores(self) -> None:
        report = FaithfulnessReport(
            results=[
                self._make_result(0.8, 0.6, 1.0),
                self._make_result(0.4, 0.8, 0.6),
            ]
        )
        agg = report.aggregate_scores
        assert abs(agg["forward"] - 0.6) < 0.01
        assert abs(agg["reverse"] - 0.7) < 0.01
        assert abs(agg["cross_model"] - 0.8) < 0.01

    def test_faithfulness_rate(self) -> None:
        report = FaithfulnessReport(
            results=[
                self._make_result(0.9, 0.8, 1.0),  # overall 0.9 → faithful
                self._make_result(0.1, 0.2, 0.1),  # overall ~0.13 → not faithful
            ]
        )
        assert report.num_faithful == 1
        assert abs(report.faithfulness_rate - 0.5) < 0.01

    def test_metadata(self) -> None:
        report = FaithfulnessReport(metadata={"model": "gpt-4", "task": "gsm8k"})
        assert report.metadata["model"] == "gpt-4"
