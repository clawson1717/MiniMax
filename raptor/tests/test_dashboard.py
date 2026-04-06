"""Tests for RAPTOR Dashboard."""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from raptor.dashboard import (
    DashboardRenderer,
    StepRecord,
    ReliabilityLevel,
    classify_reliability,
    reliability_style,
    reliability_icon,
    load_session_log,
    list_session_logs,
)


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_step_record() -> StepRecord:
    """Return a sample StepRecord for testing."""
    return StepRecord(
        step=1,
        timestamp="2026-04-06T12:00:00",
        session_id="test_session_001",
        action="respond",
        utility=0.75,
        signal_vector={
            "monotonicity_flag": True,
            "entropy_slope": -0.05,
            "disagreement_score": 0.15,
            "dispersion_score": 0.2,
            "cohesion_score": 0.85,
            "divergence_depth": 2,
        },
        traj_signal={
            "n_steps": 5,
            "monotonicity": True,
            "entropy_slope": -0.05,
            "final_entropy": 0.3,
            "confidence_score": 0.85,
            "entropies": [1.2, 0.9, 0.6, 0.45, 0.3],
        },
        disa_signal={
            "evidence_overlap": 0.75,
            "argument_strength": 0.80,
            "divergence_depth": 2,
            "dispersion": 0.15,
            "cohesion": 0.88,
            "confidence_score": 0.85,
            "disagreement_tier": "low",
        },
        breakdown={
            "gain": 0.6,
            "confidence": 0.85,
            "cost_penalty": 0.0,
            "redundancy_penalty": 0.0,
            "severity": 0.1,
        },
        reason="Entropy trajectory is monotonically decreasing (good). Agent disagreement is low — high consensus.",
    )


@pytest.fixture
def sample_step_record_non_monotone() -> StepRecord:
    """Return a sample StepRecord with non-monotone trajectory."""
    return StepRecord(
        step=2,
        timestamp="2026-04-06T12:01:00",
        session_id="test_session_001",
        action="reroll",
        utility=0.55,
        signal_vector={
            "monotonicity_flag": False,
            "entropy_slope": 0.02,
            "disagreement_score": 0.55,
            "dispersion_score": 0.4,
            "cohesion_score": 0.5,
            "divergence_depth": 1,
        },
        traj_signal={
            "n_steps": 4,
            "monotonicity": False,
            "entropy_slope": 0.02,
            "final_entropy": 0.6,
            "confidence_score": 0.35,
            "entropies": [1.0, 0.7, 0.9, 0.6],  # Jump at step 2
        },
        disa_signal={
            "evidence_overlap": 0.4,
            "argument_strength": 0.5,
            "divergence_depth": 1,
            "dispersion": 0.45,
            "cohesion": 0.55,
            "confidence_score": 0.45,
            "disagreement_tier": "weak",
        },
        breakdown={
            "gain": 0.5,
            "confidence": 0.4,
            "cost_penalty": 0.1,
            "redundancy_penalty": 0.0,
            "severity": 0.3,
        },
        reason="Non-monotone entropy trajectory detected (slope=0.020). Weak agreement among agents.",
    )


@pytest.fixture
def string_console() -> tuple[Console, StringIO]:
    """Return a Console that writes to a StringIO buffer."""
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)
    return console, buffer


# --------------------------------------------------------------------------
# Reliability classification tests
# --------------------------------------------------------------------------


class TestClassifyReliability:
    """Tests for classify_reliability function."""

    def test_green_when_monotone_and_low_disagreement(self) -> None:
        """GREEN: monotone + low disagreement."""
        result = classify_reliability(True, 0.2)
        assert result == ReliabilityLevel.GREEN

    def test_green_at_threshold_boundary(self) -> None:
        """GREEN at the low threshold boundary."""
        result = classify_reliability(True, 0.29)
        assert result == ReliabilityLevel.GREEN

    def test_yellow_when_monotone_but_medium_disagreement(self) -> None:
        """YELLOW: monotone but medium disagreement."""
        result = classify_reliability(True, 0.4)
        assert result == ReliabilityLevel.YELLOW

    def test_yellow_when_non_monotone_but_low_disagreement(self) -> None:
        """YELLOW: non-monotone but low disagreement."""
        result = classify_reliability(False, 0.2)
        assert result == ReliabilityLevel.YELLOW

    def test_yellow_when_non_monotone_and_medium_disagreement(self) -> None:
        """YELLOW: non-monotone + medium disagreement (below high threshold)."""
        result = classify_reliability(False, 0.45)
        assert result == ReliabilityLevel.YELLOW

    def test_red_when_non_monotone_and_high_disagreement(self) -> None:
        """RED: non-monotone + high disagreement."""
        result = classify_reliability(False, 0.6)
        assert result == ReliabilityLevel.RED

    def test_red_at_high_threshold(self) -> None:
        """RED: non-monotone at the high threshold."""
        result = classify_reliability(False, 0.5)
        assert result == ReliabilityLevel.RED

    def test_custom_thresholds(self) -> None:
        """Custom thresholds affect classification."""
        # Default: (False, 0.4) → YELLOW
        assert classify_reliability(False, 0.4) == ReliabilityLevel.YELLOW

        # With custom thresholds: low=0.2, high=0.6
        # (False, 0.4) → YELLOW (between thresholds)
        # (False, 0.6) → RED (at high threshold)
        # (True, 0.15) → GREEN (below low threshold)
        assert classify_reliability(False, 0.4, 0.2, 0.6) == ReliabilityLevel.YELLOW
        assert classify_reliability(False, 0.6, 0.2, 0.6) == ReliabilityLevel.RED
        assert classify_reliability(True, 0.15, 0.2, 0.6) == ReliabilityLevel.GREEN


class TestReliabilityStyle:
    """Tests for reliability_style function."""

    def test_green_style(self) -> None:
        assert reliability_style("green") == "bold green"

    def test_yellow_style(self) -> None:
        assert reliability_style("yellow") == "bold yellow"

    def test_red_style(self) -> None:
        assert reliability_style("red") == "bold red"

    def test_unknown_fallback(self) -> None:
        assert reliability_style("unknown") == "white"


class TestReliabilityIcon:
    """Tests for reliability_icon function."""

    def test_green_icon(self) -> None:
        assert reliability_icon("green") == "● GREEN"

    def test_yellow_icon(self) -> None:
        assert reliability_icon("yellow") == "◐ YELLOW"

    def test_red_icon(self) -> None:
        assert reliability_icon("red") == "○ RED"

    def test_unknown_icon(self) -> None:
        assert reliability_icon("unknown") == "? UNKNOWN"


# --------------------------------------------------------------------------
# Renderer tests
# --------------------------------------------------------------------------


class TestRenderEntropyTrajectory:
    """Tests for entropy trajectory rendering."""

    def test_renders_basic_trajectory(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Basic trajectory renders without error."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        entropies = [1.5, 1.2, 0.9, 0.6, 0.3]
        panel = renderer.render_entropy_trajectory(
            entropies=entropies,
            monotonicity=True,
            slope=-0.3,
            confidence=0.9,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "Entropy Trajectory" in output
        assert "Step" in output
        assert "Monotone" in output
        assert "Yes" in output
        assert "Slope" in output
        assert "-0.3000" in output
        assert "Confidence" in output

    def test_highlights_non_monotone_jumps(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Non-monotone jumps are highlighted."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        entropies = [1.0, 0.7, 0.9, 0.6]  # Jump at step 2→3
        panel = renderer.render_entropy_trajectory(
            entropies=entropies,
            monotonicity=False,
            slope=0.05,
            confidence=0.4,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "▲" in output  # Non-monotone jump marker
        assert "No" in output  # Monotone: No

    def test_empty_trajectory(self, string_console: tuple[Console, StringIO]) -> None:
        """Empty trajectory renders gracefully."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_entropy_trajectory(
            entropies=[],
            monotonicity=True,
            slope=0.0,
            confidence=0.5,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "No entropy data" in output


class TestRenderDisagreementSignal:
    """Tests for disagreement signal rendering."""

    def test_renders_all_features(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """All features are rendered."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_disagreement_signal({
            "evidence_overlap": 0.75,
            "argument_strength": 0.80,
            "divergence_depth": 2,
            "dispersion": 0.15,
            "cohesion": 0.88,
            "confidence_score": 0.85,
            "disagreement_tier": "low",
        })

        console.print(panel)
        output = buffer.getvalue()

        assert "Disagreement Signal" in output
        assert "Evidence Overlap" in output
        assert "Argument Strength" in output
        assert "Divergence Depth" in output
        assert "Dispersion" in output
        assert "Cohesion" in output
        assert "Confidence" in output

    def test_shows_tier(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Disagreement tier is shown."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_disagreement_signal({
            "confidence_score": 0.35,
            "disagreement_tier": "weak",
        })

        console.print(panel)
        output = buffer.getvalue()

        assert "[weak]" in output


class TestRenderUtilityScores:
    """Tests for utility scores rendering."""

    def test_renders_chosen_action(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Chosen action is highlighted."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_utility_scores(
            breakdown={"gain": 0.6, "confidence": 0.85},
            chosen_action="respond",
            chosen_utility=0.75,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "respond" in output
        assert "+0.7500" in output

    def test_renders_all_scores(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """All scores shown when provided."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_utility_scores(
            breakdown={},
            chosen_action="reroll",
            chosen_utility=0.55,
            all_scores={
                "respond": 0.45,
                "reroll": 0.55,
                "verify": 0.50,
                "escalate": 0.40,
            },
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "respond" in output
        assert "reroll" in output
        assert "verify" in output
        assert "escalate" in output


class TestRenderReliability:
    """Tests for reliability indicator rendering."""

    def test_renders_green(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Green reliability renders correctly."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_reliability(
            monotonicity=True,
            disagreement_score=0.15,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "GREEN" in output
        assert "Yes" in output

    def test_renders_red(
        self, string_console: tuple[Console, StringIO]
    ) -> None:
        """Red reliability renders correctly."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        panel = renderer.render_reliability(
            monotonicity=False,
            disagreement_score=0.65,
        )

        console.print(panel)
        output = buffer.getvalue()

        assert "RED" in output
        assert "No" in output


class TestRenderStep:
    """Tests for full step rendering."""

    def test_renders_complete_step(
        self, sample_step_record: StepRecord,
        string_console: tuple[Console, StringIO]
    ) -> None:
        """Full step renders all components."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        renderer.render_step(sample_step_record)
        output = buffer.getvalue()

        assert "RAPTOR Step" in output
        assert "test_session_001" in output
        assert "Entropy Trajectory" in output
        assert "Disagreement Signal" in output
        assert "Utility Scores" in output
        assert "Reliability" in output


class TestRenderSession:
    """Tests for session rendering."""

    def test_renders_multiple_steps(
        self,
        sample_step_record: StepRecord,
        sample_step_record_non_monotone: StepRecord,
        string_console: tuple[Console, StringIO]
    ) -> None:
        """Multiple steps are rendered."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        renderer.render_session([
            sample_step_record,
            sample_step_record_non_monotone,
        ])
        output = buffer.getvalue()

        assert "Session Replay" in output
        assert "End of Session" in output

    def test_empty_session(self, string_console: tuple[Console, StringIO]) -> None:
        """Empty session renders gracefully."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        renderer.render_session([])
        output = buffer.getvalue()

        assert "No steps to display" in output


class TestRenderSessionSummary:
    """Tests for session summary rendering."""

    def test_renders_summary_table(
        self,
        sample_step_record: StepRecord,
        sample_step_record_non_monotone: StepRecord,
        string_console: tuple[Console, StringIO]
    ) -> None:
        """Summary table renders correctly."""
        console, buffer = string_console
        renderer = DashboardRenderer(console)

        renderer.render_session_summary([
            sample_step_record,
            sample_step_record_non_monotone,
        ])
        output = buffer.getvalue()

        assert "Session Summary" in output
        assert "test_session_001" in output
        assert "respond" in output
        assert "reroll" in output


# --------------------------------------------------------------------------
# JSONL parsing tests
# --------------------------------------------------------------------------


class TestLoadSessionLog:
    """Tests for JSONL log parsing."""

    def test_loads_valid_log(self, tmp_path: Path) -> None:
        """Valid JSONL log loads correctly."""
        log_file = tmp_path / "session_test.jsonl"
        log_file.write_text(json.dumps({
            "step": 1,
            "timestamp": "2026-04-06T12:00:00",
            "session_id": "test_session",
            "action": "respond",
            "utility": 0.75,
            "signal_vector": {"monotonicity_flag": True},
            "traj_signal": {"monotonicity": True},
            "disa_signal": {"confidence_score": 0.85},
            "breakdown": {},
            "reason": "test reason",
        }) + "\n")

        records = load_session_log(log_file)
        assert len(records) == 1
        assert records[0].step == 1
        assert records[0].action == "respond"
        assert records[0].utility == 0.75

    def test_loads_multiple_steps(self, tmp_path: Path) -> None:
        """Multiple steps load in order."""
        log_file = tmp_path / "session_multi.jsonl"
        entries = [
            {"step": 2, "action": "reroll", "utility": 0.5},
            {"step": 1, "action": "respond", "utility": 0.75},
            {"step": 3, "action": "verify", "utility": 0.6},
        ]
        log_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        records = load_session_log(log_file)
        assert len(records) == 3
        # Should be sorted by step
        assert records[0].step == 1
        assert records[1].step == 2
        assert records[2].step == 3

    def test_handles_missing_fields(self, tmp_path: Path) -> None:
        """Missing fields get defaults."""
        log_file = tmp_path / "session_minimal.jsonl"
        log_file.write_text(json.dumps({"step": 1}) + "\n")

        records = load_session_log(log_file)
        assert len(records) == 1
        assert records[0].action == "unknown"  # default
        assert records[0].utility == 0.0  # default

    def test_handles_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines are skipped."""
        log_file = tmp_path / "session_empty_lines.jsonl"
        log_file.write_text("\n\n" + json.dumps({"step": 1}) + "\n\n")

        records = load_session_log(log_file)
        assert len(records) == 1


class TestListSessionLogs:
    """Tests for session log listing."""

    def test_lists_jsonl_files(self, tmp_path: Path) -> None:
        """Lists session_*.jsonl files."""
        (tmp_path / "session_001.jsonl").touch()
        (tmp_path / "session_002.jsonl").touch()
        (tmp_path / "other_file.txt").touch()

        logs = list_session_logs(tmp_path)
        assert len(logs) == 2
        names = [p.name for p in logs]
        assert "session_001.jsonl" in names
        assert "session_002.jsonl" in names
        assert "other_file.txt" not in names

    def test_returns_empty_for_nonexistent(self) -> None:
        """Nonexistent directory returns empty list."""
        logs = list_session_logs("/nonexistent/path")
        assert logs == []

    def test_returns_sorted(self, tmp_path: Path) -> None:
        """Returns sorted list."""
        (tmp_path / "session_zzz.jsonl").touch()
        (tmp_path / "session_aaa.jsonl").touch()
        (tmp_path / "session_mmm.jsonl").touch()

        logs = list_session_logs(tmp_path)
        names = [p.name for p in logs]
        assert names == ["session_aaa.jsonl", "session_mmm.jsonl", "session_zzz.jsonl"]


# --------------------------------------------------------------------------
# CLI tests (integration)
# --------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI interface."""

    def test_cli_step_command(self, tmp_path: Path) -> None:
        """CLI step command works."""
        from raptor.dashboard import _cli_main

        log_file = tmp_path / "session_test.jsonl"
        log_file.write_text(json.dumps({
            "step": 1,
            "action": "respond",
            "utility": 0.75,
            "signal_vector": {},
            "traj_signal": {"entropies": [1.0, 0.5], "monotonicity": True},
            "disa_signal": {"confidence_score": 0.85},
            "breakdown": {},
            "reason": "test",
        }) + "\n")

        # Should not raise
        _cli_main(["step", str(log_file), "--step", "1"])

    def test_cli_summary_command(self, tmp_path: Path) -> None:
        """CLI summary command works."""
        from raptor.dashboard import _cli_main

        log_file = tmp_path / "session_test.jsonl"
        log_file.write_text(json.dumps({
            "step": 1,
            "action": "respond",
            "utility": 0.75,
            "signal_vector": {},
            "traj_signal": {},
            "disa_signal": {},
            "breakdown": {},
            "reason": "test",
        }) + "\n")

        # Should not raise
        _cli_main(["summary", str(log_file)])

    def test_cli_list_command(self, tmp_path: Path) -> None:
        """CLI list command works."""
        from raptor.dashboard import _cli_main

        (tmp_path / "session_001.jsonl").touch()

        # Should not raise
        _cli_main(["list", str(tmp_path)])
