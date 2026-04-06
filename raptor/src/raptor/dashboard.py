"""RAPTOR Dashboard — Rich terminal-based signal visualizer.

Provides text-based visualization of RAPTOR orchestration signals:
  - Entropy trajectory display with non-monotone jump highlighting
  - Disagreement signal feature display with overall score
  - Utility scores table with selected action highlighting
  - Color-coded reliability indicators (GREEN / YELLOW / RED)
  - Step-by-step replay mode for JSONL session logs

Usage (CLI)::

    # Replay a session log
    python -m raptor.dashboard replay raptor_logs/session_20260406_120000_000000.jsonl

    # Show a single step summary
    python -m raptor.dashboard step raptor_logs/session_20260406_120000_000000.jsonl --step 3

    # List available session logs
    python -m raptor.dashboard list raptor_logs/

Usage (API)::

    from raptor.dashboard import DashboardRenderer, load_session_log
    renderer = DashboardRenderer()
    steps = load_session_log("raptor_logs/session_xxx.jsonl")
    renderer.render_step(steps[0])
    renderer.render_session(steps)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


# --------------------------------------------------------------------------
# Data structures for parsed log entries
# --------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Parsed representation of one JSONL log entry from a RAPTOR session."""

    step: int
    timestamp: str
    session_id: str
    action: str
    utility: float
    signal_vector: dict
    traj_signal: dict
    disa_signal: dict
    breakdown: dict
    reason: str
    context: Optional[dict] = None


# --------------------------------------------------------------------------
# JSONL log parsing
# --------------------------------------------------------------------------


def load_session_log(path: str | Path) -> list[StepRecord]:
    """Parse a JSONL session log file into a list of StepRecord objects.

    Args:
        path: Path to a ``session_*.jsonl`` file.

    Returns:
        List of :class:`StepRecord` in step order.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line is malformed JSON.
    """
    path = Path(path)
    records: list[StepRecord] = []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(
                StepRecord(
                    step=data.get("step", line_num),
                    timestamp=data.get("timestamp", ""),
                    session_id=data.get("session_id", ""),
                    action=data.get("action", "unknown"),
                    utility=data.get("utility", 0.0),
                    signal_vector=data.get("signal_vector", {}),
                    traj_signal=data.get("traj_signal", {}),
                    disa_signal=data.get("disa_signal", {}),
                    breakdown=data.get("breakdown", {}),
                    reason=data.get("reason", ""),
                    context=data.get("context"),
                )
            )

    records.sort(key=lambda r: r.step)
    return records


def list_session_logs(directory: str | Path) -> list[Path]:
    """List all JSONL session log files in a directory.

    Args:
        directory: Path to the log directory.

    Returns:
        Sorted list of session log file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("session_*.jsonl"))


# --------------------------------------------------------------------------
# Reliability classification
# --------------------------------------------------------------------------


class ReliabilityLevel:
    """Constants for reliability classification."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


def classify_reliability(
    monotonicity: bool,
    disagreement_score: float,
    disagreement_threshold_low: float = 0.3,
    disagreement_threshold_high: float = 0.5,
) -> str:
    """Classify reliability into GREEN / YELLOW / RED based on signals.

    Rules:
      - GREEN:  monotone AND disagreement_score < low threshold
      - RED:    non-monotone AND disagreement_score >= high threshold
      - YELLOW: everything else

    Args:
        monotonicity: True if entropy trajectory is monotonically decreasing.
        disagreement_score: 1 − confidence_score (higher = more disagreement).
        disagreement_threshold_low: Below this, disagreement is considered low.
        disagreement_threshold_high: At or above this, disagreement is high.

    Returns:
        One of ``"green"``, ``"yellow"``, ``"red"``.
    """
    if monotonicity and disagreement_score < disagreement_threshold_low:
        return ReliabilityLevel.GREEN
    if not monotonicity and disagreement_score >= disagreement_threshold_high:
        return ReliabilityLevel.RED
    return ReliabilityLevel.YELLOW


def reliability_style(level: str) -> str:
    """Map reliability level to Rich color style string.

    Args:
        level: One of ``"green"``, ``"yellow"``, ``"red"``.

    Returns:
        Rich markup style string.
    """
    return {
        ReliabilityLevel.GREEN: "bold green",
        ReliabilityLevel.YELLOW: "bold yellow",
        ReliabilityLevel.RED: "bold red",
    }.get(level, "white")


def reliability_icon(level: str) -> str:
    """Map reliability level to a text icon.

    Args:
        level: One of ``"green"``, ``"yellow"``, ``"red"``.

    Returns:
        Emoji/text indicator string.
    """
    return {
        ReliabilityLevel.GREEN: "● GREEN",
        ReliabilityLevel.YELLOW: "◐ YELLOW",
        ReliabilityLevel.RED: "○ RED",
    }.get(level, "? UNKNOWN")


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------


class DashboardRenderer:
    """Rich-based terminal renderer for RAPTOR signals.

    All rendering methods accept an optional ``console`` override;
    the default uses the instance console (which can be swapped to
    a ``Console(file=...)`` for testing / capture).
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Entropy trajectory
    # ------------------------------------------------------------------

    def render_entropy_trajectory(
        self,
        entropies: list[float],
        monotonicity: bool,
        slope: float,
        confidence: float,
    ) -> Panel:
        """Build a Rich Panel showing the entropy trajectory.

        Each step's entropy is shown with a bar-chart-like indicator.
        Non-monotone jumps (where entropy increased) are highlighted red.

        Args:
            entropies: List of entropy values per reasoning step.
            monotonicity: Whether the full trajectory is monotone.
            slope: Linear slope of the trajectory.
            confidence: Confidence score from entropy tracker.

        Returns:
            Rich :class:`Panel` object (can be printed via console).
        """
        text = Text()

        if not entropies:
            text.append("  No entropy data available.\n", style="dim")
        else:
            max_entropy = max(entropies) if entropies else 1.0
            bar_width = 30

            for i, entropy in enumerate(entropies):
                # Detect non-monotone jump
                is_jump = i > 0 and entropy > entropies[i - 1]
                bar_len = int((entropy / max(max_entropy, 1e-9)) * bar_width)
                bar = "█" * bar_len + "░" * (bar_width - bar_len)

                if is_jump:
                    style = "red"
                    marker = " ▲"
                else:
                    style = "green" if monotonicity else "cyan"
                    marker = ""

                text.append(f"  Step {i:>3d}: ", style="dim")
                text.append(f"{entropy:>7.4f} ", style=style)
                text.append(bar, style=style)
                text.append(marker, style="bold red")
                text.append("\n")

        # Summary line
        text.append("\n")
        mono_text = "Yes ✓" if monotonicity else "No ✗"
        mono_style = "green" if monotonicity else "red"
        text.append("  Monotone: ", style="bold")
        text.append(mono_text, style=mono_style)
        text.append(f"  |  Slope: {slope:+.4f}  |  Confidence: {confidence:.3f}\n")

        return Panel(
            text,
            title="[bold cyan]Entropy Trajectory[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    # ------------------------------------------------------------------
    # Disagreement signal
    # ------------------------------------------------------------------

    def render_disagreement_signal(
        self,
        disa_signal: dict,
    ) -> Panel:
        """Build a Rich Panel showing disagreement signal features.

        Args:
            disa_signal: Dict with keys: evidence_overlap, argument_strength,
                divergence_depth, dispersion, cohesion, confidence_score,
                disagreement_tier.

        Returns:
            Rich :class:`Panel` object.
        """
        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
        )
        table.add_column("Feature", style="bold", min_width=20)
        table.add_column("Value", justify="right", min_width=10)
        table.add_column("Indicator", min_width=12)

        features = [
            ("Evidence Overlap", disa_signal.get("evidence_overlap", 0.0), True),
            ("Argument Strength", disa_signal.get("argument_strength", 0.0), True),
            ("Divergence Depth", disa_signal.get("divergence_depth", 0), None),
            ("Dispersion", disa_signal.get("dispersion", 0.0), False),
            ("Cohesion", disa_signal.get("cohesion", 0.0), True),
        ]

        for name, value, higher_is_better in features:
            if isinstance(value, float):
                val_str = f"{value:.4f}"
                if higher_is_better is not None:
                    if higher_is_better:
                        indicator_style = "green" if value > 0.6 else ("yellow" if value > 0.3 else "red")
                    else:
                        indicator_style = "green" if value < 0.3 else ("yellow" if value < 0.6 else "red")
                    indicator = "●" if indicator_style == "green" else ("◐" if indicator_style == "yellow" else "○")
                    indicator_text = Text(indicator, style=indicator_style)
                else:
                    indicator_text = Text("")
            else:
                val_str = str(value)
                indicator_text = Text("")

            table.add_row(name, val_str, indicator_text)

        # Overall score row
        confidence = disa_signal.get("confidence_score", 0.0)
        tier = disa_signal.get("disagreement_tier", "unknown")
        tier_style = {"low": "green", "medium": "yellow", "weak": "red"}.get(tier, "white")

        table.add_row("", "", "")  # spacer
        table.add_row(
            Text("Confidence", style="bold"),
            f"{confidence:.4f}",
            Text(f"[{tier}]", style=tier_style),
        )

        return Panel(
            table,
            title="[bold magenta]Disagreement Signal[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        )

    # ------------------------------------------------------------------
    # Utility scores
    # ------------------------------------------------------------------

    def render_utility_scores(
        self,
        breakdown: dict[str, float],
        chosen_action: str,
        chosen_utility: float,
        all_scores: Optional[dict[str, float]] = None,
    ) -> Panel:
        """Build a Rich Panel showing utility scores for all actions.

        Args:
            breakdown: Feature contributions for the chosen action.
            chosen_action: The selected action string (e.g., "respond").
            chosen_utility: The utility score of the chosen action.
            all_scores: Optional dict mapping action names to utility scores.
                If provided, all actions are shown; otherwise only the chosen.

        Returns:
            Rich :class:`Panel` object.
        """
        table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
        )
        table.add_column("Action", style="bold", min_width=12)
        table.add_column("Utility", justify="right", min_width=10)
        table.add_column("", min_width=4)

        if all_scores:
            # Sort by utility descending
            sorted_actions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for action, utility in sorted_actions:
                is_chosen = action == chosen_action
                action_style = "bold green" if is_chosen else "white"
                marker = " ◄" if is_chosen else ""
                table.add_row(
                    Text(action, style=action_style),
                    Text(f"{utility:+.4f}", style=action_style),
                    Text(marker, style="bold green"),
                )
        else:
            table.add_row(
                Text(chosen_action, style="bold green"),
                f"{chosen_utility:+.4f}",
                Text(" ◄", style="bold green"),
            )

        # Feature breakdown for chosen action
        if breakdown:
            table.add_row("", "", "")
            table.add_row(Text("── Feature Breakdown ──", style="dim"), "", "")
            for feature, value in sorted(breakdown.items()):
                feat_style = "green" if value > 0 else ("red" if value < 0 else "dim")
                table.add_row(
                    Text(f"  {feature}", style="dim"),
                    Text(f"{value:+.4f}", style=feat_style),
                    Text(""),
                )

        return Panel(
            table,
            title="[bold blue]Utility Scores[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )

    # ------------------------------------------------------------------
    # Reliability indicator
    # ------------------------------------------------------------------

    def render_reliability(
        self,
        monotonicity: bool,
        disagreement_score: float,
    ) -> Panel:
        """Build a compact reliability indicator panel.

        Args:
            monotonicity: True if entropy trajectory is monotone.
            disagreement_score: 1 − confidence (higher = more disagreement).

        Returns:
            Rich :class:`Panel` with color-coded reliability level.
        """
        level = classify_reliability(monotonicity, disagreement_score)
        style = reliability_style(level)
        icon = reliability_icon(level)

        text = Text()
        text.append(f"  {icon}\n\n", style=style)
        text.append("  Monotone: ", style="bold")
        text.append("Yes" if monotonicity else "No", style="green" if monotonicity else "red")
        text.append("  |  Disagreement: ", style="bold")
        text.append(f"{disagreement_score:.3f}", style=style)
        text.append("\n")

        return Panel(
            text,
            title="[bold]Reliability[/bold]",
            border_style=level,
            box=box.ROUNDED,
        )

    # ------------------------------------------------------------------
    # Full step rendering
    # ------------------------------------------------------------------

    def render_step(self, record: StepRecord) -> None:
        """Render a complete step to the console.

        Displays all four panels: entropy trajectory, disagreement signal,
        utility scores, and reliability indicator.

        Args:
            record: A parsed :class:`StepRecord` from a session log.
        """
        sv = record.signal_vector
        traj = record.traj_signal
        disa = record.disa_signal

        # Header
        header = Text()
        header.append(f"  Session: {record.session_id}", style="dim")
        header.append(f"  |  Step: {record.step}", style="bold")
        header.append(f"  |  Time: {record.timestamp}", style="dim")
        if record.context and "prompt" in record.context:
            prompt_preview = record.context["prompt"][:80]
            header.append(f"\n  Prompt: {prompt_preview}...", style="dim italic")

        self.console.print(
            Panel(header, title="[bold]RAPTOR Step[/bold]", border_style="white", box=box.DOUBLE)
        )

        # Entropy trajectory panel
        entropies = traj.get("entropies", [])
        mono = traj.get("monotonicity", sv.get("monotonicity_flag", True))
        slope = traj.get("entropy_slope", sv.get("entropy_slope", 0.0))
        confidence = traj.get("confidence_score", 0.5)
        self.console.print(
            self.render_entropy_trajectory(entropies, mono, slope, confidence)
        )

        # Disagreement panel
        self.console.print(self.render_disagreement_signal(disa))

        # Utility scores panel
        # Reconstruct all_scores from signal_history if available
        all_scores_dict: Optional[dict[str, float]] = None
        # The JSONL log doesn't store all_scores directly, but we have breakdown
        # for the chosen action. If step data has them in signal_vector context, use it.
        self.console.print(
            self.render_utility_scores(
                breakdown=record.breakdown,
                chosen_action=record.action,
                chosen_utility=record.utility,
                all_scores=all_scores_dict,
            )
        )

        # Reliability indicator
        disagreement_score = sv.get("disagreement_score", 1.0 - disa.get("confidence_score", 0.5))
        self.console.print(
            self.render_reliability(mono, disagreement_score)
        )

        # Reason
        if record.reason:
            self.console.print(
                Panel(
                    Text(f"  {record.reason}", style="italic"),
                    title="[bold]Reason[/bold]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

    # ------------------------------------------------------------------
    # Session replay
    # ------------------------------------------------------------------

    def render_session(
        self,
        records: list[StepRecord],
        delay: float = 0.0,
    ) -> None:
        """Render all steps in a session sequentially.

        Args:
            records: List of :class:`StepRecord` from :func:`load_session_log`.
            delay: Seconds to pause between steps (for replay effect).
        """
        if not records:
            self.console.print("[dim]No steps to display.[/dim]")
            return

        session_id = records[0].session_id
        self.console.print(
            f"\n[bold cyan]═══ RAPTOR Session Replay: {session_id} "
            f"({len(records)} steps) ═══[/bold cyan]\n"
        )

        for i, record in enumerate(records):
            self.render_step(record)
            if delay > 0 and i < len(records) - 1:
                time.sleep(delay)

        self.console.print(
            f"\n[bold cyan]═══ End of Session ({len(records)} steps) ═══[/bold cyan]\n"
        )

    # ------------------------------------------------------------------
    # Session summary (compact overview)
    # ------------------------------------------------------------------

    def render_session_summary(self, records: list[StepRecord]) -> None:
        """Render a compact summary table for a full session.

        Shows step number, action, utility, reliability level, and
        monotonicity in a single table.

        Args:
            records: List of :class:`StepRecord` from :func:`load_session_log`.
        """
        if not records:
            self.console.print("[dim]No steps to summarize.[/dim]")
            return

        table = Table(
            title=f"Session Summary: {records[0].session_id}",
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
        )
        table.add_column("Step", justify="right", style="bold")
        table.add_column("Action", min_width=10)
        table.add_column("Utility", justify="right")
        table.add_column("Monotone")
        table.add_column("Disagreement", justify="right")
        table.add_column("Reliability")

        for record in records:
            sv = record.signal_vector
            disa = record.disa_signal
            mono = record.traj_signal.get("monotonicity", sv.get("monotonicity_flag", True))
            disag_score = sv.get("disagreement_score", 1.0 - disa.get("confidence_score", 0.5))

            level = classify_reliability(mono, disag_score)
            style = reliability_style(level)
            icon = reliability_icon(level)

            table.add_row(
                str(record.step),
                Text(record.action, style="bold"),
                f"{record.utility:+.4f}",
                Text("Yes" if mono else "No", style="green" if mono else "red"),
                f"{disag_score:.3f}",
                Text(icon, style=style),
            )

        self.console.print(table)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def _cli_main(args: Optional[list[str]] = None) -> None:
    """CLI entry point for the dashboard.

    Commands:
      replay <logfile>               Step-by-step replay of a session
      step <logfile> --step N        Show a single step
      summary <logfile>              Compact table summary
      list <directory>               List available session logs
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="raptor-dashboard",
        description="RAPTOR Signal Dashboard — Rich terminal visualizer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Dashboard commands")

    # replay
    replay_parser = subparsers.add_parser("replay", help="Replay a session log step by step")
    replay_parser.add_argument("logfile", help="Path to session JSONL log file")
    replay_parser.add_argument("--delay", type=float, default=1.0, help="Delay between steps (seconds)")

    # step
    step_parser = subparsers.add_parser("step", help="Show a single step from a session log")
    step_parser.add_argument("logfile", help="Path to session JSONL log file")
    step_parser.add_argument("--step", type=int, required=True, help="Step number to display")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Show compact session summary")
    summary_parser.add_argument("logfile", help="Path to session JSONL log file")

    # list
    list_parser = subparsers.add_parser("list", help="List available session logs")
    list_parser.add_argument("directory", help="Path to log directory")

    parsed = parser.parse_args(args)
    renderer = DashboardRenderer()

    if parsed.command == "replay":
        records = load_session_log(parsed.logfile)
        renderer.render_session(records, delay=parsed.delay)

    elif parsed.command == "step":
        records = load_session_log(parsed.logfile)
        matching = [r for r in records if r.step == parsed.step]
        if not matching:
            renderer.console.print(f"[red]Step {parsed.step} not found in log.[/red]")
            sys.exit(1)
        renderer.render_step(matching[0])

    elif parsed.command == "summary":
        records = load_session_log(parsed.logfile)
        renderer.render_session_summary(records)

    elif parsed.command == "list":
        logs = list_session_logs(parsed.directory)
        if not logs:
            renderer.console.print(f"[dim]No session logs found in {parsed.directory}[/dim]")
        else:
            for log_path in logs:
                renderer.console.print(f"  {log_path}")

    else:
        parser.print_help()


def main() -> None:
    """CLI entry point for ``raptor-dashboard`` console script."""
    _cli_main()


if __name__ == "__main__":
    main()
