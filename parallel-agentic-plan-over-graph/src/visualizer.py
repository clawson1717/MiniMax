"""Graph visualization module for PAPoG.

Exports TaskGraph state to DOT, PNG, and SVG formats with status-based
node coloring.  Also renders execution timelines from BenchmarkResult.

DOT text export always works.  PNG/SVG require the ``graphviz`` Python
package *and* the system ``dot`` binary — their absence is handled
gracefully with clear error messages.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from src.models import TaskGraph, TaskStatus

if TYPE_CHECKING:
    from src.benchmark import BenchmarkResult

# ---------------------------------------------------------------------------
# Status → colour mapping
# ---------------------------------------------------------------------------

STATUS_COLORS: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "gray",
    TaskStatus.RUNNING: "dodgerblue",
    TaskStatus.COMPLETED: "green3",
    TaskStatus.FAILED: "red",
    TaskStatus.SKIPPED: "orange",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, maxlen: int = 40) -> str:
    """Truncate *text* to *maxlen* characters, adding '…' if needed."""
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1] + "…"


def _dot_escape(text: str) -> str:
    """Escape characters that are special in DOT labels."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _get_graphviz():
    """Import and return the ``graphviz`` Python package, or ``None``."""
    try:
        import graphviz  # type: ignore[import-untyped]

        return graphviz
    except ImportError:
        return None


def _has_dot_binary() -> bool:
    """Return True if the ``dot`` binary is on PATH."""
    return shutil.which("dot") is not None


# ---------------------------------------------------------------------------
# DOT source generation (always works — no external deps)
# ---------------------------------------------------------------------------


def _build_dot_source(graph: TaskGraph) -> str:
    """Return a DOT-language string representing *graph*."""
    lines: list[str] = []
    lines.append("digraph PAPoG {")
    lines.append('    rankdir=LR;')
    lines.append('    node [shape=box, style=filled, fontname="Helvetica"];')
    lines.append("")

    for node_id, node in graph.nodes.items():
        color = STATUS_COLORS.get(node.status, "white")
        label = _dot_escape(f"{node_id}\\n{_truncate(node.description)}")
        lines.append(
            f'    "{_dot_escape(node_id)}" '
            f'[label="{label}", fillcolor="{color}"];'
        )

    lines.append("")

    for node_id, node in graph.nodes.items():
        for dep in node.dependencies:
            lines.append(
                f'    "{_dot_escape(dep)}" -> "{_dot_escape(node_id)}";'
            )

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — export functions
# ---------------------------------------------------------------------------


def export_dot(graph: TaskGraph, filepath: str) -> str:
    """Write the graph as a DOT file and return the DOT source string.

    This function has **no external dependencies** — it always works.
    """
    source = _build_dot_source(graph)
    Path(filepath).write_text(source, encoding="utf-8")
    return source


def export_png(graph: TaskGraph, filepath: str) -> str:
    """Render the graph as a PNG image.

    Raises
    ------
    RuntimeError
        If the ``graphviz`` Python package or the ``dot`` binary is
        missing.

    Returns the output filepath.
    """
    gv = _get_graphviz()
    if gv is None:
        raise RuntimeError(
            "The 'graphviz' Python package is required for PNG export. "
            "Install it with: pip install graphviz"
        )
    if not _has_dot_binary():
        raise RuntimeError(
            "The 'dot' binary (Graphviz system package) is required for "
            "PNG rendering. Install it with: apt install graphviz"
        )

    source = _build_dot_source(graph)
    # graphviz.Source.render writes <filepath>.png
    src = gv.Source(source)
    # Strip .png suffix if present — graphviz appends the format extension
    outpath = filepath
    if outpath.endswith(".png"):
        outpath = outpath[:-4]
    src.render(outpath, format="png", cleanup=True)
    return f"{outpath}.png"


def export_svg(graph: TaskGraph, filepath: str) -> str:
    """Render the graph as an SVG image.

    Raises
    ------
    RuntimeError
        If the ``graphviz`` Python package or the ``dot`` binary is
        missing.

    Returns the output filepath.
    """
    gv = _get_graphviz()
    if gv is None:
        raise RuntimeError(
            "The 'graphviz' Python package is required for SVG export. "
            "Install it with: pip install graphviz"
        )
    if not _has_dot_binary():
        raise RuntimeError(
            "The 'dot' binary (Graphviz system package) is required for "
            "SVG rendering. Install it with: apt install graphviz"
        )

    source = _build_dot_source(graph)
    src = gv.Source(source)
    outpath = filepath
    if outpath.endswith(".svg"):
        outpath = outpath[:-4]
    src.render(outpath, format="svg", cleanup=True)
    return f"{outpath}.svg"


# ---------------------------------------------------------------------------
# Execution timeline
# ---------------------------------------------------------------------------


def render_execution_timeline(result: "BenchmarkResult", filepath: str) -> str:
    """Generate a DOT-based timeline visualization of execution events.

    Each node is shown as a horizontal bar spanning its RUNNING→COMPLETED
    duration.  Uses the ``execution_timeline`` entries.

    Returns the DOT source (always works).  If *filepath* ends with
    ``.png`` or ``.svg`` and graphviz is available, the rendered image is
    also produced.
    """
    from src.benchmark import BenchmarkResult  # noqa: F811 — deferred import

    lines: list[str] = []
    lines.append("digraph Timeline {")
    lines.append('    rankdir=LR;')
    lines.append('    node [shape=record, style=filled, fontname="Helvetica"];')
    lines.append("")

    # Collect start/end per node
    node_times: dict[str, dict[str, float]] = {}
    for entry in result.execution_timeline:
        nid = entry.node_id
        if nid not in node_times:
            node_times[nid] = {}
        if entry.to_status == "running":
            node_times[nid]["start"] = entry.timestamp
        elif entry.to_status in ("completed", "failed"):
            node_times[nid]["end"] = entry.timestamp
            node_times[nid]["final_status"] = entry.to_status

    if not node_times:
        # Empty timeline — produce minimal valid DOT
        lines.append('    empty [label="No events", fillcolor="gray"];')
        lines.append("}")
        source = "\n".join(lines)
        Path(filepath).write_text(source, encoding="utf-8")
        return source

    # Normalise timestamps to start at 0
    min_t = min(
        t.get("start", float("inf"))
        for t in node_times.values()
    )

    for nid, times in sorted(node_times.items()):
        start = times.get("start", min_t) - min_t
        end = times.get("end", start) - min_t
        duration = end - start
        status = times.get("final_status", "running")
        color = {
            "completed": "green3",
            "failed": "red",
            "running": "dodgerblue",
        }.get(status, "gray")
        label = _dot_escape(
            f"{nid} | {start:.3f}s → {end:.3f}s ({duration:.3f}s)"
        )
        lines.append(
            f'    "{_dot_escape(nid)}" '
            f'[label="{label}", fillcolor="{color}"];'
        )

    lines.append("}")
    source = "\n".join(lines)

    # Write DOT source
    dot_path = filepath
    if dot_path.endswith((".png", ".svg")):
        dot_path = filepath.rsplit(".", 1)[0] + ".dot"
    Path(dot_path).write_text(source, encoding="utf-8")

    # Attempt rendered output if requested
    gv = _get_graphviz()
    if gv and _has_dot_binary():
        fmt = None
        if filepath.endswith(".png"):
            fmt = "png"
        elif filepath.endswith(".svg"):
            fmt = "svg"
        if fmt:
            base = filepath.rsplit(".", 1)[0]
            src = gv.Source(source)
            src.render(base, format=fmt, cleanup=True)

    return source
