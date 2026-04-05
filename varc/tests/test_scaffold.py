"""Scaffold-level tests for VARC core data structures and TrajectoryGraph."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import src...` when running pytest from the varc/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trajectory import TrajectoryGraph  # noqa: E402
from src.types import (  # noqa: E402
    EvidenceItem,
    LayerResult,
    Milestone,
    VerificationStatus,
)


def test_verification_status_enum_values() -> None:
    names = {s.name for s in VerificationStatus}
    assert names == {"PENDING", "PASS", "FAIL", "RECTIFIED", "PRUNED"}
    assert VerificationStatus.PENDING.value == "pending"
    assert VerificationStatus.PASS.value == "pass"


def test_evidence_item_construction_with_defaults() -> None:
    e = EvidenceItem(source="doc://a", claim="x > 0")
    assert e.source == "doc://a"
    assert e.claim == "x > 0"
    assert e.confidence == 0.0
    assert e.metadata == {}
    # each instance should get its own dict
    e.metadata["k"] = 1
    other = EvidenceItem(source="s", claim="c")
    assert other.metadata == {}


def test_layer_result_construction_with_defaults() -> None:
    r = LayerResult(layer_name="memory_grounding", passed=True, reason="ok")
    assert r.layer_name == "memory_grounding"
    assert r.passed is True
    assert r.reason == "ok"
    assert r.confidence == 0.0


def test_milestone_construction_with_defaults() -> None:
    m = Milestone(id="m1", step_index=0, content="first step")
    assert m.id == "m1"
    assert m.step_index == 0
    assert m.content == "first step"
    assert m.evidence_chain == []
    assert m.verification_status is VerificationStatus.PENDING
    assert m.layer_results == {}
    assert m.parent_id is None
    assert m.children_ids == []
    # independent mutable defaults per instance
    m.children_ids.append("m2")
    other = Milestone(id="x", step_index=1, content="y")
    assert other.children_ids == []


def test_trajectory_add_milestone_and_root() -> None:
    g = TrajectoryGraph()
    root = Milestone(id="m1", step_index=0, content="root")
    g.add_milestone(root)
    assert g.root_id == "m1"
    assert "m1" in g.milestones

    child = Milestone(id="m2", step_index=1, content="child", parent_id="m1")
    g.add_milestone(child)
    assert g.milestones["m1"].children_ids == ["m2"]
    # adding again should not duplicate in parent's children_ids
    g.add_milestone(child)
    assert g.milestones["m1"].children_ids == ["m2"]


def test_trajectory_marking_accept_prune_rectify() -> None:
    g = TrajectoryGraph()
    g.add_milestone(Milestone(id="a", step_index=0, content="a"))
    g.add_milestone(Milestone(id="b", step_index=1, content="b", parent_id="a"))
    g.add_milestone(Milestone(id="c", step_index=2, content="c", parent_id="a"))

    g.mark_accepted("a")
    g.mark_pruned("b")
    g.mark_rectified("c")

    assert g.milestones["a"].verification_status is VerificationStatus.PASS
    assert g.milestones["b"].verification_status is VerificationStatus.PRUNED
    assert g.milestones["c"].verification_status is VerificationStatus.RECTIFIED
    assert g.accepted_ids == ["a"]
    assert g.pruned_ids == ["b"]
    assert g.rectified_ids == ["c"]

    # idempotent — calling twice should not duplicate
    g.mark_accepted("a")
    assert g.accepted_ids == ["a"]


def test_trajectory_marking_unknown_raises() -> None:
    g = TrajectoryGraph()
    import pytest

    with pytest.raises(KeyError):
        g.mark_accepted("nope")
    with pytest.raises(KeyError):
        g.mark_pruned("nope")
    with pytest.raises(KeyError):
        g.mark_rectified("nope")


def test_get_lineage_returns_ancestry() -> None:
    g = TrajectoryGraph()
    g.add_milestone(Milestone(id="r", step_index=0, content="root"))
    g.add_milestone(Milestone(id="a", step_index=1, content="a", parent_id="r"))
    g.add_milestone(Milestone(id="b", step_index=2, content="b", parent_id="a"))
    g.add_milestone(Milestone(id="c", step_index=3, content="c", parent_id="b"))

    assert g.get_lineage("r") == ["r"]
    assert g.get_lineage("a") == ["r", "a"]
    assert g.get_lineage("c") == ["r", "a", "b", "c"]

    import pytest

    with pytest.raises(KeyError):
        g.get_lineage("ghost")


def test_to_dict_round_trips_basic_fields() -> None:
    g = TrajectoryGraph()
    root = Milestone(id="r", step_index=0, content="root")
    root.evidence_chain.append(
        EvidenceItem(source="s1", claim="fact", confidence=0.8, metadata={"k": "v"})
    )
    root.layer_results["memory_grounding"] = LayerResult(
        layer_name="memory_grounding", passed=True, reason="consistent", confidence=0.9
    )
    g.add_milestone(root)
    g.add_milestone(Milestone(id="c", step_index=1, content="child", parent_id="r"))
    g.mark_accepted("r")
    g.mark_pruned("c")

    d = g.to_dict()
    assert d["root_id"] == "r"
    assert d["accepted_ids"] == ["r"]
    assert d["pruned_ids"] == ["c"]
    assert d["rectified_ids"] == []
    assert set(d["milestones"].keys()) == {"r", "c"}

    r = d["milestones"]["r"]
    assert r["id"] == "r"
    assert r["step_index"] == 0
    assert r["content"] == "root"
    assert r["verification_status"] == "pass"
    assert r["children_ids"] == ["c"]
    assert r["parent_id"] is None
    assert r["evidence_chain"][0] == {
        "source": "s1",
        "claim": "fact",
        "confidence": 0.8,
        "metadata": {"k": "v"},
    }
    assert r["layer_results"]["memory_grounding"]["passed"] is True
    assert r["layer_results"]["memory_grounding"]["confidence"] == 0.9

    # JSON-serializable
    import json

    json.dumps(d)

    c = d["milestones"]["c"]
    assert c["parent_id"] == "r"
    assert c["verification_status"] == "pruned"
