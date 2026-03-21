import pytest
from src.watchdog import (
    WatchdogOrchestrator,
    ReasoningStep,
    ReasoningStatus,
    WatchdogResult,
)


def test_watchdog_orchestrator_init():
    orch = WatchdogOrchestrator()
    assert orch.steps == []
    assert orch._step_counter == 0


def test_add_step():
    orch = WatchdogOrchestrator()
    step = orch.add_step("First reasoning step")
    assert step.step_id == "step_1"
    assert step.content == "First reasoning step"
    assert len(orch.steps) == 1


def test_add_step_with_parent():
    orch = WatchdogOrchestrator()
    parent = orch.add_step("Parent step")
    child = orch.add_step("Child step", parent_id=parent.step_id)
    assert child.parent_id == "step_1"
    assert len(orch.steps) == 2


def test_get_step():
    orch = WatchdogOrchestrator()
    step = orch.add_step("Test step")
    found = orch.get_step(step.step_id)
    assert found == step
    assert orch.get_step("nonexistent") is None


def test_rollback_to():
    orch = WatchdogOrchestrator()
    s1 = orch.add_step("Step 1")
    s2 = orch.add_step("Step 2")
    s3 = orch.add_step("Step 3")
    
    assert len(orch.steps) == 3
    
    # Rollback to step_2 (exclusive)
    count = orch.rollback_to("step_2")
    assert count == 1
    assert len(orch.steps) == 2
    assert orch.steps[-1].step_id == "step_2"


def test_rollback_full():
    orch = WatchdogOrchestrator()
    orch.add_step("Step 1")
    orch.add_step("Step 2")
    
    count = orch.rollback_to("step_1")
    assert count == 1
    assert len(orch.steps) == 1


def test_run_returns_watchdog_result():
    orch = WatchdogOrchestrator()
    result = orch.run("What is 2+2?")
    assert isinstance(result, WatchdogResult)
    assert result.status in ReasoningStatus


def test_reasoning_status_enum():
    assert ReasoningStatus.VERIFIED.value == "verified"
    assert ReasoningStatus.UNCERTAIN.value == "uncertain"
    assert ReasoningStatus.DRIFTED.value == "drifted"
    assert ReasoningStatus.COMPLETE.value == "complete"
