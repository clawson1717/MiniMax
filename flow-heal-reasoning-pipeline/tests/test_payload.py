"""Tests for the ReasoningStep payload module."""
from __future__ import annotations

import pytest
from typing import Any, Optional
from datetime import datetime

from flow_heal.src.payload import ReasoningStep


def test_reasoning_step_creation() -> None:
    """Test basic ReasoningStep creation with all required fields."""
    step = ReasoningStep(
        agent_id="agent-001",
        intent_hash="intent-xyz789",
        content="The capital of France is Paris.",
        uncertainty_score=0.15,
        causal_parent="parent-123"
    )
    
    assert step.agent_id == "agent-001"
    assert step.intent_hash == "intent-xyz789"
    assert step.content == "The capital of France is Paris."
    assert step.uncertainty_score == 0.15
    assert step.causal_parent == "parent-123"
    assert isinstance(step.timestamp, datetime)
    assert step.metadata == {}


def test_reasoning_step_optional_parent() -> None:
    """Test ReasoningStep with optional causal_parent (None)."""
    step = ReasoningStep(
        agent_id="agent-002",
        intent_hash="intent-xyz789",
        content="The capital of France is Paris.",
        uncertainty_score=0.0,
        causal_parent=None
    )
    
    assert step.causal_parent is None


def test_reasoning_step_default_uncertainty() -> None:
    """Test default uncertainty_score value."""
    step = ReasoningStep(
        agent_id="agent-003",
        intent_hash="intent-xyz789",
        content="Some content",
        causal_parent="parent-456"
    )
    
    assert step.uncertainty_score == 0.0


def test_reasoning_step_metadata() -> None:
    """Test ReasoningStep with metadata."""
    metadata = {"source": "LLM", "model": "gpt-4", "confidence": 0.85}
    step = ReasoningStep(
        agent_id="agent-004",
        intent_hash="intent-xyz789",
        content="Some content",
        uncertainty_score=0.2,
        causal_parent="parent-789",
        metadata=metadata
    )
    
    assert step.metadata == metadata


def test_to_dict_serialization() -> None:
    """Test serialization to dictionary."""
    step = ReasoningStep(
        agent_id="agent-005",
        intent_hash="intent-xyz789",
        content="Test content",
        uncertainty_score=0.3,
        causal_parent="parent-000",
        metadata={"key": "value"}
    )
    
    data = step.to_dict()
    
    assert isinstance(data, dict)
    assert data["agent_id"] == "agent-005"
    assert data["intent_hash"] == "intent-xyz789"
    assert data["content"] == "Test content"
    assert data["uncertainty_score"] == 0.3
    assert data["causal_parent"] == "parent-000"
    assert isinstance(data["timestamp"], str)
    assert data["metadata"] == {"key": "value"}


def test_from_dict_deserialization() -> None:
    """Test deserialization from dictionary."""
    input_data = {
        "agent_id": "agent-006",
        "intent_hash": "intent-xyz789",
        "content": "Deserialized content",
        "uncertainty_score": 0.45,
        "causal_parent": "parent-111",
        "timestamp": "2026-04-04T12:34:56.789012",
        "metadata": {"test": "data"}
    }
    
    step = ReasoningStep.from_dict(input_data)
    
    assert step.agent_id == "agent-006"
    assert step.intent_hash == "intent-xyz789"
    assert step.content == "Deserialized content"
    assert step.uncertainty_score == 0.45
    assert step.causal_parent == "parent-111"
    assert isinstance(step.timestamp, datetime)
    assert step.metadata == {"test": "data"}


def test_from_dict_without_timestamp() -> None:
    """Test deserialization from dictionary without timestamp."""
    input_data = {
        "agent_id": "agent-007",
        "intent_hash": "intent-xyz789",
        "content": "Content without timestamp",
        "uncertainty_score": 0.6,
        "causal_parent": None,
        "metadata": {}
    }
    
    step = ReasoningStep.from_dict(input_data)
    
    assert step.agent_id == "agent-007"
    assert step.intent_hash == "intent-xyz789"
    assert step.content == "Content without timestamp"
    assert step.uncertainty_score == 0.6
    assert step.causal_parent is None
    assert isinstance(step.timestamp, datetime)  # Should have current timestamp
    assert step.metadata == {}


def test_type_hints() -> None:
    """Test that ReasoningStep has proper type hints for all fields."""
    # This is a compile-time check, but we can verify by attempting to use the class
    # with various types and checking that type errors are raised appropriately
    step = ReasoningStep(
        agent_id="agent-008",
        intent_hash="intent-xyz789",
        content="Test content",
        uncertainty_score=0.7,
        causal_parent="parent-222"
    )
    
    # Verify types of attributes
    assert isinstance(step.agent_id, str)
    assert isinstance(step.intent_hash, str)
    assert isinstance(step.content, str)
    assert isinstance(step.uncertainty_score, float)
    assert (isinstance(step.causal_parent, str) or step.causal_parent is None)


def test_uncertainty_score_range() -> None:
    """Test that uncertainty_score is validated to be between 0.0 and 1.0."""
    # Create a step with valid uncertainty
    step1 = ReasoningStep(
        agent_id="agent-009",
        intent_hash="intent-xyz789",
        content="Content",
        uncertainty_score=0.5
    )
    assert 0.0 <= step1.uncertainty_score <= 1.0
    
    # Create a step with uncertainty_score at boundary 0.0
    step2 = ReasoningStep(
        agent_id="agent-010",
        intent_hash="intent-xyz789",
        content="Content",
        uncertainty_score=0.0
    )
    assert step2.uncertainty_score == 0.0
    
    # Create a step with uncertainty_score at boundary 1.0
    step3 = ReasoningStep(
        agent_id="agent-011",
        intent_hash="intent-xyz789",
        content="Content",
        uncertainty_score=1.0
    )
    assert step3.uncertainty_score == 1.0


def test_str_representation() -> None:
    """Test string representation of ReasoningStep."""
    step = ReasoningStep(
        agent_id="agent-012",
        intent_hash="intent-xyz789",
        content="Test content",
        uncertainty_score=0.25,
        causal_parent="parent-333"
    )
    
    result = str(step)
    assert "agent-012" in result
    assert "0.25" in result
    assert "parent-333" in result


def test_importability() -> None:
    """Test that the module can be imported without errors."""
    try:
        from flow_heal.src.payload import ReasoningStep
        assert ReasoningStep is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ReasoningStep: {e}")


if __name__ == "__main__":
    # Run tests manually if executed directly
    import unittest
    unittest.main()