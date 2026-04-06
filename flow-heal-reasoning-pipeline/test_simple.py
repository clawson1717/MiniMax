#!/usr/bin/env python3
"""Simple test script to verify ReasoningStep implementation."""
import sys
sys.path.insert(0, 'src')

from flow_heal.src.payload import ReasoningStep

# Test basic creation
step = ReasoningStep(
    agent_id="agent-001",
    intent_hash="intent-xyz789",
    content="The capital of France is Paris.",
    uncertainty_score=0.15,
    causal_parent="parent-123"
)

# Verify attributes
assert step.agent_id == "agent-001"
assert step.intent_hash == "intent-xyz789"
assert step.content == "The capital of France is Paris."
assert step.uncertainty_score == 0.15
assert step.causal_parent == "parent-123"

# Test default parent
step2 = ReasoningStep(
    agent_id="agent-002",
    intent_hash="intent-xyz789",
    content="Some content",
    uncertainty_score=0.0
)
assert step2.causal_parent is None

# Test to_dict and from_dict
data = step.to_dict()
restored = ReasoningStep.from_dict(data)
assert restored.agent_id == step.agent_id
assert restored.intent_hash == step.intent_hash
assert restored.content == step.content
assert restored.uncertainty_score == step.uncertainty_score
assert restored.causal_parent == step.causal_parent

print("All tests passed!")
print(f"Created: {step}")
print(f"Serialized: {data}")
