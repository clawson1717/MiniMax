import torch
import pytest
from src.state import ReasoningState

def test_reasoning_state_init():
    hidden_state = torch.randn(1, 128)
    uncertainty = 0.2
    confidence = 0.8
    timestamp = 0.5
    
    state = ReasoningState(
        hidden_state=hidden_state,
        uncertainty=uncertainty,
        confidence=confidence,
        timestamp=timestamp
    )
    
    assert torch.equal(state.hidden_state, hidden_state)
    assert state.uncertainty == uncertainty
    assert state.confidence == confidence
    assert state.timestamp == timestamp

def test_reasoning_state_to_dict():
    hidden_state = torch.randn(1, 64)
    state = ReasoningState(hidden_state, 0.1, 0.9, 1.0)
    d = state.to_dict()
    
    assert isinstance(d, dict)
    assert torch.equal(d["hidden_state"], hidden_state)
    assert d["uncertainty"] == 0.1
    assert d["confidence"] == 0.9
    assert d["timestamp"] == 1.0

def test_reasoning_state_clone():
    hidden_state = torch.randn(1, 64)
    state = ReasoningState(hidden_state, 0.1, 0.9, 1.0)
    cloned = state.clone()
    
    assert torch.equal(cloned.hidden_state, state.hidden_state)
    assert cloned.hidden_state is not state.hidden_state
    assert cloned.uncertainty == state.uncertainty
    assert cloned.confidence == state.confidence
    assert cloned.timestamp == state.timestamp
