import pytest
from src.scenarios import AdversarialCycle, HintType

def test_adversarial_cycle_model():
    cycle = AdversarialCycle(
        base_logic="The light must be off if the switch is down.",
        hint="The switch is up, so the light is off.",
        hint_type=HintType.MISLEADING,
        ground_truth="The light status is unknown or the light is on."
    )
    
    assert cycle.hint_type == HintType.MISLEADING
    assert cycle.expected_drift == 0.0
    assert "Cycle" in str(type(cycle))

def test_hint_types():
    assert HintType.HELPFUL == "helpful"
    assert HintType.CORRUPTED == "corrupted"
