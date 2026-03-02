import pytest
import torch
from src.reward_model import ChecklistRewardModel
from src.grpo import IterativeGRPO
from unittest.mock import MagicMock

def test_checklist_reward_model_basic():
    rm = ChecklistRewardModel()
    trace = "<thought> Step 1: Analyze. Step 2: Calculate. Therefore, the answer is 42. </thought>"
    
    results = rm.evaluate_trace(trace)
    assert results["contains_thought_header"] is True
    assert results["contains_conclusion"] is True
    assert results["step_by_step_format"] is True
    
    reward = rm.calculate_reward(trace)
    assert reward > 0.5

def test_checklist_reward_model_poor_trace():
    rm = ChecklistRewardModel()
    trace = "Too short."
    
    reward = rm.calculate_reward(trace)
    assert reward < 0.5
    
    results = rm.evaluate_trace(trace)
    assert results["not_empty"] is False # Because it's < 50 chars now

def test_checklist_reward_model_repetition():
    rm = ChecklistRewardModel()
    repetitive_trace = "This is a very long line that is repeated over and over again.\n" * 10
    
    results = rm.evaluate_trace(repetitive_trace)
    assert results["no_excessive_repetition"] is False

def test_custom_criteria():
    custom_criteria = {
        "is_upper": lambda t: t.isupper(),
        "has_exclamation": lambda t: "!" in t
    }
    rm = ChecklistRewardModel(criteria=custom_criteria)
    
    trace = "HELLO WORLD!"
    results = rm.evaluate_trace(trace)
    assert results["is_upper"] is True
    assert results["has_exclamation"] is True
    assert rm.calculate_reward(trace) == 1.0

def test_grpo_integration():
    mock_agent = MagicMock()
    rm = ChecklistRewardModel()
    grpo = IterativeGRPO(agent=mock_agent, reward_model=rm)
    
    prompt = "What is 2+2?"
    responses = [
        "<thought> Step 1: It is 2. Step 2: Add 2. Therefore, 4. </thought> 2+2=4", # Good
        "It is 4.", # Bad (too short, no header)
    ]
    
    scores = grpo.score_responses(prompt, responses)
    assert len(scores) == 2
    # The first one should have a higher raw reward, thus a higher standardized score
    assert scores[0] > scores[1]

if __name__ == "__main__":
    pytest.main([__file__])
