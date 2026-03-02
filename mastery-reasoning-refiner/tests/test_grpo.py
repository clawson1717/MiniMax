import pytest
import torch
from unittest.mock import MagicMock
from src.grpo import IterativeGRPO
from src.agent import ReasoningAgent

@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=ReasoningAgent)
    # Mock generate_reasoning to return a list of dummy traces
    agent.generate_reasoning.return_value = [
        "Step 1: Short answer.",
        "Step 1: First we do this. Step 2: Then we do that. Therefore the answer is 42.",
        "Bad response.",
        "Analysis: This is a complex problem. Step 1: Let's analyze. Step 2: Solve. Conclusion: Done."
    ]
    # Mock refine_reasoning to return a refined trace
    agent.refine_reasoning.return_value = "Refined: Step 1: Detailed logic. Step 2: Even more detail. Therefore the final answer is 42."
    return agent

def test_grpo_scoring(mock_agent):
    grpo = IterativeGRPO(agent=mock_agent, group_size=4)
    prompt = "What is 2+2?"
    responses = [
        "Short",
        "Step 1: This is a long and detailed response with keywords like therefore and analysis.",
        "Another long one with step and since.",
        "Just some text."
    ]
    
    scores = grpo.score_responses(prompt, responses)
    
    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == 4
    # Check that standardization occurred (mean should be close to 0)
    assert torch.abs(scores.mean()) < 1e-5

def test_refinement_cycle(mock_agent):
    grpo = IterativeGRPO(agent=mock_agent, group_size=4)
    prompt = "Solve for x."
    
    result = grpo.run_refinement_cycle(prompt, num_iterations=2)
    
    assert "prompt" in result
    assert "best_initial_trace" in result
    assert "final_trace" in result
    assert len(result["history"]) == 3  # Initial + 2 refinements
    
    mock_agent.generate_reasoning.assert_called_once()
    assert mock_agent.refine_reasoning.call_count == 2

def test_train_step_placeholder(mock_agent):
    grpo = IterativeGRPO(agent=mock_agent, group_size=4)
    prompt = "Dummy prompt"
    
    scores = grpo.train_step_placeholder(prompt)
    assert isinstance(scores, torch.Tensor)
