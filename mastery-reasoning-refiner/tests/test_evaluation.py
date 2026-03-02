import pytest
import torch
from unittest.mock import MagicMock
from src.trainer import MasteryTrainer
from src.agent import ReasoningAgent
from src.pedagogy import PedagogicalController

def test_evaluate_world_model():
    # Setup mock agent and trainer
    mock_agent = MagicMock(spec=ReasoningAgent)
    # Mock get_logits to return a tensor of probabilities/logits
    # For 5 tokens action space
    mock_logits = torch.tensor([[1.0, 2.0, 0.5, 0.1, 0.0]])
    mock_agent.get_logits.return_value = mock_logits
    
    trainer = MasteryTrainer(agent=mock_agent)
    
    dataset = [{"prompt": "Test prompt 1"}, {"prompt": "Test prompt 2"}]
    
    capacity = trainer.evaluate_world_model(dataset, env_states=10)
    
    # Capacity should be a non-negative float
    assert isinstance(capacity, float)
    assert capacity >= 0.0
    
    # Check that get_logits was called for each example
    assert mock_agent.get_logits.call_count == 2

def test_train_step_information_gain():
    # Setup mock agent and trainer
    mock_agent = MagicMock(spec=ReasoningAgent)
    
    # Mock IterativeGRPO refinement cycle
    mock_grpo = MagicMock()
    mock_grpo.run_refinement_cycle.return_value = {
        "best_initial_trace": "I think this is true.",
        "final_trace": "I think this is true because the evidence points to it.",
        "history": []
    }
    
    trainer = MasteryTrainer(agent=mock_agent)
    trainer.grpo = mock_grpo # Inject mock grpo
    
    example = {"prompt": "Is it true?"}
    metrics = trainer.train_step(example)
    
    assert "info_gain_bits" in metrics
    assert isinstance(metrics["info_gain_bits"], float)
    # Gain should be positive since final_trace is more descriptive/structured in our mock
    assert metrics["info_gain_bits"] > 0
