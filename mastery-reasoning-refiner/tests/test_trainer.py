import pytest
import torch
from unittest.mock import MagicMock
from src.trainer import MasteryTrainer
from src.agent import ReasoningAgent
from src.pedagogy import PedagogicalController, BloomTier

@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=ReasoningAgent)
    # Mock generate_reasoning to return a list of strings
    agent.generate_reasoning.return_value = [
        "Thinking step 1... therefore step 2... conclude.",
        "Short answer.",
        "Detailed logic because of X and Y, since Z is true.",
        "Step by step analysis."
    ]
    # Mock get_logits to return a tensor
    agent.get_logits.return_value = torch.randn(1, 1000)
    # Mock refine_reasoning to return a string
    agent.refine_reasoning.return_value = "Thought: Refined reasoning: therefore the answer is 42."
    # Mock device for torch tensors
    agent.model = MagicMock()
    agent.model.device = "cpu"
    return agent

def test_trainer_initialization(mock_agent):
    trainer = MasteryTrainer(agent=mock_agent)
    assert trainer.agent == mock_agent
    assert isinstance(trainer.pedagogy, PedagogicalController)
    assert trainer.optimizer is not None

def test_train_step(mock_agent):
    trainer = MasteryTrainer(agent=mock_agent)
    example = {"prompt": "What is 2+2?", "tier": BloomTier.KNOWLEDGE}
    
    initial_mastery = trainer.pedagogy.get_mastery_report()[BloomTier.KNOWLEDGE.name]
    
    result = trainer.train_step(example)
    
    assert "success_score" in result
    assert "loss" in result
    assert result["example_tier"] == BloomTier.KNOWLEDGE.name
    
    # Check if mastery was updated
    new_mastery = trainer.pedagogy.get_mastery_report()[BloomTier.KNOWLEDGE.name]
    assert new_mastery != initial_mastery

def test_run_epoch(mock_agent):
    trainer = MasteryTrainer(agent=mock_agent)
    dataset = [
        {"prompt": "P1", "tier": BloomTier.KNOWLEDGE},
        {"prompt": "P2", "tier": BloomTier.KNOWLEDGE}
    ]
    
    metrics = trainer.run_epoch(dataset)
    
    assert len(metrics) == 2
    assert metrics[0]["prompt"] == "P1"
    assert metrics[1]["prompt"] == "P2"

def test_placeholder_success_metric():
    trainer = MasteryTrainer(agent=MagicMock(spec=ReasoningAgent))
    
    # Low score for short, keyword-less trace
    score_low = trainer._calculate_placeholder_success("No logic.")
    
    # Higher score for longer trace with keywords
    score_high = trainer._calculate_placeholder_success(
        "First step is this. Therefore we do that. Because of reason, we conclude."
    )
    
    assert score_high > score_low
    assert 0.0 <= score_low <= 1.0
    assert 0.0 <= score_high <= 1.0
