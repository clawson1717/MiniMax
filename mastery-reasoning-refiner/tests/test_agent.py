import pytest
import torch
from src.agent import ReasoningAgent

@pytest.fixture(scope="module")
def agent():
    # Use a small model and CPU if necessary, but here we follow instructions for 4-bit which requires GPU
    # If no GPU is available, bitsandbytes might fail. 
    # For CI/testing purposes, usually we'd mock, but instructions say "verify that the agent can be initialized and generate a non-empty string".
    return ReasoningAgent(model_id="Qwen/Qwen2.5-0.5B-Instruct")

def test_agent_initialization(agent):
    assert agent.model is not None
    assert agent.tokenizer is not None

def test_generate_reasoning(agent):
    prompt = "What is 2+2?"
    response = agent.generate_reasoning(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\nResponse: {response}")
