import pytest
from src.reasoning_agent import DomainReasoningAgent

class MockLLMClient:
    def generate(self, prompt: str) -> str:
        return f"LLM Response to: {prompt[:50]}..."

def test_agent_initialization():
    agent = DomainReasoningAgent(domain="legal")
    assert agent.domain == "legal"
    assert agent.model_client is None

def test_prompt_construction():
    agent = DomainReasoningAgent(domain="medical")
    query = "What are the side effects of aspirin?"
    context = agent._retrieve_context(query)
    prompt = agent._build_cot_prompt(query, context)
    
    assert "medical" in prompt.lower()
    assert query in prompt
    assert context in prompt
    assert "Chain-of-Thought" in prompt

def test_generate_response_without_client():
    agent = DomainReasoningAgent(domain="legal")
    query = "What is a contract?"
    response = agent.generate_response(query)
    
    assert "reasoning" in response.lower()
    assert "legal" in response.lower()
    assert query in response

def test_generate_response_with_client():
    client = MockLLMClient()
    agent = DomainReasoningAgent(domain="medical", model_client=client)
    query = "Treating a common cold"
    response = agent.generate_response(query)
    
    assert response.startswith("LLM Response to:")
    assert "medical" in response or "Treating" in response # Check that it called the client
