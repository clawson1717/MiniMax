import pytest
from src.environment import AgentEnvironment
from src.reasoning_agent import DomainReasoningAgent

def test_agent_registration():
    env = AgentEnvironment()
    agent_legal = DomainReasoningAgent(domain="legal")
    agent_medical = DomainReasoningAgent(domain="medical")
    
    env.register_agent(agent_legal)
    env.register_agent(agent_medical)
    
    assert len(env.agents) == 2
    assert env.agents[0].domain == "legal"
    assert env.agents[1].domain == "medical"

def test_message_passing_via_blackboard():
    env = AgentEnvironment()
    agent_legal = DomainReasoningAgent(domain="legal")
    env.register_agent(agent_legal)
    
    env.post_to_blackboard("legal", "The legal perspective is clear.")
    
    assert len(env.blackboard) == 1
    assert env.blackboard[0]["agent_domain"] == "legal"
    assert "The legal perspective" in env.blackboard[0]["content"]
    
    content = env.get_blackboard_content()
    assert "[legal]: The legal perspective is clear." in content

def test_process_query_cycle():
    env = AgentEnvironment()
    env.register_agent(DomainReasoningAgent(domain="legal"))
    env.register_agent(DomainReasoningAgent(domain="medical"))
    
    query = "What are the implications of AI in healthcare?"
    result = env.process_query(query, iterations=1)
    
    assert "query" in result
    assert result["query"] == query
    assert "legal" in result["responses"]
    assert "medical" in result["responses"]
    # 2 agents * 1 iteration = 2 blackboard entries
    assert len(result["blackboard_history"]) == 2
