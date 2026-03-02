import pytest
from src.integrated_loop import IntegratedAdversarialLoop

def test_integrated_loop_initialization():
    agent_configs = [
        {"id": "a1", "domain": "logic", "morphology": "analytical"},
        {"id": "a2", "domain": "creativity", "morphology": "generative"}
    ]
    expert_ref = "Expert knowledge about logic and creativity."
    
    loop = IntegratedAdversarialLoop(
        agent_configs=agent_configs,
        expert_reference=expert_ref,
        max_iterations=2
    )
    
    assert len(loop.agent_map) == 2
    assert "a1" in loop.agent_map
    assert loop.max_iterations == 2
    assert loop.expert_reference == expert_ref

def test_run_iteration_flow():
    agent_configs = [
        {"id": "a1", "domain": "test", "morphology": "test_style"}
    ]
    expert_ref = "The ultimate truth of testing."
    
    loop = IntegratedAdversarialLoop(
        agent_configs=agent_configs,
        expert_reference=expert_ref,
        max_iterations=2
    )
    
    initial_query = "What is the goal of this test?"
    results = loop.run_iteration(initial_query)
    
    assert "history" in results
    assert len(results["history"]) == 2
    assert "final_gap_score" in results
    assert "final_query" in results
    
    # Check history structure
    first_iter = results["history"][0]
    assert first_iter["iteration"] == 0
    assert first_iter["query"] == initial_query
    assert "gap_score" in first_iter
    assert "consensus_summary" in first_iter

def test_evaluation_logic():
    agent_configs = [{"id": "a1", "domain": "test"}]
    loop = IntegratedAdversarialLoop(agent_configs, "Ref")
    
    # Mock responses
    responses = {"a1": ["This is a reasonably long response to test the scoring logic."]}
    score = loop.evaluate_performance(responses, "Ref")
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
