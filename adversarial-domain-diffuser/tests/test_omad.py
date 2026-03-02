import pytest
import numpy as np
from src.omad import OMADOrchestrator
from src.diffusion import DiffusionPolicy
from src.environment import AgentEnvironment
from src.reasoning_agent import DomainReasoningAgent

def test_omad_orchestrator_initialization():
    agents = {
        "agent1": DiffusionPolicy(action_dim=2, horizon=10),
        "agent2": DiffusionPolicy(action_dim=2, horizon=10)
    }
    orchestrator = OMADOrchestrator(agents, alpha=0.5)
    assert orchestrator.alpha == 0.5
    assert len(orchestrator.agents) == 2

def test_entropy_bonus():
    orchestrator = OMADOrchestrator(agents={}, alpha=1.0)
    
    # Trajectory with low variance
    traj_low = np.array([[0.1, 0.1], [0.1, 0.1]])
    # Trajectory with high variance
    traj_high = np.array([[1.0, -1.0], [2.0, -2.0]])
    
    bonus_low = orchestrator.calculate_entropy_bonus(traj_low)
    bonus_high = orchestrator.calculate_entropy_bonus(traj_high)
    
    assert bonus_high > bonus_low
    assert bonus_low == 0.0

def test_coordinate_logic():
    agents = {
        "a1": DiffusionPolicy(action_dim=1, horizon=5),
        "a2": DiffusionPolicy(action_dim=1, horizon=5)
    }
    orchestrator = OMADOrchestrator(agents)
    
    trajectories = {
        "a1": np.ones((5, 1)),
        "a2": np.zeros((5, 1))
    }
    
    consensus = orchestrator.coordinate(trajectories)
    
    # Simple average should be 0.5
    assert consensus.shape == (5, 1)
    assert np.allclose(consensus, 0.5)

def test_environment_integration():
    # Setup agents and policies
    policy1 = DiffusionPolicy(action_dim=1, horizon=3)
    policy2 = DiffusionPolicy(action_dim=1, horizon=3)
    
    orchestrator = OMADOrchestrator({
        "math": policy1,
        "logic": policy2
    })
    
    env = AgentEnvironment(orchestrator=orchestrator)
    
    agent1 = DomainReasoningAgent("math")
    agent2 = DomainReasoningAgent("logic")
    
    env.register_agent(agent1)
    env.register_agent(agent2)
    
    result = env.process_query("What is 2+2?", iterations=1)
    
    assert "coordination_history" in result
    assert len(result["coordination_history"]) == 1
    assert "consensus_path" in result["coordination_history"][0]
    assert result["coordination_history"][0]["consensus_path"].shape == (3, 1)
