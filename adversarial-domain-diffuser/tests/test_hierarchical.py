import pytest
import numpy as np
from src.omad import OMADOrchestrator
from src.diffusion import DiffusionPolicy

def test_hierarchical_coordination():
    # Setup agents
    agents = {
        "m1": DiffusionPolicy(action_dim=1, horizon=2),
        "m2": DiffusionPolicy(action_dim=1, horizon=2),
        "c1": DiffusionPolicy(action_dim=1, horizon=2),
    }
    
    # Metadata for grouping
    metadata = [
        {"id": "m1", "morphology": "math"},
        {"id": "m2", "morphology": "math"},
        {"id": "c1", "morphology": "coding"},
    ]
    
    orchestrator = OMADOrchestrator(agents, agent_metadata=metadata)
    
    # Mock trajectories
    # Group 'math': [1, 1], [3, 3] -> Group consensus 2.0
    # Group 'coding': [10, 10] -> Group consensus 10.0
    # Global consensus: (2.0 + 10.0) / 2 = 6.0
    trajectories = {
        "m1": np.array([[1.0], [1.0]]),
        "m2": np.array([[3.0], [3.0]]),
        "c1": np.array([[10.0], [10.0]]),
    }
    
    consensus = orchestrator.coordinate(trajectories)
    
    expected_consensus = np.array([[6.0], [6.0]])
    assert np.allclose(consensus, expected_consensus)

if __name__ == "__main__":
    test_hierarchical_coordination()
