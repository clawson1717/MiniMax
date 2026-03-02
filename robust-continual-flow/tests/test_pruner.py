import pytest
from src.trajectory import TrajectoryLogger
from src.pruner import TrajectoryPruner

def test_detect_cycles():
    logger = TrajectoryLogger()
    
    # Create nodes
    n1 = logger.add_node("state 1")
    n2 = logger.add_node("state 2")
    n3 = logger.add_node("state 3")
    
    # Create a cycle: n1 -> n2 -> n3 -> n1
    logger.add_edge(n1, n2, "action 1")
    logger.add_edge(n2, n3, "action 2")
    logger.add_edge(n3, n1, "action 3")
    
    pruner = TrajectoryPruner(logger)
    cycles = pruner.detect_cycles()
    
    assert len(cycles) > 0
    # The cycle should contain n1, n2, n3
    cycle = cycles[0]
    assert n1 in cycle
    assert n2 in cycle
    assert n3 in cycle

def test_prune_unproductive_branches():
    logger = TrajectoryLogger()
    
    # Add a low uncertainty node
    n1 = logger.add_node("low uncertainty", probabilities=[0.9, 0.1]) # low uncertainty
    # Add a high uncertainty node
    n2 = logger.add_node("high uncertainty", probabilities=[0.5, 0.5]) # max uncertainty (1.0)
    
    logger.add_edge(n1, n2, "explore")
    
    pruner = TrajectoryPruner(logger)
    # n2 uncertainty is 1.0. Let's prune with threshold 0.5
    pruner.prune_unproductive_branches(uncertainty_threshold=0.5)
    
    assert n2 in pruner.pruned_node_ids
    assert n1 not in pruner.pruned_node_ids
    
    clean = pruner.get_clean_trajectory()
    
    # Only n1 should remain
    node_ids = [node["id"] for node in clean["nodes"]]
    assert n1 in node_ids
    assert n2 not in node_ids
    assert len(clean["edges"]) == 0 # The edge to/from n2 should be pruned

def test_get_clean_trajectory_multi_branch():
    logger = TrajectoryLogger()
    n1 = logger.add_node("root")
    n2 = logger.add_node("good branch", probabilities=[0.95, 0.05])
    n3 = logger.add_node("bad branch", probabilities=[0.4, 0.6])
    
    logger.add_edge(n1, n2, "go good")
    logger.add_edge(n1, n3, "go bad")
    
    pruner = TrajectoryPruner(logger)
    pruner.prune_unproductive_branches(uncertainty_threshold=0.8)
    
    clean = pruner.get_clean_trajectory()
    node_ids = [node["id"] for node in clean["nodes"]]
    
    assert n1 in node_ids
    assert n2 in node_ids
    assert n3 not in node_ids
    assert len(clean["edges"]) == 1
    assert clean["edges"][0]["to"] == n2
