import pytest
from src.node import TrajectoryNode, NodeStatus
from src.graph import TrajectoryGraph
from src.pruning import PruningPolicy

def test_pruning_policy_cycle_detection():
    policy = PruningPolicy()
    graph = TrajectoryGraph()
    
    n1 = TrajectoryNode(id="n1", content="Step 1")
    n2 = TrajectoryNode(id="n2", content="Step 2")
    n3 = TrajectoryNode(id="n3", content="Step 3")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    
    # Normally add_edge prevents cycles, but we can manually force one for testing the policy
    # if we want to test the detection logic in isolation.
    n1.parent_id = "n3" # Manual cycle: n1 -> n2 -> n3 -> n1
    
    assert policy.detect_cycle_causal_node("n1", graph) == "n1"
    assert policy.detect_cycle_causal_node("n3", graph) == "n3"

def test_pruning_policy_unproductive_detection():
    policy = PruningPolicy(unproductive_threshold=2)
    graph = TrajectoryGraph()
    
    nodes = []
    for i in range(5):
        node = TrajectoryNode(id=f"n{i}", content=f"Step {i}", score=10.0, confidence=0.5)
        graph.add_node(node)
        nodes.append(node)
        if i > 0:
            graph.add_edge(f"n{i-1}", f"n{i}")
            
    # Initial state (i=0..4) all have same score/confidence
    # Path is n0 -> n1 -> n2 -> n3 -> n4
    # Length is 5. Threshold is 2.
    # Window of N+1 = 3.
    # At n2: path(n0, n1, n2), window(n0, n1, n2), window[0]=n0, window[1:]=[n1, n2]. No improvement over n0.
    assert policy.is_unproductive("n2", graph) == True
    
    # Test with improvement
    nodes[3].score = 11.0 # n3 improves over n1 (base of window for n3 is n1)
    # n3 window: n1, n2, n3. Base n1 (score 10). n3 (score 11) > n1. Not unproductive.
    assert policy.is_unproductive("n3", graph) == False
    
    # n4 window: n2, n3, n4. Base n2 (score 10). n3 (score 11) > n2. Not unproductive.
    assert policy.is_unproductive("n4", graph) == False

def test_pruning_policy_metrics():
    policy = PruningPolicy(compute_cost_per_node=2.5)
    graph = TrajectoryGraph()
    
    n1 = TrajectoryNode(id="n1", content="Root")
    n2 = TrajectoryNode(id="n2", content="Branch 1")
    n3 = TrajectoryNode(id="n3", content="Leaf 1")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    
    policy.apply_pruning("n2", graph)
    
    assert n2.status == NodeStatus.PRUNED
    assert n3.status == NodeStatus.PRUNED
    metrics = policy.get_metrics()
    assert metrics["nodes_pruned"] == 2 # n2 and n3
    assert metrics["compute_saved"] == 5.0 # 2 * 2.5

def test_decide_prune_vs_backtrack():
    policy = PruningPolicy(unproductive_threshold=1)
    graph = TrajectoryGraph()
    
    n1 = TrajectoryNode(id="n1", content="v1", score=1.0)
    n2 = TrajectoryNode(id="n2", content="v2", score=1.0)
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_edge("n1", "n2")
    
    # Unproductive at n2 because threshold is 1 and no improvement.
    assert policy.decide_prune_vs_backtrack("n2", graph) == "prune"
    
    n2.score = 2.0
    assert policy.decide_prune_vs_backtrack("n2", graph) == "none"
    
    # Force cycle
    n1.parent_id = "n2"
    assert policy.decide_prune_vs_backtrack("n2", graph) == "prune" # Should detect cycle first
