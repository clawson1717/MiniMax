import pytest
from src.node import TrajectoryNode, NodeStatus
from src.graph import TrajectoryGraph

def test_add_node():
    graph = TrajectoryGraph()
    node = TrajectoryNode(id="n1", content="Node 1")
    graph.add_node(node)
    
    assert graph.get_node("n1") == node
    assert graph.get_node("n2") is None

def test_add_node_duplicate():
    graph = TrajectoryGraph()
    node = TrajectoryNode(id="n1", content="Node 1")
    graph.add_node(node)
    
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(node)

def test_add_edge_success():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    n2 = TrajectoryNode(id="n2", content="Node 2")
    graph.add_node(n1)
    graph.add_node(n2)
    
    graph.add_edge("n1", "n2")
    
    assert "n2" in n1.children_ids
    assert n2.parent_id == "n1"

def test_add_edge_missing_node():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    graph.add_node(n1)
    
    with pytest.raises(ValueError, match="Parent node n2 not found"):
        graph.add_edge("n2", "n1")
        
    with pytest.raises(ValueError, match="Child node n2 not found"):
        graph.add_edge("n1", "n2")

def test_cycle_detection_self_loop():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    graph.add_node(n1)
    
    with pytest.raises(ValueError, match="would create a cycle"):
        graph.add_edge("n1", "n1")

def test_cycle_detection_direct():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    n2 = TrajectoryNode(id="n2", content="Node 2")
    graph.add_node(n1)
    graph.add_node(n2)
    
    graph.add_edge("n1", "n2")
    
    with pytest.raises(ValueError, match="would create a cycle"):
        graph.add_edge("n2", "n1")

def test_cycle_detection_indirect():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    n2 = TrajectoryNode(id="n2", content="Node 2")
    n3 = TrajectoryNode(id="n3", content="Node 3")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    
    with pytest.raises(ValueError, match="would create a cycle"):
        graph.add_edge("n3", "n1")

def test_get_path():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Node 1")
    n2 = TrajectoryNode(id="n2", content="Node 2")
    n3 = TrajectoryNode(id="n3", content="Node 3")
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n2", "n3")
    
    path = graph.get_path("n3")
    assert [n.id for n in path] == ["n1", "n2", "n3"]
    
    # Path for non-existent node
    assert graph.get_path("n4") == []
    
    # Single node path
    n4 = TrajectoryNode(id="n4", content="Standalone")
    graph.add_node(n4)
    assert [n.id for n in graph.get_path("n4")] == ["n4"]

def test_prune_branch():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Root")
    n2 = TrajectoryNode(id="n2", content="Child 1")
    n3 = TrajectoryNode(id="n3", content="Child 2")
    n4 = TrajectoryNode(id="n4", content="Grandchild")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n3")
    graph.add_edge("n2", "n4")
    
    graph.prune_branch("n2")
    
    assert graph.get_node("n1").status == NodeStatus.PENDING
    assert graph.get_node("n2").status == NodeStatus.PRUNED
    assert graph.get_node("n3").status == NodeStatus.PENDING
    assert graph.get_node("n4").status == NodeStatus.PRUNED

def test_prune_missing_node():
    graph = TrajectoryGraph()
    graph.prune_branch("non_existent") # Should not raise

def test_find_alternatives():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="Root")
    n2 = TrajectoryNode(id="n2", content="C1")
    n3 = TrajectoryNode(id="n3", content="C2")
    n4 = TrajectoryNode(id="n4", content="C3")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n3")
    graph.add_edge("n1", "n4")
    
    alts = graph.find_alternatives("n2")
    assert set(alts) == {"n3", "n4"}
    
    alts_r = graph.find_alternatives("n1")
    assert alts_r == []
    
    assert graph.find_alternatives("non_existent") == []

def test_find_alternatives_no_parent():
    graph = TrajectoryGraph()
    graph.add_node(TrajectoryNode(id="n1", content="Root"))
    assert graph.find_alternatives("n1") == []

def test_find_alternatives_missing_parent_in_graph():
    graph = TrajectoryGraph()
    node = TrajectoryNode(id="n1", content="Node 1", parent_id="missing")
    graph.add_node(node)
    assert graph.find_alternatives("n1") == []

def test_would_create_cycle_corrupted_graph():
    graph = TrajectoryGraph()
    n2 = TrajectoryNode(id="n2", content="N2", children_ids=["missing"])
    graph.add_node(n2)
    # Testing _would_create_cycle indirectly via add_edge
    n1 = TrajectoryNode(id="n1", content="N1")
    graph.add_node(n1)
    # n1 -> n2. n2 has a child "missing" not in graph.
    # We want to ensure _would_create_cycle doesn't crash when it encounters "missing".
    graph.add_edge("n1", "n2")
    assert "n2" in n1.children_ids

def test_would_create_cycle_visited_twice():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="N1")
    n2 = TrajectoryNode(id="n2", content="N2")
    n3 = TrajectoryNode(id="n3", content="N3")
    n4 = TrajectoryNode(id="n4", content="N4")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    # n1 -> n2 -> n4
    # n1 -> n3 -> n4
    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n3")
    graph.add_edge("n2", "n4")
    graph.add_edge("n3", "n4")
    
    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge("n4", "n1")

def test_would_create_cycle_visited_twice_not_target():
    graph = TrajectoryGraph()
    nodes = ["n1", "n2", "n3", "n4", "target"]
    for nid in nodes:
        graph.add_node(TrajectoryNode(id=nid, content=nid))
        
    graph.add_edge("n1", "target") # Added first
    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n4")
    graph.add_edge("n2", "n3")
    graph.add_edge("n4", "n3")
    
    # Target is processed last due to stack.pop() and n1.children_ids = [target, n2, n4]
    with pytest.raises(ValueError, match="cycle"):
         graph.add_edge("target", "n1")
