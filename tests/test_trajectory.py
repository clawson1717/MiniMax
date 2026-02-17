"""Tests for trajectory graph module."""

import pytest
from src.trajectory_graph import TrajectoryGraph, TrajectoryNode


class TestTrajectoryNode:
    """Tests for TrajectoryNode."""
    
    def test_node_creation(self):
        """Test creating a trajectory node."""
        node = TrajectoryNode(
            step_index=0,
            action="navigate",
            observation="page loaded",
            uncertainty=0.5,
        )
        assert node.step_index == 0
        assert node.action == "navigate"
        assert node.uncertainty == 0.5
        assert node.is_pruned is False


class TestTrajectoryGraph:
    """Tests for TrajectoryGraph."""
    
    def test_initialization(self):
        """Test graph initialization."""
        graph = TrajectoryGraph()
        assert len(graph.nodes) == 0
        assert graph.current_node_id == -1
    
    def test_add_node(self):
        """Test adding a node."""
        graph = TrajectoryGraph()
        node_id = graph.add_node("action1", "obs1")
        assert node_id == 0
        assert len(graph.nodes) == 1
        assert graph.root_node_id == 0
    
    def test_add_child_node(self):
        """Test adding child nodes."""
        graph = TrajectoryGraph()
        parent_id = graph.add_node("action1")
        child_id = graph.add_node("action2", parent_id=parent_id)
        assert child_id == 1
        assert graph.nodes[child_id].parent == parent_id
    
    def test_get_path(self):
        """Test getting path from root to node."""
        graph = TrajectoryGraph()
        graph.add_node("action1")
        graph.add_node("action2", parent_id=0)
        graph.add_node("action3", parent_id=1)
        
        path = graph.get_path(2)
        assert len(path) == 3
        assert path[0].action == "action1"
        assert path[2].action == "action3"
    
    def test_find_recovery_point(self):
        """Test finding recovery point."""
        graph = TrajectoryGraph()
        graph.add_node("action1", uncertainty=0.3)
        graph.add_node("action2", uncertainty=0.8)
        graph.add_node("action3", uncertainty=0.9)
        
        recovery = graph.find_recovery_point(threshold=0.7)
        assert recovery == 0  # First node below threshold
    
    def test_prune_branch(self):
        """Test pruning a branch."""
        graph = TrajectoryGraph()
        graph.add_node("action1")
        graph.add_node("action2", parent_id=0)
        graph.add_node("action3", parent_id=1)
        
        pruned = graph.prune_branch(1)
        assert len(pruned) >= 1
        assert graph.nodes[1].is_pruned is True
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        graph = TrajectoryGraph()
        graph.add_node("action1")
        graph.add_node("action2", parent_id=0)
        
        d = graph.to_dict()
        assert "nodes" in d
        assert len(d["nodes"]) == 2
    
    def test_reset(self):
        """Test resetting graph."""
        graph = TrajectoryGraph()
        graph.add_node("action1")
        graph.reset()
        assert len(graph.nodes) == 0
