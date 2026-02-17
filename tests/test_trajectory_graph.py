"""Tests for trajectory graph module - WebClipper approach."""

import pytest
import json
import hashlib
from src.trajectory_graph import TrajectoryGraph, GraphNode, GraphEdge


class TestGraphNode:
    """Tests for GraphNode dataclass."""
    
    def test_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(
            state_id=0,
            observation_hash="abc123",
            action="navigate",
            observation_data="page loaded",
            metadata={"url": "https://example.com"},
        )
        assert node.state_id == 0
        assert node.observation_hash == "abc123"
        assert node.action == "navigate"
        assert node.is_pruned is False
    
    def test_node_defaults(self):
        """Test node default values."""
        node = GraphNode(
            state_id=1,
            observation_hash="def456",
            action="click",
        )
        assert node.observation_data is None
        assert node.metadata == {}
        assert node.timestamp is None


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""
    
    def test_edge_creation(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            edge_id=0,
            source=0,
            target=1,
            weight=1.5,
            success=True,
            action="navigate",
        )
        assert edge.edge_id == 0
        assert edge.source == 0
        assert edge.target == 1
        assert edge.weight == 1.5
        assert edge.success is True
    
    def test_edge_defaults(self):
        """Test edge default values."""
        edge = GraphEdge(
            edge_id=1,
            source=1,
            target=2,
        )
        assert edge.weight == 1.0
        assert edge.success is True
        assert edge.action == ""
        assert edge.metadata == {}


class TestTrajectoryGraphInitialization:
    """Tests for graph initialization."""
    
    def test_initialization(self):
        """Test graph initialization."""
        graph = TrajectoryGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.root_state_id is None
        assert graph._next_node_id == 0
        assert graph._next_edge_id == 0
    
    def test_reset(self):
        """Test resetting graph."""
        graph = TrajectoryGraph()
        graph.add_state("obs1", "action1")
        graph.reset()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.root_state_id is None


class TestStateHash:
    """Tests for state hashing."""
    
    def test_string_hash(self):
        """Test hashing string observations."""
        graph = TrajectoryGraph()
        hash1 = graph.get_state_hash("test observation")
        hash2 = graph.get_state_hash("test observation")
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA-256 truncated
    
    def test_dict_hash_consistency(self):
        """Test hashing dict observations is consistent."""
        graph = TrajectoryGraph()
        obs = {"key": "value", "num": 42}
        hash1 = graph.get_state_hash(obs)
        hash2 = graph.get_state_hash({"num": 42, "key": "value"})  # Different order
        assert hash1 == hash2
    
    def test_different_observations_different_hashes(self):
        """Test different observations produce different hashes."""
        graph = TrajectoryGraph()
        hash1 = graph.get_state_hash("obs1")
        hash2 = graph.get_state_hash("obs2")
        assert hash1 != hash2


class TestAddState:
    """Tests for add_state method."""
    
    def test_add_single_state(self):
        """Test adding a single state."""
        graph = TrajectoryGraph()
        state_id = graph.add_state("observation1", "action1")
        assert state_id == 0
        assert len(graph.nodes) == 1
        assert graph.root_state_id == 0
        assert graph.nodes[0].action == "action1"
    
    def test_add_multiple_states(self):
        """Test adding multiple states."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        assert s1 == 0
        assert s2 == 1
        assert s3 == 2
        assert len(graph.nodes) == 3
    
    def test_state_deduplication(self):
        """Test that duplicate observations return existing state_id."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("same observation", "action1")
        s2 = graph.add_state("same observation", "action2")
        assert s1 == s2
        assert len(graph.nodes) == 1
    
    def test_add_state_with_metadata(self):
        """Test adding state with metadata."""
        graph = TrajectoryGraph()
        state_id = graph.add_state(
            "obs",
            "action",
            metadata={"url": "https://example.com", "score": 0.9}
        )
        assert graph.nodes[state_id].metadata["url"] == "https://example.com"


class TestAddEdge:
    """Tests for add_edge method."""
    
    def test_add_single_edge(self):
        """Test adding a single edge."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        edge_id = graph.add_edge(s1, s2)
        assert edge_id == 0
        assert len(graph.edges) == 1
        assert graph.edges[0].source == s1
        assert graph.edges[0].target == s2
    
    def test_add_multiple_edges(self):
        """Test adding multiple edges."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        e1 = graph.add_edge(s1, s2)
        e2 = graph.add_edge(s2, s3)
        assert e1 == 0
        assert e2 == 1
        assert len(graph.edges) == 2
    
    def test_add_edge_with_parameters(self):
        """Test adding edge with weight and success flag."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        edge_id = graph.add_edge(s1, s2, weight=2.5, success=False, action="click")
        edge = graph.edges[edge_id]
        assert edge.weight == 2.5
        assert edge.success is False
        assert edge.action == "click"
    
    def test_add_edge_invalid_source(self):
        """Test adding edge with invalid source state."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        with pytest.raises(ValueError, match="Source state 999"):
            graph.add_edge(999, s1)
    
    def test_add_edge_invalid_target(self):
        """Test adding edge with invalid target state."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        with pytest.raises(ValueError, match="Target state 999"):
            graph.add_edge(s1, 999)


class TestCycleDetection:
    """Tests for detect_cycles method."""
    
    def test_no_cycles_linear(self):
        """Test detecting no cycles in linear graph."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
    
    def test_simple_cycle(self):
        """Test detecting simple cycle."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        graph.add_edge(s3, s1)  # Creates cycle
        cycles = graph.detect_cycles()
        assert len(cycles) == 1
        assert len(cycles[0]) == 4  # Includes repeated start/end
    
    def test_self_loop(self):
        """Test detecting self-loop cycle."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        graph.add_edge(s1, s1)
        cycles = graph.detect_cycles()
        assert len(cycles) == 1
    
    def test_multiple_cycles(self):
        """Test detecting multiple cycles."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "a1")
        s2 = graph.add_state("obs2", "a2")
        s3 = graph.add_state("obs3", "a3")
        s4 = graph.add_state("obs4", "a4")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        graph.add_edge(s3, s2)  # Cycle s2->s3->s2
        graph.add_edge(s3, s4)
        graph.add_edge(s4, s1)  # Cycle s1->s2->s3->s4->s1
        cycles = graph.detect_cycles()
        assert len(cycles) >= 2


class TestGetPrunableBranches:
    """Tests for get_prunable_branches method."""
    
    def test_dead_end_prunable(self):
        """Test that dead-end leaf nodes are prunable."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        graph.add_edge(s1, s2)
        # s2 is a dead end
        prunable = graph.get_prunable_branches()
        assert s2 in prunable
    
    def test_failed_edges_prunable(self):
        """Test that nodes with all failed edges are prunable."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2, success=True)
        graph.add_edge(s2, s3, success=False)
        # s3 is a dead end from failed edge
        prunable = graph.get_prunable_branches()
        assert s3 in prunable
    
    def test_cycle_nodes_prunable(self):
        """Test that cycle nodes without productive branches are prunable."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "a1")
        s2 = graph.add_state("obs2", "a2")
        s3 = graph.add_state("obs3", "a3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        graph.add_edge(s3, s2)  # Cycle between s2 and s3
        prunable = graph.get_prunable_branches()
        assert s2 in prunable or s3 in prunable
    
    def test_not_prunable_with_successful_children(self):
        """Test that nodes with successful productive children are not prunable."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        s4 = graph.add_state("obs4", "action4")  # Productive child outside cycle
        graph.add_edge(s1, s2, success=True)
        graph.add_edge(s2, s3, success=True)
        graph.add_edge(s3, s1, success=True)  # Cycle
        graph.add_edge(s3, s4, success=True)  # Productive exit from cycle (direct from s3)
        prunable = graph.get_prunable_branches()
        # s4 is a dead-end leaf
        # s3 has direct productive exit to s4 (outside cycle), so not prunable
        # s1 and s2 are in cycle but don't have direct productive exits (may be prunable)
        assert s4 in prunable  # Dead end
        assert s3 not in prunable  # Has direct productive exit


class TestPruneBranch:
    """Tests for prune_branch method."""
    
    def test_prune_single_node(self):
        """Test pruning a single node."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        graph.add_edge(s1, s2)
        pruned = graph.prune_branch(s2)
        assert s2 in pruned
        assert graph.nodes[s2].is_pruned is True
        assert graph.nodes[s1].is_pruned is False
    
    def test_prune_branch_recursive(self):
        """Test recursively pruning a branch."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        s4 = graph.add_state("obs4", "action4")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        graph.add_edge(s2, s4)
        pruned = graph.prune_branch(s2)
        assert s2 in pruned
        assert s3 in pruned
        assert s4 in pruned
        assert s1 not in pruned
    
    def test_prune_invalid_state(self):
        """Test pruning non-existent state."""
        graph = TrajectoryGraph()
        with pytest.raises(ValueError, match="State 999"):
            graph.prune_branch(999)


class TestGetPathToState:
    """Tests for get_path_to_state method."""
    
    def test_path_from_root(self):
        """Test getting path from root to state."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        path = graph.get_path_to_state(s3)
        assert path == [s1, s2, s3]
    
    def test_path_to_root(self):
        """Test getting path to root state."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        graph.add_edge(s1, s2)
        path = graph.get_path_to_state(s1)
        assert path == [s1]
    
    def test_path_invalid_state(self):
        """Test getting path to non-existent state."""
        graph = TrajectoryGraph()
        graph.add_state("obs1", "action1")
        with pytest.raises(ValueError, match="State 999"):
            graph.get_path_to_state(999)
    
    def test_path_no_connection(self):
        """Test path when no connection exists."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        # No edge between s1 and s2
        path = graph.get_path_to_state(s2)
        assert path == []  # No path found


class TestGetGraphStats:
    """Tests for get_graph_stats method."""
    
    def test_empty_graph_stats(self):
        """Test stats for empty graph."""
        graph = TrajectoryGraph()
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["cycle_count"] == 0
        assert stats["depth"] == 0
    
    def test_linear_graph_stats(self):
        """Test stats for linear graph."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["cycle_count"] == 0
        assert stats["depth"] == 2
    
    def test_cyclic_graph_stats(self):
        """Test stats for cyclic graph."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s2, s3)
        graph.add_edge(s3, s1)
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 3
        assert stats["cycle_count"] == 1
    
    def test_pruned_stats(self):
        """Test stats with pruned nodes."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s1, s3)
        graph.prune_branch(s3)
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 3
        assert stats["pruned_count"] == 1
        assert stats["active_count"] == 2


class TestGraphNavigation:
    """Tests for navigation methods."""
    
    def test_get_successors(self):
        """Test getting successor states."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s2)
        graph.add_edge(s1, s3)
        successors = graph.get_successors(s1)
        assert s2 in successors
        assert s3 in successors
        assert len(successors) == 2
    
    def test_get_predecessors(self):
        """Test getting predecessor states."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        s3 = graph.add_state("obs3", "action3")
        graph.add_edge(s1, s3)
        graph.add_edge(s2, s3)
        predecessors = graph.get_predecessors(s3)
        assert s1 in predecessors
        assert s2 in predecessors
        assert len(predecessors) == 2
    
    def test_invalid_state_navigation(self):
        """Test navigation with invalid state."""
        graph = TrajectoryGraph()
        assert graph.get_successors(999) == []
        assert graph.get_predecessors(999) == []


class TestSerialization:
    """Tests for serialization methods."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1", metadata={"key": "value"})
        s2 = graph.add_state("obs2", "action2")
        graph.add_edge(s1, s2, weight=2.0, success=True)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "stats" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["stats"]["node_count"] == 2
    
    def test_save_and_load(self, tmp_path):
        """Test saving to file."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("obs1", "action1")
        s2 = graph.add_state("obs2", "action2")
        graph.add_edge(s1, s2)
        filepath = tmp_path / "graph.json"
        graph.save(str(filepath))
        assert filepath.exists()
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert len(data["nodes"]) == 2


class TestComplexScenarios:
    """Tests for complex graph scenarios."""
    
    def test_branching_trajectory(self):
        """Test graph with branching paths."""
        graph = TrajectoryGraph()
        root = graph.add_state("root", "start")
        
        # Branch 1
        b1_s1 = graph.add_state("b1_obs1", "action1")
        b1_s2 = graph.add_state("b1_obs2", "action2")
        graph.add_edge(root, b1_s1)
        graph.add_edge(b1_s1, b1_s2)
        
        # Branch 2
        b2_s1 = graph.add_state("b2_obs1", "action3")
        b2_s2 = graph.add_state("b2_obs2", "action4")
        graph.add_edge(root, b2_s1)
        graph.add_edge(b2_s1, b2_s2)
        
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        
        # Both branches should be reachable from root
        root_successors = graph.get_successors(root)
        assert b1_s1 in root_successors
        assert b2_s1 in root_successors
    
    def test_merge_states(self):
        """Test converging paths (diamond pattern)."""
        graph = TrajectoryGraph()
        s1 = graph.add_state("start", "action1")
        s2 = graph.add_state("left", "action2")
        s3 = graph.add_state("right", "action3")
        s4 = graph.add_state("merge", "action4")
        graph.add_edge(s1, s2)
        graph.add_edge(s1, s3)
        graph.add_edge(s2, s4)
        graph.add_edge(s3, s4)
        
        stats = graph.get_graph_stats()
        assert stats["node_count"] == 4
        assert stats["edge_count"] == 4
        
        # s4 should have 2 predecessors
        preds = graph.get_predecessors(s4)
        assert s2 in preds
        assert s3 in preds
