"""
Tests for the Reasoning Trajectory Graph.

Tests for StepNode, TrajectoryGraph, and all associated functionality
including cycle detection, path finding, and graph traversal.
"""

import pytest
import time
from src.trajectory import StepNode, TrajectoryGraph


class TestStepNode:
    """Tests for the StepNode dataclass."""
    
    def test_step_node_creation(self):
        """Test basic StepNode creation."""
        node = StepNode(
            step_id="test-1",
            content="Test reasoning",
            agent_id="agent-1",
            tokens_used=100,
            confidence=0.8,
            timestamp=1234567890.0
        )
        
        assert node.step_id == "test-1"
        assert node.content == "Test reasoning"
        assert node.agent_id == "agent-1"
        assert node.tokens_used == 100
        assert node.confidence == 0.8
        assert node.timestamp == 1234567890.0
        assert node.metadata == {}
        assert node.in_cycle is False
    
    def test_step_node_with_metadata(self):
        """Test StepNode with custom metadata."""
        metadata = {"key": "value", "number": 42}
        node = StepNode(
            step_id="test-2",
            content="Test",
            agent_id="agent-1",
            tokens_used=50,
            confidence=0.9,
            timestamp=time.time(),
            metadata=metadata
        )
        
        assert node.metadata == metadata
    
    def test_step_node_invalid_confidence(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            StepNode(
                step_id="test-3",
                content="Test",
                agent_id="agent-1",
                tokens_used=100,
                confidence=1.5,
                timestamp=time.time()
            )
        
        with pytest.raises(ValueError, match="confidence"):
            StepNode(
                step_id="test-4",
                content="Test",
                agent_id="agent-1",
                tokens_used=100,
                confidence=-0.1,
                timestamp=time.time()
            )
    
    def test_step_node_invalid_tokens(self):
        """Test that negative tokens raises ValueError."""
        with pytest.raises(ValueError, match="tokens_used"):
            StepNode(
                step_id="test-5",
                content="Test",
                agent_id="agent-1",
                tokens_used=-10,
                confidence=0.5,
                timestamp=time.time()
            )
    
    def test_step_node_empty_step_id(self):
        """Test that empty step_id raises ValueError."""
        with pytest.raises(ValueError, match="step_id"):
            StepNode(
                step_id="",
                content="Test",
                agent_id="agent-1",
                tokens_used=100,
                confidence=0.5,
                timestamp=time.time()
            )
    
    def test_step_node_empty_agent_id(self):
        """Test that empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id"):
            StepNode(
                step_id="test-6",
                content="Test",
                agent_id="",
                tokens_used=100,
                confidence=0.5,
                timestamp=time.time()
            )
    
    def test_step_node_serialization(self):
        """Test StepNode to_dict and from_dict."""
        original = StepNode(
            step_id="test-7",
            content="Test reasoning content",
            agent_id="agent-2",
            tokens_used=150,
            confidence=0.75,
            timestamp=1234567890.0,
            metadata={"foo": "bar"},
            in_cycle=True
        )
        
        data = original.to_dict()
        restored = StepNode.from_dict(data)
        
        assert restored.step_id == original.step_id
        assert restored.content == original.content
        assert restored.agent_id == original.agent_id
        assert restored.tokens_used == original.tokens_used
        assert restored.confidence == original.confidence
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata
        assert restored.in_cycle == original.in_cycle


class TestTrajectoryGraph:
    """Tests for the TrajectoryGraph class."""
    
    def test_graph_creation(self):
        """Test creating an empty graph."""
        graph = TrajectoryGraph()
        
        assert len(graph) == 0
        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0
    
    def test_add_step(self):
        """Test adding a step to the graph."""
        graph = TrajectoryGraph()
        
        step_id = graph.add_step(
            content="First thought",
            agent_id="agent-1",
            tokens=100,
            confidence=0.8
        )
        
        assert len(graph) == 1
        assert step_id in graph
        
        step = graph.get_step(step_id)
        assert step is not None
        assert step.content == "First thought"
        assert step.agent_id == "agent-1"
    
    def test_add_step_with_parent(self):
        """Test adding a step with a parent."""
        graph = TrajectoryGraph()
        
        parent_id = graph.add_step(
            content="Parent step",
            agent_id="agent-1",
            tokens=100,
            confidence=0.8
        )
        
        child_id = graph.add_step(
            content="Child step",
            agent_id="agent-2",
            tokens=50,
            confidence=0.9,
            parent_id=parent_id
        )
        
        assert len(graph) == 2
        assert graph.get_edge_count() == 1
        
        children = graph.get_children(parent_id)
        assert len(children) == 1
        assert children[0].step_id == child_id
        
        parents = graph.get_parents(child_id)
        assert len(parents) == 1
        assert parents[0].step_id == parent_id
    
    def test_add_step_invalid_parent(self):
        """Test that adding a step with invalid parent raises ValueError."""
        graph = TrajectoryGraph()
        
        with pytest.raises(ValueError, match="Parent step"):
            graph.add_step(
                content="Test",
                agent_id="agent-1",
                tokens=100,
                confidence=0.8,
                parent_id="non-existent"
            )
    
    def test_add_step_duplicate_id(self):
        """Test that adding a step with duplicate ID raises ValueError."""
        graph = TrajectoryGraph()
        
        graph.add_step(
            content="First",
            agent_id="agent-1",
            tokens=100,
            confidence=0.8,
            step_id="custom-id"
        )
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_step(
                content="Second",
                agent_id="agent-1",
                tokens=50,
                confidence=0.9,
                step_id="custom-id"
            )
    
    def test_get_children(self):
        """Test getting children of a step."""
        graph = TrajectoryGraph()
        
        parent_id = graph.add_step("Parent", "agent-1", 100, 0.8)
        child1_id = graph.add_step("Child 1", "agent-2", 50, 0.9, parent_id=parent_id)
        child2_id = graph.add_step("Child 2", "agent-3", 50, 0.7, parent_id=parent_id)
        
        children = graph.get_children(parent_id)
        assert len(children) == 2
        
        child_ids = {c.step_id for c in children}
        assert child1_id in child_ids
        assert child2_id in child_ids
    
    def test_get_ancestors(self):
        """Test getting ancestors of a step."""
        graph = TrajectoryGraph()
        
        root_id = graph.add_step("Root", "agent-1", 100, 0.8)
        middle_id = graph.add_step("Middle", "agent-2", 50, 0.9, parent_id=root_id)
        leaf_id = graph.add_step("Leaf", "agent-3", 30, 0.7, parent_id=middle_id)
        
        ancestors = graph.get_ancestors(leaf_id)
        assert len(ancestors) == 2
        
        ancestor_ids = [a.step_id for a in ancestors]
        assert middle_id in ancestor_ids
        assert root_id in ancestor_ids
    
    def test_get_descendants(self):
        """Test getting descendants of a step."""
        graph = TrajectoryGraph()
        
        root_id = graph.add_step("Root", "agent-1", 100, 0.8)
        child1_id = graph.add_step("Child 1", "agent-2", 50, 0.9, parent_id=root_id)
        child2_id = graph.add_step("Child 2", "agent-3", 50, 0.7, parent_id=root_id)
        grandchild_id = graph.add_step("Grandchild", "agent-4", 30, 0.8, parent_id=child1_id)
        
        descendants = graph.get_descendants(root_id)
        assert len(descendants) == 3
        
        descendant_ids = {d.step_id for d in descendants}
        assert child1_id in descendant_ids
        assert child2_id in descendant_ids
        assert grandchild_id in descendant_ids
    
    def test_get_path(self):
        """Test finding a path between two nodes."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9, parent_id=step1)
        step3 = graph.add_step("Step 3", "agent-3", 30, 0.7, parent_id=step2)
        
        path = graph.get_path(step1, step3)
        assert path is not None
        assert len(path) == 3
        assert path[0].step_id == step1
        assert path[1].step_id == step2
        assert path[2].step_id == step3
    
    def test_get_path_no_path(self):
        """Test path finding when no path exists."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9)  # No parent
        
        path = graph.get_path(step1, step2)
        assert path is None
    
    def test_get_path_same_node(self):
        """Test path finding for same start and end node."""
        graph = TrajectoryGraph()
        
        step_id = graph.add_step("Step", "agent-1", 100, 0.8)
        path = graph.get_path(step_id, step_id)
        
        assert path is not None
        assert len(path) == 1
        assert path[0].step_id == step_id
    
    def test_get_all_paths(self):
        """Test finding all paths between two nodes."""
        graph = TrajectoryGraph()
        
        root = graph.add_step("Root", "agent-1", 100, 0.8)
        left = graph.add_step("Left", "agent-2", 50, 0.9, parent_id=root)
        right = graph.add_step("Right", "agent-3", 50, 0.7, parent_id=root)
        leaf = graph.add_step("Leaf", "agent-4", 30, 0.8, parent_id=left)
        # Also connect right to leaf
        graph._edges[right].add(leaf)
        graph._reverse_edges[leaf].add(right)
        
        paths = graph.get_all_paths(root, leaf)
        assert len(paths) == 2
        
        # Verify both paths exist
        path_steps = [
            {node.step_id for node in path}
            for path in paths
        ]
        assert any(left in steps for steps in path_steps)
        assert any(right in steps for steps in path_steps)
    
    def test_detect_cycles_no_cycle(self):
        """Test cycle detection on acyclic graph."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9, parent_id=step1)
        step3 = graph.add_step("Step 3", "agent-3", 30, 0.7, parent_id=step2)
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0
        assert not graph.has_cycles()
    
    def test_detect_cycles_simple_cycle(self):
        """Test detecting a simple cycle."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9, parent_id=step1)
        step3 = graph.add_step("Step 3", "agent-3", 30, 0.7, parent_id=step2)
        
        # Create cycle: step3 -> step1
        graph._edges[step3].add(step1)
        graph._reverse_edges[step1].add(step3)
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0
        assert graph.has_cycles()
        
        # Check that nodes are marked
        for node in graph:
            assert node.in_cycle is True
    
    def test_detect_cycles_multiple_cycles(self):
        """Test detecting multiple cycles."""
        graph = TrajectoryGraph()
        
        # Create first cycle
        a = graph.add_step("A", "agent-1", 100, 0.8)
        b = graph.add_step("B", "agent-2", 50, 0.9, parent_id=a)
        graph._edges[b].add(a)
        graph._reverse_edges[a].add(b)
        
        # Create second disconnected cycle
        c = graph.add_step("C", "agent-3", 30, 0.7)
        d = graph.add_step("D", "agent-4", 30, 0.8, parent_id=c)
        graph._edges[d].add(c)
        graph._reverse_edges[c].add(d)
        
        cycles = graph.detect_cycles()
        assert len(cycles) >= 2
        assert graph.has_cycles()
    
    def test_get_roots(self):
        """Test getting root nodes."""
        graph = TrajectoryGraph()
        
        root1 = graph.add_step("Root 1", "agent-1", 100, 0.8)
        root2 = graph.add_step("Root 2", "agent-2", 100, 0.8)
        child = graph.add_step("Child", "agent-3", 50, 0.9, parent_id=root1)
        
        roots = graph.get_roots()
        assert len(roots) == 2
        
        root_ids = {r.step_id for r in roots}
        assert root1 in root_ids
        assert root2 in root_ids
    
    def test_get_leaves(self):
        """Test getting leaf nodes."""
        graph = TrajectoryGraph()
        
        root = graph.add_step("Root", "agent-1", 100, 0.8)
        child = graph.add_step("Child", "agent-2", 50, 0.9, parent_id=root)
        leaf1 = graph.add_step("Leaf 1", "agent-3", 30, 0.7, parent_id=child)
        leaf2 = graph.add_step("Leaf 2", "agent-4", 30, 0.8, parent_id=child)
        
        leaves = graph.get_leaves()
        assert len(leaves) == 2
        
        leaf_ids = {l.step_id for l in leaves}
        assert leaf1 in leaf_ids
        assert leaf2 in leaf_ids
    
    def test_serialization(self):
        """Test graph serialization to/from dict."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9, parent_id=step1)
        step3 = graph.add_step("Step 3", "agent-3", 30, 0.7, parent_id=step1)
        
        data = graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
        
        # Restore graph
        restored = TrajectoryGraph.from_dict(data)
        
        assert len(restored) == 3
        assert restored.get_edge_count() == 2
        
        # Verify structure
        children = restored.get_children(step1)
        assert len(children) == 2
    
    def test_get_steps_by_agent(self):
        """Test filtering steps by agent."""
        graph = TrajectoryGraph()
        
        graph.add_step("Step 1", "agent-1", 100, 0.8)
        graph.add_step("Step 2", "agent-2", 50, 0.9)
        graph.add_step("Step 3", "agent-1", 30, 0.7)
        
        agent1_steps = graph.get_steps_by_agent("agent-1")
        assert len(agent1_steps) == 2
        
        agent2_steps = graph.get_steps_by_agent("agent-2")
        assert len(agent2_steps) == 1
    
    def test_get_total_tokens(self):
        """Test calculating total tokens."""
        graph = TrajectoryGraph()
        
        graph.add_step("Step 1", "agent-1", 100, 0.8)
        graph.add_step("Step 2", "agent-2", 50, 0.9)
        graph.add_step("Step 3", "agent-3", 30, 0.7)
        
        assert graph.get_total_tokens() == 180
    
    def test_get_average_confidence(self):
        """Test calculating average confidence."""
        graph = TrajectoryGraph()
        
        graph.add_step("Step 1", "agent-1", 100, 0.8)
        graph.add_step("Step 2", "agent-2", 50, 0.9)
        graph.add_step("Step 3", "agent-3", 30, 0.7)
        
        avg = graph.get_average_confidence()
        assert abs(avg - 0.8) < 0.01  # (0.8 + 0.9 + 0.7) / 3 = 0.8
    
    def test_clear_cycle_marks(self):
        """Test clearing cycle marks."""
        graph = TrajectoryGraph()
        
        step1 = graph.add_step("Step 1", "agent-1", 100, 0.8)
        step2 = graph.add_step("Step 2", "agent-2", 50, 0.9, parent_id=step1)
        
        # Manually create and mark cycle
        graph._edges[step2].add(step1)
        graph._reverse_edges[step1].add(step2)
        graph.detect_cycles()
        
        # Check marks are set
        for node in graph:
            assert node.in_cycle is True
        
        # Clear marks
        graph.clear_cycle_marks()
        
        # Check marks are cleared
        for node in graph:
            assert node.in_cycle is False
    
    def test_iteration(self):
        """Test iterating over graph nodes."""
        graph = TrajectoryGraph()
        
        ids = []
        for i in range(5):
            step_id = graph.add_step(f"Step {i}", "agent-1", 100, 0.8)
            ids.append(step_id)
        
        iterated_ids = {node.step_id for node in graph}
        assert iterated_ids == set(ids)
    
    def test_get_nonexistent_step(self):
        """Test getting a step that doesn't exist."""
        graph = TrajectoryGraph()
        
        step = graph.get_step("non-existent")
        assert step is None
        
        children = graph.get_children("non-existent")
        assert children == []
        
        ancestors = graph.get_ancestors("non-existent")
        assert ancestors == []
    
    def test_path_between_nonexistent_nodes(self):
        """Test path finding with non-existent nodes."""
        graph = TrajectoryGraph()
        
        path = graph.get_path("non-existent-1", "non-existent-2")
        assert path is None
    
    def test_custom_step_id(self):
        """Test adding a step with custom step_id."""
        graph = TrajectoryGraph()
        
        step_id = graph.add_step(
            content="Custom ID step",
            agent_id="agent-1",
            tokens=100,
            confidence=0.8,
            step_id="my-custom-id"
        )
        
        assert step_id == "my-custom-id"
        assert "my-custom-id" in graph
    
    def test_custom_timestamp_and_metadata(self):
        """Test adding step with custom timestamp and metadata."""
        graph = TrajectoryGraph()
        
        custom_time = 1234567890.0
        metadata = {"source": "test", "priority": 1}
        
        step_id = graph.add_step(
            content="Test",
            agent_id="agent-1",
            tokens=100,
            confidence=0.8,
            timestamp=custom_time,
            metadata=metadata
        )
        
        step = graph.get_step(step_id)
        assert step.timestamp == custom_time
        assert step.metadata == metadata
    
    def test_large_graph(self):
        """Test operations on a larger graph."""
        graph = TrajectoryGraph()
        
        # Create a tree structure
        root = graph.add_step("Root", "agent-1", 100, 0.8)
        
        prev_level = [root]
        for depth in range(3):
            new_level = []
            for parent_id in prev_level:
                for i in range(2):
                    child_id = graph.add_step(
                        f"Node at depth {depth+1}",
                        f"agent-{i}",
                        50,
                        0.9,
                        parent_id=parent_id
                    )
                    new_level.append(child_id)
            prev_level = new_level
        
        # Verify structure
        assert graph.get_node_count() == 15  # 1 + 2 + 4 + 8
        assert graph.get_edge_count() == 14  # n-1 for tree
        
        # No cycles
        assert not graph.has_cycles()
        
        # Path exists from root to any node
        leaves = graph.get_leaves()
        for leaf in leaves:
            path = graph.get_path(root, leaf.step_id)
            assert path is not None
