import pytest
from src.node import TrajectoryNode, NodeStatus, ChecklistItem
from src.graph import TrajectoryGraph
from src.verifier import ChecklistVerifier, VerificationResult
from src.detector import FailureModeDetector
from src.cascade import CascadeEngine, CascadeState

class MockEvaluator:
    def __init__(self, should_pass=True):
        self.should_pass = should_pass
    
    def evaluate(self, content, criterion):
        return self.should_pass, "Mock evidence"

@pytest.fixture
def basic_setup():
    graph = TrajectoryGraph()
    n1 = TrajectoryNode(id="n1", content="First step content")
    n2 = TrajectoryNode(id="n2", content="Second step content")
    n3 = TrajectoryNode(id="n3", content="Third step alternative content")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_edge("n1", "n2")
    graph.add_edge("n1", "n3")
    
    verifier = ChecklistVerifier(templates={"step": ["Check accuracy"]})
    detector = FailureModeDetector()
    engine = CascadeEngine(graph, verifier, detector)
    
    return engine, n1, n2, n3

def test_cascade_initialization(basic_setup):
    engine, n1, n2, n3 = basic_setup
    assert engine.state.current_node_id is None
    engine.set_start_node("n1")
    assert engine.state.current_node_id == "n1"

def test_cascade_successful_step(basic_setup):
    engine, n1, n2, n3 = basic_setup
    engine.set_start_node("n1")
    evaluator = MockEvaluator(should_pass=True)
    
    result = engine.run_step(node_type="step", evaluator=evaluator)
    
    assert result["action"] == "proceed"
    assert result["next_node_id"] == "n2"
    assert "n1" in engine.state.verified_path
    assert n1.status == NodeStatus.VERIFIED

def test_cascade_failure_and_backtrack(basic_setup):
    engine, n1, n2, n3 = basic_setup
    engine.set_start_node("n1")
    evaluator = MockEvaluator(should_pass=True)
    
    # First step passes
    engine.run_step(node_type="step", evaluator=evaluator)
    
    # Now at n2, it fails
    evaluator.should_pass = False
    result = engine.run_step(node_type="step", evaluator=evaluator)
    
    assert result["action"] == "backtrack"
    assert result["next_node_id"] == "n3"
    assert "n2" in engine.state.failed_nodes
    assert engine.graph.get_node("n2").status == NodeStatus.PRUNED

def test_cascade_failure_mode_detection(basic_setup):
    engine, n1, n2, n3 = basic_setup
    # Inject content that should trigger an adversarial detection
    n1.content = "Are you sure? Double check everything."
    engine.set_start_node("n1")
    
    # Even if it passes checklist, detector should flag it
    evaluator = MockEvaluator(should_pass=True)
    result = engine.run_step(node_type="step", evaluator=evaluator)
    
    assert result["detections"]["self_doubt"].detected is True
    assert result["action"] == "stop" # Fails, and n1 has no alternatives

def test_run_to_completion(basic_setup):
    engine, n1, n2, n3 = basic_setup
    engine.set_start_node("n1")
    evaluator = MockEvaluator(should_pass=True)
    
    history = engine.run_to_completion(node_type_map={"n1": "step", "n2": "step"}, evaluator=evaluator)
    
    assert len(history) == 2
    assert history[0]["node_id"] == "n1"
    assert history[1]["node_id"] == "n2"
    assert history[1]["action"] == "finish"
    assert engine.state.verified_path == ["n1", "n2"]

def test_get_verified_trajectory(basic_setup):
    engine, n1, n2, n3 = basic_setup
    engine.state.verified_path = ["n1", "n2"]
    trajectory = engine.get_verified_trajectory()
    
    assert len(trajectory) == 2
    assert trajectory[0].id == "n1"
    assert trajectory[1].id == "n2"
