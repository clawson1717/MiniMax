from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from src.node import TrajectoryNode, NodeStatus
from src.graph import TrajectoryGraph
from src.verifier import ChecklistVerifier, VerificationResult
from src.detector import FailureModeDetector, DetectionResult

@dataclass
class CascadeState:
    """
    Maintains the state of the verification cascade.
    """
    current_node_id: Optional[str] = None
    verified_path: List[str] = field(default_factory=list)
    pruned_branches: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)

class CascadeEngine:
    """
    Orchestrates the verification cascade through a trajectory graph.
    """
    def __init__(self, 
                 graph: TrajectoryGraph, 
                 verifier: ChecklistVerifier, 
                 detector: FailureModeDetector):
        self.graph = graph
        self.verifier = verifier
        self.detector = detector
        self.state = CascadeState()

    def set_start_node(self, node_id: str):
        """Sets the starting node for the cascade."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found in graph.")
        self.state.current_node_id = node_id

    def run_step(self, node_type: str, evaluator: Any = None) -> Dict[str, Any]:
        """
        Executes a single step of the cascade on the current node.
        
        Steps:
        1. Verify checklist.
        2. Detect failure modes.
        3. Decide next action (proceed, prune, or backtrack).
        """
        if not self.state.current_node_id:
            raise ValueError("No current node set. Call set_start_node first.")

        node_id = self.state.current_node_id
        node = self.graph.get_node(node_id)
        
        if not node:
            raise ValueError(f"Current node {node_id} not found in graph.")

        # 1. Verify checklist
        v_result = self.verifier.verify(node, node_type, evaluator)
        
        # 2. Detect failure modes
        d_results = self.detector.detect_all(node.content)
        any_failure_detected = any(res.detected for res in d_results.values())

        # 3. Decision Logic
        action = "proceed"
        next_node_id = None

        if v_result.overall_status == NodeStatus.FAILED or any_failure_detected:
            # Node failed verification or a failure mode was detected
            node.status = NodeStatus.FAILED
            self.state.failed_nodes.add(node_id)
            
            # Action: Prune this branch and try to backtrack/find alternative
            self.graph.prune_branch(node_id)
            self.state.pruned_branches.add(node_id)
            
            alternatives = self.graph.find_alternatives(node_id)
            valid_alternatives = [a for a in alternatives 
                                if a not in self.state.pruned_branches 
                                and a not in self.state.failed_nodes]
            
            if valid_alternatives:
                action = "backtrack"
                next_node_id = valid_alternatives[0]
            else:
                action = "stop"
                next_node_id = None
        else:
            # Node passed
            node.status = NodeStatus.VERIFIED
            if node_id not in self.state.verified_path:
                self.state.verified_path.append(node_id)
            
            # Attempt to proceed to children
            children = node.children_ids
            valid_children = [c for c in children 
                            if c not in self.state.pruned_branches 
                            and c not in self.state.failed_nodes]
            
            if valid_children:
                action = "proceed"
                next_node_id = valid_children[0]
            else:
                # No children, but we passed. Could be the end of a trajectory.
                action = "finish"
                next_node_id = None

        # Update state for next step
        self.state.current_node_id = next_node_id
        
        return {
            "node_id": node_id,
            "verification": v_result,
            "detections": d_results,
            "action": action,
            "next_node_id": next_node_id
        }

    def run_to_completion(self, node_type_map: Dict[str, str], evaluator: Any = None) -> List[Dict[str, Any]]:
        """
        Runs the cascade until it reaches a 'finish' or 'stop' action.
        
        Args:
            node_type_map: Mapping of node IDs or patterns to node types for the verifier.
        """
        history = []
        while self.state.current_node_id:
            node_id = self.state.current_node_id
            # Determine node type (fallback to "default" if not specified)
            node_type = node_type_map.get(node_id, "default")
            
            step_result = self.run_step(node_type, evaluator)
            history.append(step_result)
            
            if step_result["action"] in ["finish", "stop"]:
                break
                
        return history

    def get_verified_trajectory(self) -> List[TrajectoryNode]:
        """Returns the full list of verified nodes in the current path."""
        return [self.graph.get_node(node_id) for node_id in self.state.verified_path]
