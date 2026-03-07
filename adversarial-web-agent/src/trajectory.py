from typing import Any, List, Dict, Tuple

class TrajectoryGraph:
    """
    Tracks action states of a web agent as a directed graph.
    Nodes represent states, and edges represent actions leading to next states.
    """
    def __init__(self):
        self.nodes: Dict[Any, List[Tuple[Any, Any]]] = {}
        self.trajectory: List[Tuple[Any, Any, Any]] = []

    def add_step(self, state: Any, action: Any, next_state: Any) -> None:
        """
        Adds a transition to the graph.
        
        Args:
            state: The starting state.
            action: The action taken.
            next_state: The resulting state.
        """
        if state not in self.nodes:
            self.nodes[state] = []
        
        # Add transition if not already present
        transition = (action, next_state)
        if transition not in self.nodes[state]:
            self.nodes[state].append(transition)
            
        # Ensure next_state is in nodes even if it has no outgoing edges yet
        if next_state not in self.nodes:
            self.nodes[next_state] = []
            
        self.trajectory.append((state, action, next_state))

    def get_path(self) -> List[Tuple[Any, Any, Any]]:
        """
        Returns the sequence of (state, action, next_state) steps taken.
        """
        return self.trajectory

    def reset(self) -> None:
        """
        Resets the graph and trajectory.
        """
        self.nodes = {}
        self.trajectory = []
