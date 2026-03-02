import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.uncertainty import UncertaintyEstimator

class TrajectoryLogger:
    """
    Captures agent actions and environment states as a directed graph.
    Nodes represent states/observations, and edges represent actions/tool calls.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.current_node_id = None

    def add_node(self, 
                 observation: Any, 
                 metadata: Optional[Dict[str, Any]] = None,
                 votes: Optional[List[Any]] = None,
                 probabilities: Optional[List[float]] = None) -> str:
        """
        Adds a node to the trajectory graph.
        
        Args:
            observation: The state or observation at this point.
            metadata: Additional info (uncertainty, model name, etc.)
            votes: Optional list of votes to calculate uncertainty.
            probabilities: Optional list of probabilities to calculate uncertainty.
            
        Returns:
            The unique ID of the added node.
        """
        node_id = str(uuid.uuid4())
        meta = metadata or {}
        
        if votes is not None:
            meta["uncertainty"] = UncertaintyEstimator.from_votes(votes)
        elif probabilities is not None:
            meta["uncertainty"] = UncertaintyEstimator.from_probs(probabilities)
            
        self.nodes[node_id] = {
            "id": node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observation": observation,
            "metadata": meta
        }
        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str, action: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a directed edge between two nodes.
        
        Args:
            from_node_id: ID of the starting node.
            to_node_id: ID of the ending node.
            action: The action or tool call that led to the transition.
            metadata: Additional info about the action.
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found.")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found.")

        edge = {
            "from": from_node_id,
            "to": to_node_id,
            "action": action,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the graph."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges
        }

    def save(self, filepath: str):
        """Serializes the graph to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryLogger':
        """Loads a graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger = cls()
        for node in data["nodes"]:
            logger.nodes[node["id"]] = node
        logger.edges = data["edges"]
        return logger
