"""
FLOW-HEAL Dynamic Interaction Graph Tracker

Implements the ReasoningDIG class for tracking causal dependencies and
interaction graphs in multi-agent reasoning pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import json
from typing_extensions import Literal
from pydantic import BaseModel, Field, ConfigDict
import networkx as nx
from pydantic import BaseModel, Field

# ============================================================================
# DYNAMIC INTERACTION GRAPH (DIG)
# ============================================================================

class CausalRelationship(str, Enum):
    """Types of causal relationships between reasoning steps."""
    DIRECT_CAUSE = "direct_cause"          # Direct causal influence
    INDIRECT_CAUSE = "indirect_cause"      # Indirect causal influence
    EVIDENCE_FOR = "evidence_for"          # Provides evidence supporting
    EVIDENCE_AGAINST = "evidence_against"  # Provides evidence contradicting
    FOLLOWS_FROM = "follows_from"          # Logical consequence
    CONTEXTUAL = "contextual"              # Provides context or background
    CORRELATION = "correlation"            # Statistical correlation
    DEPENDENCY = "dependency"              # General dependency

@dataclass
class CausalLink:
    """Represents a causal link between two reasoning steps in the DIG."""
    source_step_id: str
    target_step_id: str
    relationship_type: CausalRelationship
    confidence: float = 1.0
    strength: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_step_id": self.source_step_id,
            "target_step_id": self.target_step_id,
            "relationship_type": self.relationship_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class ReasoningNode:
    """A node in the Dynamic Interaction Graph representing a reasoning step."""
    step_id: str
    agent_id: str
    agent_role: str
    intent_hash: str
    content: str
    uncertainty_score: float = 0.5
    coherence_score: float = 0.5
    logic_score: float = 0.5
    quality_score: float = 0.5
    timestamp: datetime = Field(default_factory=datetime.now)
    causal_in_degree: int = 0
    causal_out_degree: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_quality_scores(self, payload: Dict[str, float]) -> None:
        """Update quality scores from payload metrics."""
        self.uncertainty_score = payload.get('uncertainty_score', self.uncertainty_score)
        self.coherence_score = payload.get('coherence_score', self.coherence_score)
        self.logic_score = payload.get('logic_score', self.logic_score)
        self.quality_score = self.calculate_overall_quality()
    
    def calculate_overall_quality(self) -> float:
        """Calculate weighted quality score."""
        weights = {'uncertainty': 0.4, 'coherence': 0.3, 'logic': 0.3}
        return (1 - self.uncertainty_score) * weights['uncertainty'] + \
               self.coherence_score * weights['coherence'] + \
               self.logic_score * weights['logic']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "intent_hash": self.intent_hash,
            "content": self.content,
            "uncertainty_score": self.uncertainty_score,
            "coherence_score": self.coherence_score,
            "logic_score": self.logic_score,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            "causal_in_degree": self.causal_in_degree,
            "causal_out_degree": self.causal_out_degree,
            "metadata": self.metadata
        }

class ReasoningDIG(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """
    Dynamic Interaction Graph (DIG) for tracking reasoning processes.
    
    The DIG captures causal relationships between reasoning steps,
    enabling the FLOW-HEAL system to identify where errors occur and
    target healing interventions precisely.
    """
    session_id: str
    conversation_id: str
    graph: nx.DiGraph = Field(default_factory=lambda: nx.DiGraph())
    nodes: Dict[str, ReasoningNode] = Field(default_factory=dict)
    links: Dict[Tuple[str, str], CausalLink] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the DIG."""
        self.nodes[node.step_id] = node
        self.graph.add_node(node.step_id, node_data=node.to_dict())
    
    def add_link(self, link: CausalLink) -> None:
        """Add a causal link between two nodes."""
        key = (link.source_step_id, link.target_step_id)
        self.links[key] = link
        self.graph.add_edge(link.source_step_id, link.target_step_id, link_data=link.to_dict())
        # Update degrees
        if link.source_step_id in self.nodes:
            self.nodes[link.source_step_id].causal_out_degree += 1
        if link.target_step_id in self.nodes:
            self.nodes[link.target_step_id].causal_in_degree += 1
    
    def get_node(self, step_id: str) -> Optional[ReasoningNode]:
        """Retrieve a node by step ID."""
        return self.nodes.get(step_id)
    
    def get_neighbors(self, step_id: str, direction: Literal['in', 'out'] = 'out') -> List[str]:
        """Get neighboring node IDs."""
        if direction == 'out':
            return list(self.graph.successors(step_id))
        else:
            return list(self.graph.predecessors(step_id))
    
    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Get shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_all_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """Get all paths between two nodes."""
        return list(nx.all_simple_paths(self.graph, source=source_id, target=target_id))
    
    def calculate_node_importance(self, step_id: str) -> Dict[str, float]:
        """Calculate importance metrics for a node."""
        if step_id not in self.nodes:
            return {}
        
        node = self.nodes[step_id]
        in_degree = node.causal_in_degree
        out_degree = node.causal_out_degree
        total_nodes = len(self.nodes)
        
        # Betweenness centrality (approximate)
        betweenness = 0.0
        if total_nodes > 2:
            try:
                betweenness = nx.betweenness_centrality(self.graph)[step_id]
            except:
                betweenness = 0.0
        
        # Closeness centrality
        closeness = 0.0
        try:
            closeness = nx.closeness_centrality(self.graph, step_id)
        except:
            closeness = 0.0
        
        return {
            "betweenness_centrality": betweenness,
            "closeness_centrality": closeness,
            "in_degree_ratio": in_degree / max(total_nodes - 1, 1),
            "out_degree_ratio": out_degree / max(total_nodes - 1, 1),
            "connectivity_score": (in_degree + out_degree) / (2 * (total_nodes - 1))
        }
    
    def identify_root_causes(self, target_step_id: str) -> List[str]:
        """
        Identify potential root cause nodes that lead to the target.
        
        This is used by the Regulator module to find the source of errors.
        """
        if target_step_id not in self.nodes:
            return []
        
        # Get all ancestors of the target step
        ancestors = set()
        stack = [target_step_id]
        
        while stack:
            current = stack.pop()
            parents = self.get_neighbors(current, direction='in')
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        
        # Rank root causes by quality scores and connectivity
        ranked_causes = []
        for ancestor in ancestors:
            if ancestor in self.nodes:
                node = self.nodes[ancestor]
                importance = self.calculate_node_importance(ancestor)
                ranked_causes.append({
                    "step_id": ancestor,
                    "quality_score": node.quality_score,
                    "importance_score": importance.get("connectivity_score", 0),
                    "uncertainty": node.uncertainty_score
                })
        
        # Sort by: lowest quality first (most likely to be problematic)
        # Then by highest uncertainty
        ranked_causes.sort(key=lambda x: (x["quality_score"], -x["uncertainty"]))
        return [item["step_id"] for item in ranked_causes]
    
    def visualize_graph(self, highlight_nodes: List[str] = None) -> str:
        """
        Generate a Graphviz DOT representation for visualization.
        
        Returns: DOT format string
        """
        import graphviz as gv
        
        dot = gv.Digraph()
        
        # Add nodes with attributes
        for step_id, node in self.nodes.items():
            node_data = self.graph.nodes[step_id]['node_data']
            quality = node_data.get('quality_score', 0.5)
            uncertainty = node_data.get('uncertainty_score', 0.5)
            
            # Color based on quality (green=good, red=poor)
            if quality > 0.7:
                color = "palegreen2"
            elif quality > 0.4:
                color = "gold"
            else:
                color = "lightcoral"
            
            # Shape based on role
            shape = "ellipse"
            if node_data.get('agent_role', '') in ['healer', 'validator']:
                shape = "diamond"
            
            dot.node(
                step_id,
                label=f"{step_id[:20]}...",
                shape=shape,
                style="filled",
                color=color,
                tooltip=str(node_data)
            )
        
        # Add edges
        for (src, dst), link in self.links.items():
            link_data = self.graph.edges[src, dst]['link_data']
            relationship = link_data.get('relationship_type', 'dependency')
            
            # Edge color based on confidence
            confidence = link_data.get('confidence', 1.0)
            if confidence > 0.8:
                edgecolor = "darkgreen"
            elif confidence > 0.5:
                edgecolor = "goldenrod"
            else:
                edgecolor = "red"
            
            dot.edge(
                src,
                dst,
                label=relationship.replace('_', ' '),
                color=edgecolor,
                penwidth=str(1 + 2 * confidence)
            )
        
        # Highlight specific nodes if requested
        if highlight_nodes:
            for node_id in highlight_nodes:
                if node_id in self.nodes:
                    dot.node(node_id, **{"style": "striped", "color": "yellow"})
        
        return dot.source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire DIG to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "links": {str(k): v.to_dict() for k, v in self.links.items()},
            "metadata": self.metadata,
            "graph_stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.links),
                "density": nx.density(self.graph),
                "connected_components": nx.number_connected_components(self.graph.to_undirected())
            }
        }
    
    def save_to_json(self, filepath: str) -> None:
        """Save DIG to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ReasoningDIG':
        """Load DIG from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dig = cls(
            session_id=data['session_id'],
            conversation_id=data['conversation_id']
        )
        
        # Reconstruct nodes
        for step_id, node_data in data['nodes'].items():
            node = ReasoningNode(**node_data)
            dig.add_node(node)
        
        # Reconstruct links
        for (src_dst_key, link_data) in data['links'].items():
            # Parse the tuple key
            import ast
            src, dst = ast.literal_eval(src_dst_key)
            link = CausalLink(**link_data)
            dig.add_link(link)
        
        return dig

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a sample DIG
    from uuid import uuid4
    
    # Initialize DIG
    dig = ReasoningDIG(
        session_id="session_001",
        conversation_id="conv_001"
    )
    
    # Create sample nodes
    node1 = ReasoningNode(
        step_id="step_001",
        agent_id="planner",
        agent_role="planner",
        intent_hash="plan_initial",
        content="I will create an initial project plan for FLOW-HEAL."
    )
    
    node2 = ReasoningNode(
        step_id="step_002",
        agent_id="researcher",
        agent_role="researcher",
        intent_hash="research_requirements",
        content="I will research the requirements for the FLOW-HEAL system from the three source papers."
    )
    
    node3 = ReasoningNode(
        step_id="step_003",
        agent_id="thinker",
        agent_role="thinker",
        intent_hash="design_architecture",
        content="Based on the research, I will design the overall architecture for FLOW-HEAL."
    )
    
    # Add nodes to DIG
    dig.add_node(node1)
    dig.add_node(node2)
    dig.add_node(node3)
    
    # Create causal links
    link1 = CausalLink(
        source_step_id="step_001",
        target_step_id="step_002",
        relationship_type=CausalRelationship.CONTEXTUAL,
        confidence=0.9
    )
    
    link2 = CausalLink(
        source_step_id="step_002",
        target_step_id="step_003",
        relationship_type=CausalRelationship.EVIDENCE_FOR,
        confidence=0.85
    )
    
    # Add links to DIG
    dig.add_link(link1)
    dig.add_link(link2)
    
    # Demonstrate functionality
    print("=== FLOW-HEAL Dynamic Interaction Graph ===")
    print(f"Session: {dig.session_id}")
    print(f"Conversation: {dig.conversation_id}")
    print(f"Nodes: {len(dig.nodes)}")
    print(f"Links: {len(dig.links)}")
    
    print("\nNode Details:")
    for step_id, node in dig.nodes.items():
        print(f"  {step_id}: {node.agent_role} - Quality: {node.quality_score:.2f}")
    
    print("\nGraph Visualization (DOT format):")
    print(dig.visualize_graph())
    
    print("\nIdentifying Root Causes for step_003:")
    root_causes = dig.identify_root_causes("step_003")
    for cause in root_causes:
        print(f"  - {cause}: Quality={dig.get_node(cause).quality_score:.2f}")
    
    # Save to JSON
    dig.save_to_json("sample_flow_heal.dig.json")
    print("\nSaved DIG to 'sample_flow_heal.dig.json'")
