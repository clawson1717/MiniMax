"""
FLOW-HEAL Payload Module

Defines the core data structures for reasoning steps in the self-correcting pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from enum import Enum
import json
from pydantic import BaseModel, Field
from typing_extensions import Literal

# ============================================================================
# REASONING STEP PAYLOAD
# ============================================================================

class StepStatus(str, Enum):
    """Status of a reasoning step."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    HEALED = "healed"

class AgentRole(str, Enum):
    """Role of the agent in the reasoning step."""
    THINKER = "thinker"          # Primary reasoning agent
    CRITIC = "critic"            # Critical evaluation agent
    PLANNER = "planner"          # Planning and strategy agent
    RESEARCHER = "researcher"    # Information gathering agent
    VALIDATOR = "validator"      # Logic and fact validation agent
    HEALER = "healer"            # Self-correction/healing agent

@dataclass
class CausalDependency:
    """Represents a causal relationship between reasoning steps."""
    parent_step_id: str
    relationship_type: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_step_id": self.parent_step_id,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence
        }

@dataclass
class ReasoningContext:
    """Contextual metadata for a reasoning step."""
    turn_number: int = 0
    session_id: str = ""
    conversation_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    domain: Optional[str] = None
    task_type: Optional[str] = None

class ReasoningStep(BaseModel):
    """
    A single step in a multi-agent reasoning pipeline.
    
    This class captures all metadata needed for the FLOW-HEAL system to:
    1. Track reasoning dependencies (DIG - Dynamic Interaction Graph)
    2. Sense semantic noise and uncertainty
    3. Identify root causes for healing
    4. Validate healed reasoning paths
    """
    # Core identifiers
    step_id: str = Field(default_factory=lambda: f"step_{datetime.now().timestamp()}")
    agent_id: str
    agent_role: AgentRole
    intent_hash: str  # Hash of the intent/purpose of this step
    
    # Content and reasoning
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)  # Additional structured data
    
    # Causal tracking (DIG)
    causal_dependencies: List[CausalDependency] = Field(default_factory=list)
    causal_children: List[str] = Field(default_factory=list)  # Step IDs that depend on this
    
    # Quality metrics
    uncertainty_score: float = 0.5  # 0 (certain) to 1 (uncertain)
    coherence_score: float = 0.5    # Semantic coherence with previous steps
    logic_score: float = 0.5        # Logical consistency score
    
    # Status tracking
    status: StepStatus = StepStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    healing_attempts: int = 0
    
    # Context
    context: ReasoningContext = Field(default_factory=ReasoningContext)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "intent_hash": self.intent_hash,
            "content": self.content,
            "payload": self.payload,
            "causal_dependencies": [cd.to_dict() for cd in self.causal_dependencies],
            "causal_children": self.causal_children,
            "uncertainty_score": self.uncertainty_score,
            "coherence_score": self.coherence_score,
            "logic_score": self.logic_score,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "healing_attempts": self.healing_attempts,
            "context": {
                "turn_number": self.context.turn_number,
                "session_id": self.context.session_id,
                "conversation_id": self.context.conversation_id,
                "timestamp": self.context.timestamp.isoformat(),
                "domain": self.context.domain,
                "task_type": self.context.task_type
            },
            "metadata": self.metadata
        }
    
    def update_status(self, new_status: StepStatus) -> None:
        """Update the status of the reasoning step."""
        self.status = new_status
        if new_status == StepStatus.EXECUTING:
            self.started_at = datetime.now()
        elif new_status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.HEALED]:
            self.completed_at = datetime.now()
    
    def add_causal_child(self, child_step_id: str) -> None:
        """Add a child step that depends on this step."""
        if child_step_id not in self.causal_children:
            self.causal_children.append(child_step_id)
    
    def calculate_overall_quality(self) -> float:
        """Calculate an overall quality score (0-1)."""
        # Weighted average of quality metrics
        weights = {
            'uncertainty': 0.4,
            'coherence': 0.3,
            'logic': 0.3
        }
        return (
            self.uncertainty_score * weights['uncertainty'] +
            self.coherence_score * weights['coherence'] +
            self.logic_score * weights['logic']
        )

# ============================================================================
# REASONING SESSION STATE
# ============================================================================

class ReasoningSession(BaseModel):
    """
    Tracks the state of an entire reasoning session.
    
    This is used by the FLOW-HEAL system to maintain session context
    and coordinate multi-agent collaboration.
    """
    session_id: str
    conversation_id: str
    steps: Dict[str, ReasoningStep] = Field(default_factory=dict)
    dig: Dict[str, List[str]] = Field(default_factory=dict)  # Dynamic Interaction Graph
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the session."""
        self.steps[step.step_id] = step
        # Update DIG
        for dependency in step.causal_dependencies:
            if dependency.parent_step_id not in self.dig:
                self.dig[dependency.parent_step_id] = []
            self.dig[dependency.parent_step_id].append(step.step_id)
    
    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """Retrieve a step by its ID."""
        return self.steps.get(step_id)
    
    def get_children(self, step_id: str) -> List[str]:
        """Get all child steps that depend on this step."""
        return self.dig.get(step_id, [])
    
    def get_parents(self, step_id: str) -> List[str]:
        """Get all parent steps that this step depends on."""
        parents = []
        for parent, children in self.dig.items():
            if step_id in children:
                parents.append(parent)
        return parents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire session state to dictionary."""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "dig": self.dig
        }

# ============================================================================
# SERIALIZATION UTILITIES
# ============================================================================

def save_session_to_json(session: ReasoningSession, filepath: str) -> None:
    """Save a reasoning session to a JSON file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)

def load_session_from_json(filepath: str) -> ReasoningSession:
    """Load a reasoning session from a JSON file."""
    import json
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Reconstruct session (simplified for demo)
    from pydantic import parse_obj_as
    session = ReasoningSession(
        session_id=data['session_id'],
        conversation_id=data['conversation_id']
    )
    for step_id, step_data in data['steps'].items():
        step = ReasoningStep(**step_data)
        session.add_step(step)
    return session

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a reasoning step
    from uuid import uuid4
    
    step1 = ReasoningStep(
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_2025_04_04_01",
        content="I will create a detailed project plan for the FLOW-HEAL system.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8,
        causal_dependencies=[
            CausalDependency(parent_step_id="initial_prompt", relationship_type="follows_from")
        ]
    )
    
    print("Created ReasoningStep:")
    print(f"Step ID: {step1.step_id}")
    print(f"Status: {step1.status}")
    print(f"Overall Quality: {step1.calculate_overall_quality():.2f}")
    
    # Example: Update status
    step1.update_status(StepStatus.EXECUTING)
    print(f"\nUpdated Status: {step1.status}")
    
    # Example: Create a session
    session = ReasoningSession(
        session_id="session_001",
        conversation_id="conv_001"
    )
    session.add_step(step1)
    
    print(f"\nSession DIG: {session.dig}")
    print(f"Session Steps: {list(session.steps.keys())}")
