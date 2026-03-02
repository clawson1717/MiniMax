from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class NodeStatus(Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    PRUNED = "PRUNED"

@dataclass
class ChecklistItem:
    criterion: str
    passed: bool
    evidence: str

@dataclass
class TrajectoryNode:
    id: str
    content: str
    checklist_items: List[ChecklistItem] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
