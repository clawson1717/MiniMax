"""Checklist-based task tracking."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChecklistItem:
    """Single item in a checklist."""
    id: str
    description: str
    status: CheckStatus = CheckStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "metadata": self.metadata,
        }


class Checklist:
    """
    Task checklist for tracking progress and dependencies.
    
    Inspired by CATTS checklist-based approach.
    """
    
    def __init__(self, name: str = "Task Checklist"):
        """Initialize checklist.
        
        Args:
            name: Name of the checklist.
        """
        self.name = name
        self.items: Dict[str, ChecklistItem] = {}
        
    def add_item(self, item_id: str, description: str, 
                 depends_on: Optional[List[str]] = None) -> ChecklistItem:
        """Add an item to the checklist.
        
        Args:
            item_id: Unique identifier for the item.
            description: Description of what needs to be done.
            depends_on: List of item IDs that must be completed first.
            
        Returns:
            The created ChecklistItem.
        """
        item = ChecklistItem(
            id=item_id,
            description=description,
            depends_on=depends_on or [],
        )
        self.items[item_id] = item
        return item
    
    def get_item(self, item_id: str) -> Optional[ChecklistItem]:
        """Get an item by ID."""
        return self.items.get(item_id)
    
    def update_status(self, item_id: str, status: CheckStatus) -> bool:
        """Update the status of an item.
        
        Args:
            item_id: ID of the item to update.
            status: New status.
            
        Returns:
            True if update was successful.
        """
        if item_id not in self.items:
            return False
            
        # Check dependencies for completion
        if status == CheckStatus.COMPLETED:
            for dep_id in self.items[item_id].depends_on:
                if dep_id in self.items:
                    if self.items[dep_id].status != CheckStatus.COMPLETED:
                        return False
                        
        self.items[item_id].status = status
        return True
    
    def complete(self, item_id: str) -> bool:
        """Mark an item as completed.
        
        Args:
            item_id: ID of the item to complete.
            
        Returns:
            True if successful.
        """
        return self.update_status(item_id, CheckStatus.COMPLETED)
    
    def is_complete(self) -> bool:
        """Check if all items are completed."""
        return all(
            item.status == CheckStatus.COMPLETED 
            for item in self.items.values()
        )
    
    def get_ready_items(self) -> List[ChecklistItem]:
        """Get items that are ready to be worked on (all deps satisfied)."""
        ready = []
        for item in self.items.values():
            if item.status != CheckStatus.PENDING:
                continue
            deps_satisfied = all(
                self.items.get(dep_id, ChecklistItem(id="", description="")).status == CheckStatus.COMPLETED
                for dep_id in item.depends_on
            )
            if deps_satisfied:
                ready.append(item)
        return ready
    
    def get_pending_items(self) -> List[ChecklistItem]:
        """Get all pending items."""
        return [item for item in self.items.values() if item.status == CheckStatus.PENDING]
    
    def get_completed_items(self) -> List[ChecklistItem]:
        """Get all completed items."""
        return [item for item in self.items.values() if item.status == CheckStatus.COMPLETED]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary."""
        total = len(self.items)
        completed = len(self.get_completed_items())
        pending = len(self.get_pending_items())
        ready = len(self.get_ready_items())
        
        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "ready": ready,
            "percent_complete": (completed / total * 100) if total > 0 else 0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checklist to dictionary."""
        return {
            "name": self.name,
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "progress": self.get_progress(),
        }
    
    def reset(self) -> None:
        """Reset all items to pending."""
        for item in self.items.values():
            item.status = CheckStatus.PENDING