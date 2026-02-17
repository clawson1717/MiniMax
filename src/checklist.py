"""Checklist-based task tracking with CM2-style fine-grained evaluation.

CM2 (Checklist-based Multi-turn Metrics) uses binary checklists to provide
fine-grained rewards that outperform sparse outcome rewards by 8-12 points on
multi-turn benchmarks.
"""

from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re


class CheckStatus(Enum):
    """Status of a checklist item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChecklistItem:
    """Single item in a checklist (legacy, kept for backward compatibility)."""
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


# ============================================================================
# CM2-Style Checklist System
# ============================================================================

@dataclass
class ChecklistCriterion:
    """
    A single criterion for evaluation with CM2-style binary checking.
    
    Attributes:
        description: Human-readable description of what to check
        weight: Importance weight (0.0-1.0, default 1.0)
        check_function: Function that takes step_data and returns bool
        is_satisfied: Whether this criterion has been met
    """
    description: str
    weight: float = 1.0
    check_function: Optional[Callable[[Dict[str, Any]], bool]] = None
    is_satisfied: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """Validate weight is in valid range."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")
    
    def evaluate(self, step_data: Dict[str, Any]) -> bool:
        """
        Evaluate this criterion against step data.
        
        Once satisfied, a criterion remains satisfied (monotonic).
        
        Args:
            step_data: Dictionary containing step information
            
        Returns:
            True if criterion is now satisfied
        """
        if self.is_satisfied:
            return True
            
        if self.check_function is None:
            return False
            
        try:
            result = self.check_function(step_data)
            if result:
                self.is_satisfied = True
            return result
        except Exception:
            # If check function fails, don't mark as satisfied
            return False
    
    def reset(self) -> None:
        """Reset criterion to unsatisfied state."""
        self.is_satisfied = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "weight": self.weight,
            "is_satisfied": self.is_satisfied,
        }


class Checklist:
    """
    Task checklist with CM2-style fine-grained binary criteria.
    
    Provides step-by-step feedback on partial completion and enables
    credit assignment across multi-turn interactions.
    
    Inspired by CATTS checklist-based approach and CM2 evaluation metrics.
    """
    
    def __init__(self, name: str = "Task Checklist", task_type: Optional[str] = None):
        """Initialize checklist.
        
        Args:
            name: Name of the checklist.
            task_type: Type of task (navigation, search, form, etc.)
        """
        self.name = name
        self.task_type = task_type
        self.items: Dict[str, ChecklistItem] = {}
        self.criteria: List[ChecklistCriterion] = []
        
    def add_item(self, item_id: str, description: str, 
                 depends_on: Optional[List[str]] = None) -> ChecklistItem:
        """Add an item to the checklist (legacy method).
        
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
    
    def add_criterion(self, description: str, weight: float = 1.0,
                      check_function: Optional[Callable[[Dict[str, Any]], bool]] = None) -> ChecklistCriterion:
        """
        Add a CM2-style criterion to the checklist.
        
        Args:
            description: Human-readable description of what to check
            weight: Importance weight (0.0-1.0)
            check_function: Function that takes step_data dict and returns bool
            
        Returns:
            The created ChecklistCriterion
        """
        criterion = ChecklistCriterion(
            description=description,
            weight=weight,
            check_function=check_function
        )
        self.criteria.append(criterion)
        return criterion
    
    def get_item(self, item_id: str) -> Optional[ChecklistItem]:
        """Get an item by ID (legacy method)."""
        return self.items.get(item_id)
    
    def update_status(self, item_id: str, status: CheckStatus) -> bool:
        """Update the status of an item (legacy method).
        
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
        """Mark an item as completed (legacy method).
        
        Args:
            item_id: ID of the item to complete.
            
        Returns:
            True if successful.
        """
        return self.update_status(item_id, CheckStatus.COMPLETED)
    
    def evaluate(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all criteria against step data (CM2-style).
        
        Args:
            step_data: Dictionary containing step information like:
                - action: The action taken
                - observation: Observation from environment
                - url: Current URL
                - metadata: Additional step metadata
                
        Returns:
            Dictionary with evaluation results:
                - newly_satisfied: List of descriptions for newly satisfied criteria
                - all_satisfied: List of all currently satisfied criteria
                - score: Current aggregate score
                - progress: Progress percentage
        """
        newly_satisfied = []
        
        for criterion in self.criteria:
            was_satisfied = criterion.is_satisfied
            is_now_satisfied = criterion.evaluate(step_data)
            
            if is_now_satisfied and not was_satisfied:
                newly_satisfied.append(criterion.description)
        
        all_satisfied = [c.description for c in self.criteria if c.is_satisfied]
        
        return {
            "newly_satisfied": newly_satisfied,
            "all_satisfied": all_satisfied,
            "score": self.get_score(),
            "progress": self.get_progress_percentage(),
            "is_complete": self.is_complete(),
        }
    
    def get_score(self) -> float:
        """
        Get aggregate checklist score (weighted average of satisfied criteria).
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not self.criteria:
            return 0.0
            
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight == 0:
            return 0.0
            
        satisfied_weight = sum(c.weight for c in self.criteria if c.is_satisfied)
        return satisfied_weight / total_weight
    
    def get_progress_percentage(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Percentage between 0 and 100
        """
        return self.get_score() * 100
    
    def is_complete(self) -> bool:
        """Check if all criteria are satisfied."""
        if not self.criteria:
            # Fall back to legacy items if no criteria defined
            return all(
                item.status == CheckStatus.COMPLETED 
                for item in self.items.values()
            )
        return all(c.is_satisfied for c in self.criteria)
    
    def get_failed_criteria(self) -> List[ChecklistCriterion]:
        """
        Get list of criteria that have not been satisfied.
        
        Returns:
            List of unsatisfied ChecklistCriterion objects
        """
        return [c for c in self.criteria if not c.is_satisfied]
    
    def get_satisfied_criteria(self) -> List[ChecklistCriterion]:
        """
        Get list of criteria that have been satisfied.
        
        Returns:
            List of satisfied ChecklistCriterion objects
        """
        return [c for c in self.criteria if c.is_satisfied]
    
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
        """Get progress summary (legacy method)."""
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
        result = {
            "name": self.name,
            "task_type": self.task_type,
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "criteria": [c.to_dict() for c in self.criteria],
            "score": self.get_score(),
            "is_complete": self.is_complete(),
        }
        
        # Include legacy progress if items exist
        if self.items:
            result["progress"] = self.get_progress()
            
        return result
    
    def reset(self) -> None:
        """Reset all items and criteria."""
        for item in self.items.values():
            item.status = CheckStatus.PENDING
        for criterion in self.criteria:
            criterion.reset()


class ChecklistEvaluator:
    """
    Evaluates agent steps against checklists and provides feedback.
    
    Enables fine-grained credit assignment across multi-turn interactions
    as described in the CM2 paper.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.checklists: Dict[str, Checklist] = {}
        self.current_checklist: Optional[Checklist] = None
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def create_checklist(self, task_type: str, name: Optional[str] = None) -> Checklist:
        """
        Factory method to create built-in checklists for common web tasks.
        
        Args:
            task_type: Type of task ('navigation', 'search', 'form', etc.)
            name: Optional custom name for the checklist
            
        Returns:
            Configured Checklist instance
        """
        checklist_name = name or f"{task_type.title()} Checklist"
        checklist = Checklist(name=checklist_name, task_type=task_type)
        
        if task_type == "navigation":
            self._setup_navigation_checklist(checklist)
        elif task_type == "search":
            self._setup_search_checklist(checklist)
        elif task_type == "form":
            self._setup_form_checklist(checklist)
        elif task_type == "information_extraction":
            self._setup_info_extraction_checklist(checklist)
        else:
            # Generic checklist with minimal criteria
            checklist.add_criterion(
                description="Started task execution",
                weight=0.2,
                check_function=lambda d: d.get("step_index", 0) > 0
            )
            checklist.add_criterion(
                description="Made progress toward goal",
                weight=0.8,
                check_function=lambda d: d.get("progress", False) is True
            )
        
        self.checklists[task_type] = checklist
        self.current_checklist = checklist
        return checklist
    
    def _setup_navigation_checklist(self, checklist: Checklist) -> None:
        """Setup criteria for navigation tasks."""
        checklist.add_criterion(
            description="Reached target domain",
            weight=0.3,
            check_function=lambda d: self._check_domain_match(d)
        )
        checklist.add_criterion(
            description="Found interactive element",
            weight=0.3,
            check_function=lambda d: self._check_interactive_element(d)
        )
        checklist.add_criterion(
            description="Successfully navigated to target page",
            weight=0.4,
            check_function=lambda d: self._check_target_page(d)
        )
    
    def _setup_search_checklist(self, checklist: Checklist) -> None:
        """Setup criteria for search tasks."""
        checklist.add_criterion(
            description="Located search interface",
            weight=0.2,
            check_function=lambda d: self._check_search_interface(d)
        )
        checklist.add_criterion(
            description="Submitted query",
            weight=0.3,
            check_function=lambda d: self._check_query_submitted(d)
        )
        checklist.add_criterion(
            description="Received search results",
            weight=0.25,
            check_function=lambda d: self._check_results_received(d)
        )
        checklist.add_criterion(
            description="Verified result relevance",
            weight=0.25,
            check_function=lambda d: self._check_result_relevance(d)
        )
    
    def _setup_form_checklist(self, checklist: Checklist) -> None:
        """Setup criteria for form tasks."""
        checklist.add_criterion(
            description="Located form",
            weight=0.2,
            check_function=lambda d: self._check_form_located(d)
        )
        checklist.add_criterion(
            description="Filled required fields",
            weight=0.3,
            check_function=lambda d: self._check_required_fields_filled(d)
        )
        checklist.add_criterion(
            description="Validated form input",
            weight=0.2,
            check_function=lambda d: self._check_input_validated(d)
        )
        checklist.add_criterion(
            description="Submitted successfully",
            weight=0.3,
            check_function=lambda d: self._check_form_submitted(d)
        )
    
    def _setup_info_extraction_checklist(self, checklist: Checklist) -> None:
        """Setup criteria for information extraction tasks."""
        checklist.add_criterion(
            description="Accessed target page",
            weight=0.25,
            check_function=lambda d: self._check_target_page(d)
        )
        checklist.add_criterion(
            description="Located relevant content",
            weight=0.35,
            check_function=lambda d: self._check_content_located(d)
        )
        checklist.add_criterion(
            description="Extracted required information",
            weight=0.4,
            check_function=lambda d: self._check_info_extracted(d)
        )
    
    # ============================================================================
    # Default Check Functions
    # ============================================================================
    
    def _check_domain_match(self, step_data: Dict[str, Any]) -> bool:
        """Check if target domain was reached."""
        url = step_data.get("url", "")
        target_domain = step_data.get("target_domain", "")
        if target_domain and target_domain in url:
            return True
        metadata = step_data.get("metadata", {})
        return metadata.get("domain_match", False)
    
    def _check_interactive_element(self, step_data: Dict[str, Any]) -> bool:
        """Check if interactive element was found."""
        observation = step_data.get("observation", "")
        interactive_patterns = ["click", "button", "link", "input", "submit", "href"]
        return any(pattern in observation.lower() for pattern in interactive_patterns)
    
    def _check_target_page(self, step_data: Dict[str, Any]) -> bool:
        """Check if target page was reached."""
        metadata = step_data.get("metadata", {})
        return metadata.get("target_reached", False) or metadata.get("success", False)
    
    def _check_search_interface(self, step_data: Dict[str, Any]) -> bool:
        """Check if search interface was found."""
        observation = step_data.get("observation", "")
        search_patterns = ["search", "query", "find", "input type=\"search\""]
        return any(pattern in observation.lower() for pattern in search_patterns)
    
    def _check_query_submitted(self, step_data: Dict[str, Any]) -> bool:
        """Check if query was submitted."""
        action = step_data.get("action", "")
        metadata = step_data.get("metadata", {})
        return "submit" in action.lower() or metadata.get("query_submitted", False)
    
    def _check_results_received(self, step_data: Dict[str, Any]) -> bool:
        """Check if search results were received."""
        observation = step_data.get("observation", "")
        result_patterns = ["result", "found", "matches", "showing", "page ", "items"]
        metadata = step_data.get("metadata", {})
        return any(pattern in observation.lower() for pattern in result_patterns) or \
               metadata.get("results_count", 0) > 0
    
    def _check_result_relevance(self, step_data: Dict[str, Any]) -> bool:
        """Check if results are relevant."""
        metadata = step_data.get("metadata", {})
        return metadata.get("relevance_verified", False) or metadata.get("found_match", False)
    
    def _check_form_located(self, step_data: Dict[str, Any]) -> bool:
        """Check if form was located."""
        observation = step_data.get("observation", "")
        return "<form" in observation.lower() or "form" in observation.lower()
    
    def _check_required_fields_filled(self, step_data: Dict[str, Any]) -> bool:
        """Check if required fields were filled."""
        metadata = step_data.get("metadata", {})
        return metadata.get("required_fields_filled", False) or \
               metadata.get("fields_completed", 0) > 0
    
    def _check_input_validated(self, step_data: Dict[str, Any]) -> bool:
        """Check if input was validated."""
        observation = step_data.get("observation", "")
        metadata = step_data.get("metadata", {})
        no_error = not any(err in observation.lower() for err in ["error", "invalid", "required"])
        return metadata.get("validated", False) or no_error
    
    def _check_form_submitted(self, step_data: Dict[str, Any]) -> bool:
        """Check if form was submitted successfully."""
        metadata = step_data.get("metadata", {})
        return metadata.get("form_submitted", False) or metadata.get("success", False)
    
    def _check_content_located(self, step_data: Dict[str, Any]) -> bool:
        """Check if relevant content was located."""
        metadata = step_data.get("metadata", {})
        return metadata.get("content_located", False) or metadata.get("target_found", False)
    
    def _check_info_extracted(self, step_data: Dict[str, Any]) -> bool:
        """Check if information was extracted."""
        metadata = step_data.get("metadata", {})
        return metadata.get("extracted_data") is not None or metadata.get("info_extracted", False)
    
    # ============================================================================
    # Evaluation Methods
    # ============================================================================
    
    def evaluate_step(self, step_data: Dict[str, Any], 
                      checklist: Optional[Checklist] = None) -> Dict[str, Any]:
        """
        Evaluate a single step against a checklist.
        
        Args:
            step_data: Step data to evaluate
            checklist: Checklist to use (defaults to current_checklist)
            
        Returns:
            Evaluation results
        """
        checklist = checklist or self.current_checklist
        if checklist is None:
            raise ValueError("No checklist provided or set as current")
        
        result = checklist.evaluate(step_data)
        result["step_index"] = step_data.get("step_index", len(self.evaluation_history))
        result["timestamp"] = step_data.get("timestamp")
        
        self.evaluation_history.append(result)
        return result
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations.
        
        Returns:
            Summary dictionary with scores and progress
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "final_score": 0.0,
                "is_complete": False,
            }
        
        # Get most recent complete evaluation
        final_eval = self.evaluation_history[-1]
        
        # Count criteria satisfaction progress
        criteria_progress = []
        for criterion in self.current_checklist.criteria if self.current_checklist else []:
            # Find when each criterion was first satisfied
            for i, eval_result in enumerate(self.evaluation_history):
                if criterion.description in eval_result.get("all_satisfied", []):
                    criteria_progress.append({
                        "criterion": criterion.description,
                        "satisfied_at_step": eval_result.get("step_index", i),
                        "weight": criterion.weight,
                    })
                    break
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "final_score": final_eval.get("score", 0.0),
            "final_progress": final_eval.get("progress", 0.0),
            "is_complete": final_eval.get("is_complete", False),
            "criteria_progress": criteria_progress,
            "evaluation_history": self.evaluation_history,
        }
    
    def reset(self) -> None:
        """Reset all checklists and evaluation history."""
        for checklist in self.checklists.values():
            checklist.reset()
        self.evaluation_history = []


# ============================================================================
# Utility Functions
# ============================================================================

def create_checklist(task_type: str, name: Optional[str] = None) -> Checklist:
    """
    Convenience function to create a checklist for a specific task type.
    
    Args:
        task_type: Type of task ('navigation', 'search', 'form', etc.)
        name: Optional custom name
        
    Returns:
        Configured Checklist
    """
    evaluator = ChecklistEvaluator()
    return evaluator.create_checklist(task_type, name)


def evaluate_trajectory(trajectory: List[Dict[str, Any]], 
                        task_type: str) -> Dict[str, Any]:
    """
    Evaluate an entire trajectory against a checklist.
    
    Args:
        trajectory: List of step data dictionaries
        task_type: Type of task
        
    Returns:
        Complete evaluation results
    """
    evaluator = ChecklistEvaluator()
    checklist = evaluator.create_checklist(task_type)
    
    for step_data in trajectory:
        evaluator.evaluate_step(step_data, checklist)
    
    return evaluator.get_evaluation_summary()
