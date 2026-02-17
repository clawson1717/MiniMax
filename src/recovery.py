"""Recovery mechanisms for agent failures."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"  # Simple retry
    BACKTRACK = "backtrack"  # Go back to previous state
    BRANCH = "branch"  # Try alternative action
    HUMAN_IN_LOOP = "human_in_loop"  # Ask for human help
    RESET = "reset"  # Start over


@dataclass
class RecoveryAction:
    """A recovery action to take."""
    strategy: RecoveryStrategy
    target_step: Optional[int] = None
    alternative_action: Optional[str] = None
    reason: str = ""


class RecoveryManager:
    """
    Manages recovery from agent failures.
    
    Implements recovery strategies from CATTS and CM2.
    """
    
    def __init__(self, max_retries: int = 3):
        """Initialize recovery manager.
        
        Args:
            max_retries: Maximum number of retry attempts.
        """
        self.max_retries = max_retries
        self.attempt_history: List[Dict[str, Any]] = []
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
        
    def _register_default_handlers(self) -> None:
        """Register default recovery handlers."""
        self.recovery_handlers[RecoveryStrategy.RETRY] = self._handle_retry
        self.recovery_handlers[RecoveryStrategy.BACKTRACK] = self._handle_backtrack
        self.recovery_handlers[RecoveryStrategy.BRANCH] = self._handle_branch
        self.recovery_handlers[RecoveryStrategy.HUMAN_IN_LOOP] = self._handle_human_in_loop
        self.recovery_handlers[RecoveryStrategy.RESET] = self._handle_reset
    
    def assess_failure(self, error: Exception, context: Dict[str, Any]) -> RecoveryAction:
        """Assess a failure and determine recovery action.
        
        Args:
            error: The exception that occurred.
            context: Context information (current step, trajectory, etc.).
            
        Returns:
            RecoveryAction to take.
        """
        current_step = context.get("current_step", 0)
        retry_count = context.get("retry_count", 0)
        uncertainty = context.get("uncertainty", 0.0)
        
        # Simple decision logic (placeholder)
        if retry_count < self.max_retries:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                reason=f"Retry attempt {retry_count + 1}/{self.max_retries}"
            )
        elif uncertainty > 0.8:
            return RecoveryAction(
                strategy=RecoveryStrategy.HUMAN_IN_LOOP,
                reason="High uncertainty - need human guidance"
            )
        elif current_step > 0:
            return RecoveryAction(
                strategy=RecoveryStrategy.BACKTRACK,
                target_step=current_step - 1,
                reason="Max retries exceeded - backtracking"
            )
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.RESET,
                reason="Unable to recover - resetting"
            )
    
    def execute_recovery(self, action: RecoveryAction, 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recovery action.
        
        Args:
            action: RecoveryAction to execute.
            context: Current context.
            
        Returns:
            Updated context after recovery.
        """
        handler = self.recovery_handlers.get(action.strategy)
        if handler:
            return handler(action, context)
        return context
    
    def _handle_retry(self, action: RecoveryAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retry strategy."""
        context["retry_count"] = context.get("retry_count", 0) + 1
        return context
    
    def _handle_backtrack(self, action: RecoveryAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backtrack strategy."""
        if action.target_step is not None:
            context["current_step"] = action.target_step
            context["retry_count"] = 0
        return context
    
    def _handle_branch(self, action: RecoveryAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle branch strategy (try alternative)."""
        context["alternative_action"] = action.alternative_action
        context["retry_count"] = 0
        return context
    
    def _handle_human_in_loop(self, action: RecoveryAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle human-in-the-loop strategy."""
        context["awaiting_human_input"] = True
        return context
    
    def _handle_reset(self, action: RecoveryAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset strategy."""
        context["reset_requested"] = True
        return context
    
    def record_attempt(self, action: RecoveryAction, success: bool) -> None:
        """Record a recovery attempt.
        
        Args:
            action: The recovery action taken.
            success: Whether the recovery was successful.
        """
        self.attempt_history.append({
            "strategy": action.strategy.value,
            "success": success,
            "reason": action.reason,
        })
    
    def get_success_rate(self, strategy: Optional[RecoveryStrategy] = None) -> float:
        """Calculate success rate of recovery attempts.
        
        Args:
            strategy: Specific strategy to calculate for. If None, calculate overall.
            
        Returns:
            Success rate (0-1).
        """
        if not self.attempt_history:
            return 0.0
            
        if strategy:
            attempts = [a for a in self.attempt_history if a["strategy"] == strategy.value]
        else:
            attempts = self.attempt_history
            
        if not attempts:
            return 0.0
            
        successes = sum(1 for a in attempts if a["success"])
        return successes / len(attempts)
    
    def reset(self) -> None:
        """Clear attempt history."""
        self.attempt_history = []