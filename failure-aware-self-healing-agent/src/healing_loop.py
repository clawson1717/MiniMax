"""
Self-Healing Loop Implementation

The SelfHealingLoop orchestrates the entire failure detection, diagnosis, recovery,
and learning process. It uses a state machine to manage the workflow and integrates
with the DiagnosticEngine, RecoveryStrategy, and FailureEvent components.

State Machine:
- HEALTHY: No failures detected, normal operation
- FAILING: A failure has been detected, entering recovery workflow
- DIAGNOSING: Determining the root cause of the failure
- RECOVERING: Executing the selected recovery strategy
- VERIFYING: Checking if the recovery was successful
- LEARNING: Recording the failure event for future learning
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Dict, Optional, Type, Union

from dataclasses import dataclass
from datetime import datetime

from .failure_event import FailureEvent, FailureType
from .diagnostic import DiagnosticEngine, RootCause
from .recovery import RecoveryStrategy, RecoveryResult, RecoveryStrategyLibrary


class HealingState(Enum):
    """States in the self-healing workflow."""
    HEALTHY = "healthy"
    FAILING = "failing"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    LEARNING = "learning"
    ERROR = "error"


@dataclass
class HealingConfig:
    """Configuration for the self-healing loop."""
    # Thresholds for state transitions (0.0-1.0)
    failure_detection_threshold: float = 0.7
    diagnosis_timeout_seconds: int = 30
    recovery_timeout_seconds: int = 60
    verification_timeout_seconds: int = 30
    learning_enabled: bool = True
    
    # Confidence decay factors
    confidence_decay_on_failure: float = 0.5
    confidence_recovery_on_success: float = 0.8


class SelfHealingLoop:
    """
    Self-healing loop that detects, diagnoses, and recovers from failures.
    
    The loop follows a state machine pattern with configurable thresholds
    and timeout handling. It integrates with DiagnosticEngine and
    RecoveryStrategyLibrary to provide intelligent recovery.
    """
    
    def __init__(
        self,
        diagnostic_engine: Optional[DiagnosticEngine] = None,
        recovery_library: Optional[RecoveryStrategyLibrary] = None,
        config: Optional[HealingConfig] = None,
    ) -> None:
        """
        Initialize the self-healing loop.
        
        Args:
            diagnostic_engine: Engine for failure diagnosis (default: creates new)
            recovery_library: Library of recovery strategies (default: creates new)
            config: Configuration settings (default: default config)
        """
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()
        self.recovery_library = recovery_library or RecoveryStrategyLibrary()
        self.config = config or HealingConfig()
        
        # Current state
        self.state: HealingState = HealingState.HEALTHY
        self.previous_state: Optional[HealingState] = None
        
        # Failure tracking
        self.current_failure: Optional[FailureEvent] = None
        self.failure_history: list[FailureEvent] = []
        
        # Timing
        self.state_start_time: Optional[float] = None
        self.total_recovery_time: float = 0.0
        
        # Callbacks for state transitions (for external monitoring)
        self.on_state_change: Dict[HealingState, Callable[[], None]] = {}
        
        # Confidence tracking
        self.current_confidence: float = 1.0
        
        # Error handling
        self.error_message: Optional[str] = None
        
    def _transition_to(self, new_state: HealingState) -> None:
        """Transition to a new state and notify callbacks."""
        self.previous_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        
        if new_state in self.on_state_change:
            try:
                self.on_state_change[new_state]()
            except Exception as e:
                self._set_error(f"State change callback failed: {e}")
    
    def _set_error(self, message: str) -> None:
        """Set error state and record error message."""
        self.state = HealingState.ERROR
        self.error_message = message
    
    def _has_timed_out(self, timeout_seconds: int) -> bool:
        """Check if the current state has timed out."""
        if self.state_start_time is None:
            return False
        return (time.time() - self.state_start_time) > timeout_seconds
    
    def _update_confidence(self, factor: float) -> None:
        """Update confidence score with decay/growth factor."""
        self.current_confidence = max(0.0, min(1.0, self.current_confidence * factor))
    
    def detect_failure(self, failure_type: FailureType, context: Dict) -> bool:
        """
        Detect a failure and transition to FAILING state if threshold met.
        
        Args:
            failure_type: Type of failure detected
            context: Context information about the failure
            
        Returns:
            bool: True if failure was detected and state transitioned, False otherwise
        """
        # Create failure event
        failure_event = FailureEvent(
            timestamp=datetime.now(),
            failure_type=failure_type,
            context=context,
            diagnosis=None,
            recovery_action=None,
            outcome=None,
        )
        
        # Update confidence based on failure severity
        severity = failure_event.get_severity()
        self._update_confidence(1.0 - (severity * self.config.failure_detection_threshold))
        
        # Check if we should transition to failing state
        if severity >= self.config.failure_detection_threshold:
            self.current_failure = failure_event
            self._transition_to(HealingState.FAILING)
            return True
        return False
    
    def diagnose_failure(self) -> Optional[RootCause]:
        """
        Diagnose the root cause of the current failure.
        
        Returns:
            RootCause: The diagnosed root cause, or None if diagnosis failed/timed out
        """
        if self.state != HealingState.FAILING:
            self._set_error("Cannot diagnose: not in FAILING state")
            return None
        
        self._transition_to(HealingState.DIAGNOSING)
        
        try:
            # Run diagnosis with timeout
            start_time = time.time()
            diagnosis = self.diagnostic_engine.diagnose(self.current_failure)
            diagnosis_time = time.time() - start_time
            
            if diagnosis_time > self.config.diagnosis_timeout_seconds:
                self._set_error("Diagnosis timed out")
                return None
            
            # Update failure event with diagnosis
            self.current_failure.diagnosis = diagnosis
            return diagnosis
            
        except Exception as e:
            self._set_error(f"Diagnosis failed: {e}")
            return None
    
    def select_recovery_strategy(self) -> Optional[Type[RecoveryStrategy]]:
        """
        Select the most appropriate recovery strategy based on diagnosis.
        
        Returns:
            Type[RecoveryStrategy]: Recovery strategy class, or None if selection failed
        """
        if self.state != HealingState.DIAGNOSING:
            self._set_error("Cannot select strategy: not in DIAGNOSING state")
            return None
        
        self._transition_to(HealingState.RECOVERING)
        
        if self.current_failure and self.current_failure.diagnosis:
            try:
                strategy_class = self.recovery_library.select_strategy(
                    self.current_failure.diagnosis
                )
                return strategy_class
            except Exception as e:
                self._set_error(f"Strategy selection failed: {e}")
                return None
        else:
            self._set_error("Cannot select strategy: no diagnosis available")
            return None
    
    def execute_recovery(self) -> Optional[RecoveryResult]:
        """
        Execute the selected recovery strategy.
        
        Returns:
            RecoveryResult: Result of the recovery attempt, or None if failed
        """
        strategy_class = self.select_recovery_strategy()
        if not strategy_class:
            return None
        
        if self.state != HealingState.RECOVERING:
            self._set_error("Cannot execute recovery: not in RECOVERING state")
            return None
        
        try:
            strategy = strategy_class()
            result = strategy.execute()
            
            # Update failure event with recovery action and outcome
            if self.current_failure:
                self.current_failure.recovery_action = strategy.name
                self.current_failure.outcome = result.success
            
            # Update confidence based on recovery execution
            if result.success:
                self._update_confidence(self.config.confidence_recovery_on_success)
            else:
                self._update_confidence(self.config.confidence_decay_on_failure)
            
            return result
            
        except Exception as e:
            self._set_error(f"Recovery execution failed: {e}")
            return None
    
    def verify_recovery(self) -> bool:
        """
        Verify that the recovery was successful.
        
        Returns:
            bool: True if recovery verified, False otherwise
        """
        if self.state != HealingState.RECOVERING:
            self._set_error("Cannot verify recovery: not in RECOVERING state")
            return False
        
        self._transition_to(HealingState.VERIFYING)
        
        try:
            # In a real implementation, this would check actual system state
            # For now, we'll use the recovery result's success flag
            if self.current_failure and self.current_failure.outcome is not None:
                verification_result = self.current_failure.outcome
                self._record_failure_event()
                return verification_result
            else:
                self._set_error("Cannot verify: no recovery outcome available")
                return False
        except Exception as e:
            self._set_error(f"Recovery verification failed: {e}")
            return False
    
    def learn_from_failure(self) -> bool:
        """
        Learn from the failure experience and update pattern memory.
        
        Returns:
            bool: True if learning succeeded, False otherwise
        """
        if self.state != HealingState.LEARNING:
            self._set_error("Cannot learn: not in LEARNING state")
            return False
        
        if not self.config.learning_enabled:
            return True
        
        try:
            # In a full implementation, this would update failure pattern memory
            # For now, we'll just record the event in history
            if self.current_failure:
                self.failure_history.append(self.current_failure)
            return True
        except Exception as e:
            self._set_error(f"Learning failed: {e}")
            return False
    
    def run(self) -> bool:
        """
        Run the complete self-healing workflow.
        
        Returns:
            bool: True if healing workflow completed successfully, False otherwise
        """
        # Detect failure (would be called from external monitoring)
        # For this implementation, we'll simulate a failure detection
        # In practice, this would be triggered by external failure detection
        
        # Simulate failure detection for testing
        if not self.detect_failure(FailureType.REASONING_ERROR, {"step": "logical deduction"}):
            return False
        
        # Main workflow
        try:
            # Diagnose
            diagnosis = self.diagnose_failure()
            if not diagnosis:
                return False
            
            # Select strategy
            strategy_class = self.select_recovery_strategy()
            if not strategy_class:
                return False
            
            # Execute recovery
            result = self.execute_recovery()
            if not result:
                return False
            
            # Verify recovery
            if not self.verify_recovery():
                return False
            
            # Learn from failure
            if self.config.learning_enabled:
                self._transition_to(HealingState.LEARNING)
                if not self.learn_from_failure():
                    return False
            
            # Return to healthy state
            self._transition_to(HealingState.HEALTHY)
            return True
            
        except Exception as e:
            self._set_error(f"Healing workflow failed: {e}")
            return False
    
    def step(self) -> HealingState:
        """
        Execute one step of the healing process (for manual control).
        
        Returns:
            HealingState: Current state after the step
        """
        if self.state == HealingState.HEALTHY:
            # No action needed in healthy state
            pass
        elif self.state == HealingState.FAILING:
            # Auto-diagnose when failing
            self.diagnose_failure()
        elif self.state == HealingState.DIAGNOSING:
            # Try to complete diagnosis
            self.diagnose_failure()
        elif self.state == HealingState.RECOVERING:
            # Try to execute recovery
            self.execute_recovery()
        elif self.state == HealingState.VERIFYING:
            # Try to verify recovery
            self.verify_recovery()
        elif self.state == HealingState.LEARNING:
            # Try to learn
            self.learn_from_failure()
        elif self.state == HealingState.ERROR:
            # Stay in error state until manually reset
            pass
        
        return self.state
    
    def get_state(self) -> HealingState:
        """
        Get the current healing state.
        
        Returns:
            HealingState: Current state
        """
        return self.state
    
    def abort(self) -> bool:
        """
        Abort the current healing process and return to healthy state.
        
        Returns:
            bool: True if aborted successfully, False otherwise
        """
        self._transition_to(HealingState.HEALTHY)
        self.current_failure = None
        self.error_message = None
        return True


# Example usage and testing functions
def example_usage() -> None:
    """Example usage of the SelfHealingLoop."""
    # Initialize components
    diagnostic_engine = DiagnosticEngine()
    recovery_library = RecoveryStrategyLibrary()
    
    # Create healing loop
    healer = SelfHealingLoop(
        diagnostic_engine=diagnostic_engine,
        recovery_library=recovery_library,
        config=HealingConfig(
            failure_detection_threshold=0.5,
            diagnosis_timeout_seconds=10,
            recovery_timeout_seconds=30,
            learning_enabled=True
        )
    )
    
    # Register state change callbacks
    def on_failing():
        print("Failure detected! Entering recovery workflow.")
    
    def on_recovering():
        print("Executing recovery strategy...")
    
    healer.on_state_change[HealingState.FAILING] = on_failing
    healer.on_state_change[HealingState.RECOVERING] = on_recovering
    
    # Simulate failure detection (in practice, this would come from monitoring)
    healer.detect_failure(FailureType.KNOWLEDGE_GAP, {"missing_concept": "quantum entanglement"})
    
    # Run the healing process
    success = healer.run()
    print(f"Healing {'succeeded' if success else 'failed'} with state: {healer.get_state()}")
    
    if healer.error_message:
        print(f"Error: {healer.error_message}")


if __name__ == "__main__":
    # Run example when executed directly
    example_usage()