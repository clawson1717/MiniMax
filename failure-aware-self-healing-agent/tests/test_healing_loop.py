"""
Test suite for SelfHealingLoop implementation.
"""

import time
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import pytest

from src.failure_event import FailureEvent, FailureType
from src.diagnostic import DiagnosticEngine, RootCause
from src.recovery import RecoveryStrategy, RecoveryResult, RecoveryStrategyLibrary
from src.healing_loop import SelfHealingLoop, HealingConfig, HealingState


# Mock classes for testing
class MockDiagnosticEngine(DiagnosticEngine):
    """Mock diagnostic engine for testing."""
    def diagnose(self, failure_event: FailureEvent) -> RootCause:
        return RootCause.KNOWLEDGE_GAP


class MockRecoveryStrategy(RecoveryStrategy):
    """Mock recovery strategy."""
    name = "mock_strategy"
    
    def execute(self) -> RecoveryResult:
        return RecoveryResult(success=True, outcome_message="Mock recovery executed")


class TestSelfHealingLoop:
    """Test cases for SelfHealingLoop."""
    
    def test_initial_state(self):
        """Test initial state is HEALTHY."""
        healer = SelfHealingLoop()
        assert healer.get_state() == HealingState.HEALTHY
    
    def test_detect_failure_transition(self):
        """Test that detecting a failure transitions to FAILING state."""
        healer = SelfHealingLoop()
        success = healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        assert success
        assert healer.get_state() == HealingState.FAILING
        assert healer.current_failure is not None
    
    def test_diagnose_failure(self):
        """Test diagnosing a failure."""
        diagnostic_engine = MockDiagnosticEngine()
        healer = SelfHealingLoop(diagnostic_engine=diagnostic_engine)
        healer.detect_failure(FailureType.KNOWLEDGE_GAP, {"concept": "test"})
        diagnosis = healer.diagnose_failure()
        assert diagnosis == RootCause.KNOWLEDGE_GAP
        assert healer.get_state() == HealingState.DIAGNOSING
    
    def test_select_recovery_strategy(self):
        """Test selecting a recovery strategy."""
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer = SelfHealingLoop(recovery_library=recovery_library)
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        healer.diagnose_failure()
        strategy_class = healer.select_recovery_strategy()
        assert strategy_class == MockRecoveryStrategy
    
    def test_execute_recovery(self):
        """Test executing a recovery strategy."""
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer = SelfHealingLoop(recovery_library=recovery_library)
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        healer.diagnose_failure()
        healer.select_recovery_strategy()
        result = healer.execute_recovery()
        assert result is not None
        assert result.success is True
    
    def test_verify_recovery(self):
        """Test verifying recovery."""
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer = SelfHealingLoop(recovery_library=recovery_library)
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        healer.diagnose_failure()
        healer.select_recovery_strategy()
        healer.execute_recovery()
        success = healer.verify_recovery()
        assert success
        assert healer.get_state() == HealingState.VERIFYING
    
    def test_complete_workflow(self):
        """Test the complete healing workflow."""
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer = SelfHealingLoop(
            recovery_library=recovery_library,
            config=HealingConfig(learning_enabled=True)
        )
        
        success = healer.run()
        assert success
        assert healer.get_state() == HealingState.HEALTHY
        assert len(healer.failure_history) == 1
    
    def test_state_transitions(self):
        """Test all state transitions."""
        healer = SelfHealingLoop()
        
        # Start in HEALTHY
        assert healer.get_state() == HealingState.HEALTHY
        
        # Detect failure -> FAILING
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        assert healer.get_state() == HealingState.FAILING
        
        # Diagnose -> DIAGNOSING
        healer.diagnose_failure()
        assert healer.get_state() == HealingState.DIAGNOSING
        
        # Select strategy -> RECOVERING
        healer.select_recovery_strategy()
        assert healer.get_state() == HealingState.RECOVERING
        
        # Execute recovery -> VERIFYING
        healer.execute_recovery()
        assert healer.get_state() == HealingState.VERIFYING
        
        # Verify -> LEARNING (then HEALTHY)
        healer.verify_recovery()
        assert healer.get_state() == HealingState.LEARNING
        
        # Complete -> HEALTHY
        healer.learn_from_failure()
        assert healer.get_state() == HealingState.LEARNING  # Still in learning
        healer._transition_to(HealingState.HEALTHY)
        assert healer.get_state() == HealingState.HEALTHY
    
    def test_timeouts(self):
        """Test timeout handling."""
        diagnostic_engine = MockDiagnosticEngine()
        healer = SelfHealingLoop(
            diagnostic_engine=diagnostic_engine,
            config=HealingConfig(diagnosis_timeout_seconds=0.01)  # Very short timeout
        )
        
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        time.sleep(0.02)  # Wait for timeout
        diagnosis = healer.diagnose_failure()
        assert diagnosis is None
        assert healer.get_state() == HealingState.ERROR
    
    def test_confidence_tracking(self):
        """Test confidence tracking updates."""
        healer = SelfHealingLoop()
        initial_confidence = healer.current_confidence
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        # Confidence should decrease on failure
        assert healer.current_confidence < initial_confidence
        
        # Simulate recovery success
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer.recovery_library = recovery_library
        healer.diagnose_failure()
        healer.select_recovery_strategy()
        healer.execute_recovery()
        # Confidence should increase on success
        assert healer.current_confidence > 0.5
    
    def test_error_handling(self):
        """Test error handling."""
        healer = SelfHealingLoop()
        
        # Invalid state transitions
        healer._transition_to(HealingState.ERROR)
        assert healer.get_state() == HealingState.ERROR
        
        # Invalid operations should set error state
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        healer.diagnose_failure()
        healer.select_recovery_strategy()
        healer.execute_recovery()
        healer.verify_recovery()
        healer.learn_from_failure()
        healer._transition_to(HealingState.HEALTHY)
        
        # Try to diagnose again (should error)
        healer.diagnose_failure()
        assert healer.get_state() == HealingState.ERROR
    
    def test_abort(self):
        """Test abort functionality."""
        recovery_library = RecoveryStrategyLibrary()
        recovery_library.register_strategy(MockRecoveryStrategy)
        healer = SelfHealingLoop(recovery_library=recovery_library)
        
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        healer.diagnose_failure()
        healer.select_recovery_strategy()
        healer.abort()
        
        assert healer.get_state() == HealingState.HEALTHY
        assert healer.current_failure is None
        assert healer.error_message is None
    
    def test_callbacks(self):
        """Test state change callbacks."""
        callback_called = False
        
        def on_failing():
            nonlocal callback_called
            callback_called = True
            print("Callback called!")
        
        healer = SelfHealingLoop()
        healer.on_state_change[HealingState.FAILING] = on_failing
        
        healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
        assert callback_called is True

# Additional test for edge cases
def test_healing_loop_with_no_diagnostic_engine():
    """Test loop works with default diagnostic engine."""
    healer = SelfHealingLoop()
    healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
    diagnosis = healer.diagnose_failure()
    # Default diagnostic engine should return a diagnosis
    assert diagnosis is not None

def test_healing_loop_with_no_recovery_library():
    """Test loop works with default recovery library."""
    healer = SelfHealingLoop()
    healer.detect_failure(FailureType.REASONING_ERROR, {"step": "test"})
    healer.diagnose_failure()
    strategy = healer.select_recovery_strategy()
    # Default recovery library should return a strategy
    assert strategy is not None