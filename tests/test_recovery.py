"""Tests for recovery module."""

import pytest
from src.recovery import (
    RecoveryManager, 
    RecoveryStrategy, 
    RecoveryAction,
)


class TestRecoveryAction:
    """Tests for RecoveryAction."""
    
    def test_action_creation(self):
        """Test creating a recovery action."""
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            reason="Try again"
        )
        assert action.strategy == RecoveryStrategy.RETRY
        assert action.reason == "Try again"


class TestRecoveryManager:
    """Tests for RecoveryManager."""
    
    def test_initialization(self):
        """Test recovery manager initialization."""
        manager = RecoveryManager(max_retries=3)
        assert manager.max_retries == 3
        assert len(manager.attempt_history) == 0
    
    def test_assess_failure_retry(self):
        """Test failure assessment for retry."""
        manager = RecoveryManager(max_retries=3)
        context = {"current_step": 5, "retry_count": 0, "uncertainty": 0.3}
        
        action = manager.assess_failure(Exception("error"), context)
        assert action.strategy == RecoveryStrategy.RETRY
    
    def test_assess_failure_human(self):
        """Test failure assessment for human in loop."""
        manager = RecoveryManager(max_retries=3)
        context = {"current_step": 5, "retry_count": 3, "uncertainty": 0.9}
        
        action = manager.assess_failure(Exception("error"), context)
        assert action.strategy == RecoveryStrategy.HUMAN_IN_LOOP
    
    def test_assess_failure_backtrack(self):
        """Test failure assessment for backtrack."""
        manager = RecoveryManager(max_retries=3)
        context = {"current_step": 5, "retry_count": 3, "uncertainty": 0.5}
        
        action = manager.assess_failure(Exception("error"), context)
        assert action.strategy == RecoveryStrategy.BACKTRACK
    
    def test_execute_recovery(self):
        """Test executing recovery."""
        manager = RecoveryManager()
        action = RecoveryAction(strategy=RecoveryStrategy.RETRY, reason="test")
        context = {"retry_count": 0}
        
        new_context = manager.execute_recovery(action, context)
        assert new_context["retry_count"] == 1
    
    def test_record_attempt(self):
        """Test recording recovery attempts."""
        manager = RecoveryManager()
        action = RecoveryAction(strategy=RecoveryStrategy.RETRY, reason="test")
        
        manager.record_attempt(action, success=True)
        manager.record_attempt(action, success=False)
        
        assert len(manager.attempt_history) == 2
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        manager = RecoveryManager()
        action = RecoveryAction(strategy=RecoveryStrategy.RETRY, reason="test")
        
        manager.record_attempt(action, success=True)
        manager.record_attempt(action, success=True)
        manager.record_attempt(action, success=False)
        
        rate = manager.get_success_rate()
        assert rate == pytest.approx(2/3)
    
    def test_reset(self):
        """Test resetting manager."""
        manager = RecoveryManager()
        action = RecoveryAction(strategy=RecoveryStrategy.RETRY, reason="test")
        manager.record_attempt(action, True)
        manager.reset()
        assert len(manager.attempt_history) == 0
