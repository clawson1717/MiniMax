"""Tests for the FailureEvent dataclass."""

import pytest
from datetime import datetime
from src.failure_event import FailureEvent


class TestFailureEvent:
    """Tests for FailureEvent class."""

    def test_init_with_all_fields(self):
        """Init should store all provided fields correctly."""
        ts = datetime(2026, 3, 27, 10, 30, 0)
        event = FailureEvent(
            timestamp=ts,
            failure_type="API_TIMEOUT",
            context="External API call to /v1/generate",
            diagnosis="Request exceeded 30s timeout threshold",
            recovery_action="Retry with exponential backoff",
            outcome="SUCCESS",
            session_id="sess-abc123",
        )
        assert event.timestamp == ts
        assert event.failure_type == "API_TIMEOUT"
        assert event.context == "External API call to /v1/generate"
        assert event.diagnosis == "Request exceeded 30s timeout threshold"
        assert event.recovery_action == "Retry with exponential backoff"
        assert event.outcome == "SUCCESS"
        assert event.session_id == "sess-abc123"

    def test_to_dict_returns_correct_structure(self):
        """to_dict should return a dictionary with ISO format timestamp."""
        ts = datetime(2026, 3, 27, 10, 30, 0)
        event = FailureEvent(
            timestamp=ts,
            failure_type="MEMORY_ERROR",
            context="Vector store query failed",
            diagnosis="Out of memory during indexing",
            recovery_action="Clear cache and retry",
            outcome="PARTIAL",
            session_id="sess-xyz789",
        )
        result = event.to_dict()
        assert result["timestamp"] == "2026-03-27T10:30:00"
        assert result["failure_type"] == "MEMORY_ERROR"
        assert result["context"] == "Vector store query failed"
        assert result["diagnosis"] == "Out of memory during indexing"
        assert result["recovery_action"] == "Clear cache and retry"
        assert result["outcome"] == "PARTIAL"
        assert result["session_id"] == "sess-xyz789"

    def test_from_dict_reconstructs_event(self):
        """from_dict should create a FailureEvent with correct values."""
        data = {
            "timestamp": "2026-03-27T14:45:00",
            "failure_type": "CONNECTION_REFUSED",
            "context": "Database connection pool exhausted",
            "diagnosis": "Too many concurrent connections",
            "recovery_action": "Scale connection pool and retry",
            "outcome": "SUCCESS",
            "session_id": "sess-db001",
        }
        event = FailureEvent.from_dict(data)
        assert event.timestamp == datetime(2026, 3, 27, 14, 45, 0)
        assert event.failure_type == "CONNECTION_REFUSED"
        assert event.context == "Database connection pool exhausted"
        assert event.diagnosis == "Too many concurrent connections"
        assert event.recovery_action == "Scale connection pool and retry"
        assert event.outcome == "SUCCESS"
        assert event.session_id == "sess-db001"

    def test_to_json_returns_valid_json(self):
        """to_json should return a valid JSON string."""
        ts = datetime(2026, 3, 27, 10, 0, 0)
        event = FailureEvent(
            timestamp=ts,
            failure_type="RATE_LIMIT",
            context="API rate limit exceeded",
            diagnosis="Too many requests in window",
            recovery_action="Wait and retry",
            outcome="SUCCESS",
            session_id="sess-rate123",
        )
        json_str = event.to_json()
        assert '"timestamp": "2026-03-27T10:00:00"' in json_str
        assert '"failure_type": "RATE_LIMIT"' in json_str

    def test_from_json_reconstructs_event(self):
        """from_json should recreate a FailureEvent from JSON string."""
        json_str = '{"timestamp": "2026-03-27T16:20:00", "failure_type": "TIMEOUT", "context": "LLM response delayed", "diagnosis": "Model overloaded", "recovery_action": "Switch to backup model", "outcome": "SUCCESS", "session_id": "sess-llm456"}'
        event = FailureEvent.from_json(json_str)
        assert event.timestamp == datetime(2026, 3, 27, 16, 20, 0)
        assert event.failure_type == "TIMEOUT"
        assert event.session_id == "sess-llm456"

    def test_round_trip_to_dict_from_dict(self):
        """to_dict followed by from_dict should produce equivalent event."""
        ts = datetime(2026, 3, 27, 8, 0, 0)
        original = FailureEvent(
            timestamp=ts,
            failure_type="NETWORK_ERROR",
            context="WebSocket disconnected",
            diagnosis="Connection reset by peer",
            recovery_action="Reconnect with backoff",
            outcome="SUCCESS",
            session_id="sess-ws999",
        )
        reconstructed = FailureEvent.from_dict(original.to_dict())
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.failure_type == original.failure_type
        assert reconstructed.context == original.context
        assert reconstructed.diagnosis == original.diagnosis
        assert reconstructed.recovery_action == original.recovery_action
        assert reconstructed.outcome == original.outcome
        assert reconstructed.session_id == original.session_id

    def test_round_trip_to_json_from_json(self):
        """to_json followed by from_json should produce equivalent event."""
        ts = datetime(2026, 3, 27, 9, 30, 0)
        original = FailureEvent(
            timestamp=ts,
            failure_type="AUTH_FAILURE",
            context="Token expired mid-session",
            diagnosis="JWT token past expiration",
            recovery_action="Refresh token and retry",
            outcome="SUCCESS",
            session_id="sess-auth111",
        )
        json_str = original.to_json()
        reconstructed = FailureEvent.from_json(json_str)
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.failure_type == original.failure_type
        assert reconstructed.session_id == original.session_id
