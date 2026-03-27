"""Failure Event Record module for FASHA."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class FailureEvent:
    """Represents a failure event in the self-healing agent system."""

    timestamp: datetime
    failure_type: str
    context: str
    diagnosis: str
    recovery_action: str
    outcome: str
    session_id: str

    def to_dict(self) -> dict:
        """Convert the FailureEvent to a dictionary.

        Returns:
            dict: Dictionary representation with ISO format timestamp.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type,
            "context": self.context,
            "diagnosis": self.diagnosis,
            "recovery_action": self.recovery_action,
            "outcome": self.outcome,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FailureEvent":
        """Create a FailureEvent from a dictionary.

        Args:
            data: Dictionary with failure event data.

        Returns:
            FailureEvent: Instance created from the dictionary.
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            failure_type=data["failure_type"],
            context=data["context"],
            diagnosis=data["diagnosis"],
            recovery_action=data["recovery_action"],
            outcome=data["outcome"],
            session_id=data["session_id"],
        )

    def to_json(self) -> str:
        """Convert the FailureEvent to a JSON string.

        Returns:
            str: JSON representation of the failure event.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "FailureEvent":
        """Create a FailureEvent from a JSON string.

        Args:
            json_str: JSON string representation of a failure event.

        Returns:
            FailureEvent: Instance created from the JSON string.
        """
        return cls.from_dict(json.loads(json_str))
