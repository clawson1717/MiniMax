import hashlib
import json
import time
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class InteractionNode(BaseModel):
    id: str
    agent_id: str
    input_payload: Any
    output_payload: Any
    causal_parents: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def payload_hash(self) -> str:
        """
        Returns a SHA-256 hash of the input and output payloads.
        Payloads are serialized to JSON for stable hashing.
        """
        combined = {
            "input": self.input_payload,
            "output": self.output_payload
        }
        # sort_keys=True ensures consistent hashing regardless of dict order
        payload_str = json.dumps(combined, sort_keys=True, default=str)
        return hashlib.sha256(payload_str.encode()).hexdigest()

    def is_root(self) -> bool:
        """
        Returns True if the node has no causal parents.
        """
        return len(self.causal_parents) == 0
