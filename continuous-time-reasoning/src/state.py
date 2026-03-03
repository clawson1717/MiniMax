from dataclasses import dataclass
from typing import Union
import torch

@dataclass
class ReasoningState:
    """
    Represents the reasoning state at a specific point in continuous time.
    """
    hidden_state: torch.Tensor
    uncertainty: Union[float, torch.Tensor]
    confidence: Union[float, torch.Tensor]
    timestamp: float

    def to_dict(self):
        return {
            "hidden_state": self.hidden_state,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

    def clone(self):
        return ReasoningState(
            hidden_state=self.hidden_state.clone(),
            uncertainty=self.uncertainty.clone() if isinstance(self.uncertainty, torch.Tensor) else self.uncertainty,
            confidence=self.confidence.clone() if isinstance(self.confidence, torch.Tensor) else self.confidence,
            timestamp=self.timestamp
        )
