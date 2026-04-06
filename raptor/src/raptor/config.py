"""Configuration dataclasses for RAPTOR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrchestrationAction(Enum):
    """Actions the RAPTOR orchestrator can select."""

    RESPOND = "respond"
    REROLL = "reroll"
    VERIFY = "verify"
    ESCALATE = "escalate"
    RETRIEVE = "retrieve"
    STOP = "stop"


@dataclass
class DisagreementConfig:
    """Configuration for DiscoUQ-style disagreement monitoring."""

    mode: str = "embed"  # "llm" | "embed" | "learn"
    n_agents: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_linguistic_features: bool = True
    use_embedding_features: bool = True
    # Learned model weights (for DiscoUQ-Learn mode)
    feature_weights: Optional[dict[str, float]] = None


@dataclass
class EntropyConfig:
    """Configuration for entropy trajectory monitoring."""

    monotonicity_threshold: float = 1.0  # 1.0 = strict (every step must decrease)
    slope_window: int = 3  # window for computing entropy slope
    token_level: bool = True  # compute entropy per token (vs per step)
    # Calibration params for log-prob → entropy
    vocab_size: int = 128_000


@dataclass
class UtilityConfig:
    """Configuration for utility-guided action selection."""

    # Feature function weights (learned or hand-tuned)
    weights: dict[str, float] = field(default_factory=lambda: {
        "gain": 0.30,
        "confidence": 0.25,
        "cost_penalty": -0.15,
        "redundancy_penalty": -0.10,
        "severity": 0.20,
    })
    # Action costs (token cost per action)
    action_costs: dict[OrchestrationAction, float] = field(default_factory=lambda: {
        OrchestrationAction.RESPOND: 0.0,
        OrchestrationAction.REROLL: 0.5,
        OrchestrationAction.VERIFY: 0.3,
        OrchestrationAction.ESCALATE: 1.0,
        OrchestrationAction.RETRIEVE: 0.4,
        OrchestrationAction.STOP: 0.0,
    })
    # Decision threshold for switching actions
    switch_threshold: float = 0.05
    # Hysteresis: require sustained signal for this many steps before acting
    hysteresis_steps: int = 1


@dataclass
class Config:
    """Top-level RAPTOR configuration."""

    disagreement: DisagreementConfig = field(default_factory=DisagreementConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)

    # Orchestration-level
    max_rerolls: int = 3
    max_steps: int = 20
    log_signal_history: bool = True
    log_dir: str = "raptor_logs"

    # Model configuration
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    top_p: float = 0.9
