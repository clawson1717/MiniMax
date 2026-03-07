from enum import Enum, auto
from typing import Dict, Any, List, Optional
import numpy as np

class TrainingStage(Enum):
    FULL_COT = auto()
    HYBRID_DRAFT = auto()
    PURE_DRAFT = auto()

class CurriculumManager:
    """
    Manages the step-wise drafting curriculum for RL training.
    
    Stages:
    - FULL_COT: Model generates full Chain-of-Thought reasoning.
    - HYBRID_DRAFT: Model generates a mix (alternating or partial).
    - PURE_DRAFT: Model generates pure drafting (no/minimal CoT).
    """
    
    def __init__(
        self, 
        mastery_threshold: float = 0.8, 
        window_size: int = 100,
        min_samples_per_stage: int = 500
    ):
        self.mastery_threshold = mastery_threshold
        self.window_size = window_size
        self.min_samples_per_stage = min_samples_per_stage
        
        self.current_stage = TrainingStage.FULL_COT
        self.history: List[float] = []
        self.samples_in_current_stage = 0
        
        # Stage sequence
        self.stage_order = [
            TrainingStage.FULL_COT,
            TrainingStage.HYBRID_DRAFT,
            TrainingStage.PURE_DRAFT
        ]

    def get_current_stage(self) -> TrainingStage:
        """Returns the current training stage."""
        return self.current_stage

    def update_mastery(self, accuracy: float):
        """Updates the internal performance history and counts samples."""
        self.history.append(accuracy)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        self.samples_in_current_stage += 1
        
        if self.should_advance():
            self._advance_stage()

    def should_advance(self) -> bool:
        """Determines if the model has mastered the current stage."""
        if self.samples_in_current_stage < self.min_samples_per_stage:
            return False
            
        if len(self.history) < self.window_size:
            return False
            
        current_performance = np.mean(self.history)
        return current_performance >= self.mastery_threshold

    def _advance_stage(self):
        """Advances the training to the next stage if available."""
        current_idx = self.stage_order.index(self.current_stage)
        if current_idx < len(self.stage_order) - 1:
            self.current_stage = self.stage_order[current_idx + 1]
            self.samples_in_current_stage = 0
            self.history = [] # Reset history for the new stage

    def modify_prompt(self, prompt: str) -> str:
        """
        Modifies the prompt based on the current stage to encourage 
        appropriate output behavior.
        """
        if self.current_stage == TrainingStage.FULL_COT:
            return prompt + "\nReason step-by-step before providing the final answer."
        elif self.current_stage == TrainingStage.HYBRID_DRAFT:
            return prompt + "\nCombine structured drafting with brief reasoning steps."
        elif self.current_stage == TrainingStage.PURE_DRAFT:
            return prompt + "\nDirectly draft the solution efficiently."
        return prompt

    def get_stage_config(self) -> Dict[str, Any]:
        """Returns configuration parameters specific to the current stage."""
        configs = {
            TrainingStage.FULL_COT: {
                "temperature": 0.7,
                "max_new_tokens": 512,
                "draft_ratio": 0.0
            },
            TrainingStage.HYBRID_DRAFT: {
                "temperature": 0.5,
                "max_new_tokens": 384,
                "draft_ratio": 0.5
            },
            TrainingStage.PURE_DRAFT: {
                "temperature": 0.3,
                "max_new_tokens": 256,
                "draft_ratio": 1.0
            }
        }
        return configs.get(self.current_stage, {})
