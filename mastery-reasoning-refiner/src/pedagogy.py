from enum import Enum, auto
from typing import Dict, List, Optional

class BloomTier(Enum):
    KNOWLEDGE = 1
    COMPREHENSION = 2
    APPLICATION = 3
    ANALYSIS = 4
    SYNTHESIS = 5
    EVALUATION = 6

    @classmethod
    def get_ordered_tiers(cls) -> List['BloomTier']:
        return sorted(list(cls), key=lambda x: x.value)

class PedagogicalController:
    """
    Manages the training curriculum based on Bloom's Taxonomy and Mastery Learning principles.
    """
    def __init__(self, mastery_threshold: float = 0.8, alpha: float = 0.2):
        """
        Initialize the controller.
        
        Args:
            mastery_threshold: The score (0.0 to 1.0) required to consider a tier mastered.
            alpha: Smoothing factor for mastery score updates (Exponential Moving Average).
        """
        self.mastery_threshold = mastery_threshold
        self.alpha = alpha
        self.mastery_scores: Dict[BloomTier, float] = {
            tier: 0.0 for tier in BloomTier
        }
        self.current_focus_tier = BloomTier.KNOWLEDGE

    def update_mastery(self, tier: BloomTier, performance_score: float):
        """
        Update the mastery score for a specific tier using an exponential moving average.
        
        Args:
            tier: The BloomTier to update.
            performance_score: The observed performance (0.0 to 1.0).
        """
        if not (0.0 <= performance_score <= 1.0):
            raise ValueError("Performance score must be between 0.0 and 1.0")
            
        current_score = self.mastery_scores[tier]
        # EMA update: new_score = (1 - alpha) * current + alpha * observation
        # This handles noise in individual evaluation steps
        new_score = (1 - self.alpha) * current_score + self.alpha * performance_score
        self.mastery_scores[tier] = new_score
        
        self._update_current_focus()

    def update_performance(self, tier: BloomTier, performance_score: float):
        """
        Generic entry point for updating performance metrics. 
        Can be overridden by subclasses (like ZPDController) to track additional data.
        """
        self.update_mastery(tier, performance_score)

    def calculate_difficulty_target(self, world_model_capacity: float) -> float:
        """
        Returns a target difficulty level. Base controller returns a constant.
        """
        return 1.0

    def _update_current_focus(self):
        """
        Determine the next tier to focus on based on mastery levels.
        We progress to the next tier only if the current one (and all previous) 
        meet the mastery threshold.
        """
        ordered_tiers = BloomTier.get_ordered_tiers()
        
        for tier in ordered_tiers:
            if self.mastery_scores[tier] < self.mastery_threshold:
                self.current_focus_tier = tier
                return
        
        # If all tiers are mastered, stay on the highest one (or could return None)
        self.current_focus_tier = ordered_tiers[-1]

    def get_current_tier(self) -> BloomTier:
        """Returns the tier the agent should currently be training on."""
        return self.current_focus_tier

    def get_mastery_report(self) -> Dict[str, float]:
        """Returns a human-readable report of mastery scores."""
        return {tier.name: score for tier, score in self.mastery_scores.items()}

    def reset(self):
        """Reset all mastery scores to zero."""
        for tier in BloomTier:
            self.mastery_scores[tier] = 0.0
        self.current_focus_tier = BloomTier.KNOWLEDGE

class ZPDController(PedagogicalController):
    """
    Extends PedagogicalController with Zone of Proximal Development (ZPD) logic.
    Adjusts task difficulty based on World Model Capacity and agent performance.
    """
    def __init__(self, 
                 base_difficulty: float = 1.0, 
                 k: float = 0.5, 
                 mastery_threshold: float = 0.8, 
                 alpha: float = 0.2):
        """
        Initialize the ZPD controller.
        
        Args:
            base_difficulty: Starting difficulty level.
            k: Scaling factor for World Model Capacity.
            mastery_threshold: Score required for Bloom tier mastery.
            alpha: EMA smoothing factor for performance tracking.
        """
        super().__init__(mastery_threshold, alpha)
        self.base_difficulty = base_difficulty
        self.k = k
        self.recent_success_rate = 0.75  # Assume mid-range ZPD to start
        
    def calculate_difficulty_target(self, world_model_capacity: float) -> float:
        """
        Calculates the target difficulty using the formula:
        Difficulty_Target = Base_Difficulty + (K * World_Model_Capacity)
        
        The result is modulated to ensure the agent stays within the ZPD
        (success rate between 60% and 90%).
        
        Args:
            world_model_capacity: Current capacity in bits (from metrics.py).
            
        Returns:
            The adjusted difficulty level.
        """
        raw_target = self.base_difficulty + (self.k * world_model_capacity)
        
        # ZPD Constraint: 0.6 <= Success Rate <= 0.9
        # If success is too low, we scale back difficulty to return to ZPD.
        if self.recent_success_rate < 0.6:
            # Scale down linearly based on how far below 60% we are
            adjustment = self.recent_success_rate / 0.6
            return raw_target * adjustment
            
        # If success is too high, we can safely push the target or even 
        # accelerate slightly to find the upper bound of ZPD.
        if self.recent_success_rate > 0.9:
            # Push slightly harder to find the limit
            adjustment = self.recent_success_rate / 0.9
            return raw_target * adjustment
            
        return raw_target

    def update_performance(self, tier: BloomTier, performance_score: float):
        """
        Updates mastery and ZPD performance tracking.
        
        Args:
            tier: The BloomTier the agent just attempted.
            performance_score: The score (0.0 to 1.0) achieved.
        """
        # Update Bloom mastery via parent
        self.update_mastery(tier, performance_score)
        
        # Update ZPD tracking
        self.recent_success_rate = (
            (1 - self.alpha) * self.recent_success_rate + 
            self.alpha * performance_score
        )
