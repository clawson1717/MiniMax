from typing import Any, List, Optional
import math
from .uncertainty import UncertaintyEstimator

class ComputeAllocator:
    """
    Dynamically allocates compute budget based on uncertainty scores.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the allocator with an uncertainty threshold.
        
        Args:
            threshold: A value between 0.0 and 1.0. Uncertainty above this 
                       triggers scaling.
        """
        self.threshold = threshold

    def should_scale(self, uncertainty_score: float) -> bool:
        """
        Determines if scaling is required based on the uncertainty score.
        
        Args:
            uncertainty_score: A normalized uncertainty score (0.0 to 1.0).
            
        Returns:
            True if the score exceeds the threshold, False otherwise.
        """
        return uncertainty_score > self.threshold

    def allocate(self, base_compute: int, uncertainty_score: float) -> int:
        """
        Returns a scaled compute budget based on uncertainty.
        
        Scaling logic:
        - If uncertainty <= threshold, return base_compute.
        - If uncertainty > threshold, scale base_compute by a multiplier 
          proportional to the excess uncertainty.
          Multiplier = 1 + (uncertainty_score - threshold) / (1 - threshold)
        
        Args:
            base_compute: The initial compute budget (e.g., reasoning steps, votes).
            uncertainty_score: The current uncertainty score (0.0 to 1.0).
            
        Returns:
            The scaled compute budget (integer).
        """
        if not self.should_scale(uncertainty_score):
            return base_compute
        
        # Calculate how far we are above the threshold
        # If threshold=0.5 and uncertainty=0.75, we are 50% through the scaling range (0.5 to 1.0)
        excess_ratio = (uncertainty_score - self.threshold) / (1.0 - self.threshold)
        
        # We'll double the compute budget at max uncertainty (1.0)
        # Multiplier ranges from 1.0 at threshold to 2.0 at uncertainty=1.0
        multiplier = 1.0 + excess_ratio
        
        return math.ceil(base_compute * multiplier)

    def allocate_from_votes(self, base_compute: int, votes: List[Any]) -> int:
        """
        Helper to calculate uncertainty from votes and then allocate.
        """
        uncertainty = UncertaintyEstimator.from_votes(votes)
        return self.allocate(base_compute, uncertainty)
