"""
Uncertainty Estimator for test-time compute scaling.

Implements entropy-based uncertainty estimation to determine when to scale
compute resources based on disagreement among agent responses.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class UncertaintyResult:
    """
    Result of an uncertainty estimation.
    
    Attributes:
        uncertainty_score: Normalized uncertainty score (0.0-1.0, higher = more uncertain)
        entropy: Raw entropy in bits
        unique_responses: Number of unique responses in the distribution
        vote_distribution: Dictionary mapping response -> proportion (0.0-1.0)
        method: Method used for estimation (entropy, vote_variance, etc.)
    """
    uncertainty_score: float
    entropy: float
    unique_responses: int
    vote_distribution: Dict[str, float]
    method: str
    
    def __post_init__(self):
        """Validate the result after initialization."""
        if not 0.0 <= self.uncertainty_score <= 1.0:
            raise ValueError("uncertainty_score must be between 0.0 and 1.0")
        if self.entropy < 0:
            raise ValueError("entropy must be non-negative")
        if self.unique_responses < 0:
            raise ValueError("unique_responses must be non-negative")
        
        # Validate vote distribution sums approximately to 1.0
        if self.vote_distribution:
            total = sum(self.vote_distribution.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"vote_distribution must sum to 1.0, got {total}")


class UncertaintyEstimator:
    """
    Estimates uncertainty from response distributions for test-time compute scaling.
    
    Uses entropy-based methods to measure disagreement among agent responses.
    High uncertainty indicates the agents disagree, suggesting the need to
    scale up compute (invoke more agents, use stronger models, etc.).
    
    Example:
        >>> estimator = UncertaintyEstimator(scale_threshold=0.5)
        >>> responses = ["answer A", "answer B", "answer A"]
        >>> result = estimator.estimate_uncertainty(responses)
        >>> if estimator.should_scale(result.uncertainty_score):
        ...     print("High uncertainty - scale up compute")
    """
    
    def __init__(
        self,
        scale_threshold: float = 0.5,
        min_samples: int = 1,
        normalize: bool = True
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            scale_threshold: Threshold above which compute should be scaled (0.0-1.0)
            min_samples: Minimum number of samples needed for estimation
            normalize: Whether to normalize entropy to 0-1 scale
        """
        if not 0.0 <= scale_threshold <= 1.0:
            raise ValueError("scale_threshold must be between 0.0 and 1.0")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        
        self.scale_threshold = scale_threshold
        self.min_samples = min_samples
        self.normalize = normalize
    
    def estimate_uncertainty(
        self,
        responses: List[str],
        method: str = "entropy"
    ) -> UncertaintyResult:
        """
        Calculate uncertainty from a distribution of responses.
        
        Args:
            responses: List of response strings from agents
            method: Estimation method (entropy, vote_variance)
            
        Returns:
            UncertaintyResult with uncertainty metrics
            
        Raises:
            ValueError: If responses is empty or contains invalid data
        """
        if not responses:
            raise ValueError("responses cannot be empty")
        
        # Filter out None and empty responses
        valid_responses = [str(r) for r in responses if r is not None and str(r).strip()]
        
        if len(valid_responses) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} valid response(s), got {len(valid_responses)}"
            )
        
        # Get vote distribution
        vote_distribution = self.get_vote_distribution(valid_responses)
        
        # Calculate entropy
        entropy = self._calculate_entropy(vote_distribution)
        
        # Calculate unique responses
        unique_responses = len(vote_distribution)
        
        # Normalize entropy to 0-1 scale
        if self.normalize and unique_responses > 1:
            # Maximum entropy for n outcomes is log2(n)
            max_entropy = math.log2(unique_responses)
            uncertainty_score = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            # For single response, uncertainty is 0
            if unique_responses == 1:
                uncertainty_score = 0.0
            else:
                uncertainty_score = min(1.0, entropy / 10.0)  # Cap at 1.0
        
        # Ensure bounds
        uncertainty_score = max(0.0, min(1.0, uncertainty_score))
        
        return UncertaintyResult(
            uncertainty_score=uncertainty_score,
            entropy=entropy,
            unique_responses=unique_responses,
            vote_distribution=vote_distribution,
            method=method
        )
    
    def should_scale(self, uncertainty: float) -> bool:
        """
        Determine if compute should be scaled up based on uncertainty.
        
        Args:
            uncertainty: Uncertainty score (0.0-1.0)
            
        Returns:
            True if uncertainty exceeds the scale threshold
        """
        return uncertainty > self.scale_threshold
    
    def get_vote_distribution(self, responses: List[str]) -> Dict[str, float]:
        """
        Calculate the distribution of unique responses (vote proportions).
        
        Args:
            responses: List of response strings
            
        Returns:
            Dictionary mapping each unique response to its proportion (0.0-1.0)
        """
        if not responses:
            return {}
        
        # Normalize responses (trim whitespace, lowercase for comparison)
        normalized = [str(r).strip() for r in responses]
        
        # Count occurrences
        counts = Counter(normalized)
        total = len(normalized)
        
        # Convert to proportions
        return {response: count / total for response, count in counts.items()}
    
    def _calculate_entropy(self, distribution: Dict[str, float]) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        
        H = -Σ p(x) * log2(p(x))
        
        Args:
            distribution: Dictionary mapping outcomes to probabilities
            
        Returns:
            Entropy in bits
        """
        entropy = 0.0
        
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def estimate_from_confidence_scores(
        self,
        confidence_scores: List[float],
        method: str = "confidence_variance"
    ) -> UncertaintyResult:
        """
        Estimate uncertainty from agent confidence scores.
        
        Lower variance in confidence scores indicates higher agreement,
        while higher variance suggests disagreement and uncertainty.
        
        Args:
            confidence_scores: List of confidence scores (0.0-1.0) from agents
            method: Method identifier for the result
            
        Returns:
            UncertaintyResult with uncertainty metrics
        """
        if not confidence_scores:
            raise ValueError("confidence_scores cannot be empty")
        
        # Validate confidence scores
        for score in confidence_scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"confidence scores must be in [0, 1], got {score}")
        
        n = len(confidence_scores)
        
        # Calculate mean and variance
        mean_conf = sum(confidence_scores) / n
        variance = sum((x - mean_conf) ** 2 for x in confidence_scores) / n if n > 0 else 0.0
        
        # Higher variance in confidence = more uncertainty
        # Maximum variance for [0, 1] bounded distribution is 0.25
        uncertainty_score = min(1.0, variance * 4)  # Scale to 0-1
        
        # Create synthetic vote distribution for consistency
        vote_distribution = {
            f"confidence_{i:.2f}": 1.0 / n 
            for i in confidence_scores
        }
        
        return UncertaintyResult(
            uncertainty_score=uncertainty_score,
            entropy=variance,  # Use variance as entropy proxy
            unique_responses=n,
            vote_distribution=vote_distribution,
            method=method
        )
    
    def combine_uncertainties(
        self,
        uncertainties: List[UncertaintyResult],
        weights: Optional[List[float]] = None
    ) -> UncertaintyResult:
        """
        Combine multiple uncertainty estimates into a single estimate.
        
        Args:
            uncertainties: List of UncertaintyResult objects
            weights: Optional weights for each estimate (defaults to equal weights)
            
        Returns:
            Combined UncertaintyResult
        """
        if not uncertainties:
            raise ValueError("uncertainties cannot be empty")
        
        n = len(uncertainties)
        
        if weights is None:
            weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("weights must match number of uncertainties")
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Weighted average of uncertainty scores
        combined_score = sum(
            w * u.uncertainty_score 
            for w, u in zip(weights, uncertainties)
        )
        
        # Weighted average of entropy
        combined_entropy = sum(
            w * u.entropy 
            for w, u in zip(weights, uncertainties)
        )
        
        # Maximum unique responses
        max_unique = max(u.unique_responses for u in uncertainties)
        
        return UncertaintyResult(
            uncertainty_score=combined_score,
            entropy=combined_entropy,
            unique_responses=max_unique,
            vote_distribution={},  # Combined distribution not meaningful
            method="combined"
        )
    
    def get_scale_recommendation(
        self,
        uncertainty: float,
        current_agents: int = 1,
        max_agents: int = 10
    ) -> Dict[str, any]:
        """
        Get a recommendation for compute scaling based on uncertainty.
        
        Args:
            uncertainty: Current uncertainty score
            current_agents: Current number of agents being used
            max_agents: Maximum number of agents available
            
        Returns:
            Dictionary with scaling recommendation
        """
        should_scale_up = self.should_scale(uncertainty)
        
        # Linear scaling: more agents for higher uncertainty
        if should_scale_up:
            recommended_agents = min(
                max_agents,
                max(current_agents + 1, int(current_agents + uncertainty * 3))
            )
        else:
            recommended_agents = current_agents
        
        return {
            "should_scale": should_scale_up,
            "current_agents": current_agents,
            "recommended_agents": recommended_agents,
            "uncertainty": uncertainty,
            "threshold": self.scale_threshold
        }
