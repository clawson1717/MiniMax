"""Uncertainty Statistics Module - CATTS (Confidence Adaptive Tree of Thought Sampling).

This module implements uncertainty quantification through vote distribution statistics.
Based on the CATTS paper, models self-assess uncertainty by analyzing distributions
over candidate actions. High uncertainty triggers more reasoning samples; low uncertainty
uses minimal compute.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import math
from collections import Counter
import random


@dataclass
class VoteDistribution:
    """Stores candidate actions and their vote counts from ensemble sampling.
    
    In CATTS, multiple reasoning samples are generated, and each "votes" for an action.
    The distribution of votes reveals the model's uncertainty about the best action.
    """
    candidates: Dict[str, int] = field(default_factory=dict)
    total_votes: int = 0
    observation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_vote(self, action: str) -> None:
        """Add a vote for a candidate action."""
        self.candidates[action] = self.candidates.get(action, 0) + 1
        self.total_votes += 1
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get probability distribution over candidates."""
        if self.total_votes == 0:
            return {}
        return {
            action: count / self.total_votes 
            for action, count in self.candidates.items()
        }
    
    def get_most_common(self) -> Tuple[str, int]:
        """Get the most voted action and its count."""
        if not self.candidates:
            return ("", 0)
        return max(self.candidates.items(), key=lambda x: x[1])
    
    def get_confidence(self) -> float:
        """Get confidence as the proportion of votes for the top action."""
        if self.total_votes == 0:
            return 0.0
        _, top_count = self.get_most_common()
        return top_count / self.total_votes


class UncertaintyEstimator:
    """
    Estimates uncertainty using vote distribution statistics from CATTS.
    
    Key insight: Instead of single action selection, generate N samples and analyze
    the distribution. Metrics like entropy, variance, and pairwise disagreement
    quantify uncertainty to guide adaptive compute allocation.
    """
    
    def __init__(
        self, 
        method: str = "ensemble", 
        num_samples: int = 5,
        random_seed: Optional[int] = None
    ):
        """Initialize uncertainty estimator.
        
        Args:
            method: Uncertainty estimation method ('ensemble', 'entropy', 'variance').
            num_samples: Default number of samples for ensemble methods.
            random_seed: Random seed for reproducible sampling.
        """
        self.method = method
        self.num_samples = num_samples
        self._history: List[Dict[str, Any]] = []
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def generate_votes(
        self, 
        observation: str, 
        n_samples: int,
        candidate_actions: Optional[List[str]] = None
    ) -> VoteDistribution:
        """Generate N candidate actions per step and collect votes.
        
        In a real implementation, this would call the LLM N times with temperature > 0
        to get diverse reasoning paths. For simulation, we can use provided candidates
        or generate plausible actions.
        
        Args:
            observation: The current observation/state.
            n_samples: Number of reasoning samples to generate.
            candidate_actions: Optional list of possible actions. If None, simulates
                             by extracting from observation context.
            
        Returns:
            VoteDistribution containing votes for each candidate action.
        """
        votes = VoteDistribution(observation=observation)
        
        # Simulate generating diverse reasoning samples
        # In practice, this would be: for i in range(n_samples): llm.generate(observation)
        for _ in range(n_samples):
            action = self._simulate_reasoning_sample(observation, candidate_actions)
            votes.add_vote(action)
        
        votes.metadata = {
            "n_samples": n_samples,
            "method": self.method,
            "observation_length": len(observation),
        }
        
        return votes
    
    def _simulate_reasoning_sample(
        self, 
        observation: str,
        candidate_actions: Optional[List[str]] = None
    ) -> str:
        """Simulate a single reasoning sample.
        
        In production, this would call an LLM. Here we simulate based on context
        to create realistic vote distributions.
        """
        if candidate_actions:
            # Weight toward first action (simulating higher confidence)
            weights = [max(1, len(candidate_actions) - i) for i in range(len(candidate_actions))]
            total = sum(weights)
            r = random.uniform(0, total)
            cumulative = 0
            for action, weight in zip(candidate_actions, weights):
                cumulative += weight
                if r <= cumulative:
                    return action
            return candidate_actions[-1]
        
        # Extract possible actions from observation (simplified simulation)
        # In reality, the LLM would generate these based on understanding
        if "button" in observation.lower():
            actions = ["click_button", "fill_form", "scroll_down", "go_back"]
        elif "form" in observation.lower():
            actions = ["fill_form", "submit_form", "clear_field", "go_back"]
        elif "search" in observation.lower():
            actions = ["type_query", "press_enter", "clear_search", "go_back"]
        else:
            actions = ["click", "type", "scroll", "wait", "go_back"]
        
        # Simulate some uncertainty in selection
        return random.choice(actions)
    
    def compute_entropy(self, votes: VoteDistribution) -> float:
        """Calculate vote entropy to measure uncertainty.
        
        Entropy measures the uncertainty in the vote distribution.
        High entropy = votes spread across many actions = high uncertainty.
        Low entropy = votes concentrated on few actions = low uncertainty.
        
        H(X) = -sum(p(x) * log2(p(x)))
        
        Args:
            votes: VoteDistribution containing candidate votes.
            
        Returns:
            Entropy value (0 = certain, higher = more uncertain).
        """
        if votes.total_votes == 0:
            return 0.0
        
        probs = votes.get_probabilities()
        entropy = 0.0
        
        for prob in probs.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def compute_variance(self, votes: VoteDistribution) -> float:
        """Calculate confidence variance across candidates.
        
        Variance measures how spread out the vote counts are.
        High variance = one action dominates = low uncertainty.
        Low variance = votes evenly distributed = high uncertainty.
        
        Args:
            votes: VoteDistribution containing candidate votes.
            
        Returns:
            Variance of vote counts.
        """
        if votes.total_votes == 0 or len(votes.candidates) <= 1:
            return 0.0
        
        counts = list(votes.candidates.values())
        mean = sum(counts) / len(counts)
        variance = sum((x - mean) ** 2 for x in counts) / len(counts)
        
        return variance
    
    def compute_pairwise_disagreement(self, votes: VoteDistribution) -> float:
        """Measure action disagreement using pairwise comparison.
        
        Calculates the proportion of vote pairs that disagree on the best action.
        This captures how often different reasoning paths lead to different conclusions.
        
        Pairwise disagreement = (pairs disagreeing) / (total pairs)
        
        Args:
            votes: VoteDistribution containing candidate votes.
            
        Returns:
            Pairwise disagreement ratio (0 = all agree, 1 = maximum disagreement).
        """
        if votes.total_votes <= 1:
            return 0.0
        
        candidates = list(votes.candidates.keys())
        if len(candidates) <= 1:
            return 0.0
        
        # Build list of individual votes
        individual_votes = []
        for action, count in votes.candidates.items():
            individual_votes.extend([action] * count)
        
        # Count disagreeing pairs
        total_pairs = 0
        disagreeing_pairs = 0
        
        for i in range(len(individual_votes)):
            for j in range(i + 1, len(individual_votes)):
                total_pairs += 1
                if individual_votes[i] != individual_votes[j]:
                    disagreeing_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return disagreeing_pairs / total_pairs
    
    def get_uncertainty_stats(self, votes: VoteDistribution) -> Dict[str, float]:
        """Compute all uncertainty metrics for a vote distribution.
        
        Args:
            votes: VoteDistribution containing candidate votes.
            
        Returns:
            Dictionary with all uncertainty statistics:
            - entropy: Vote entropy (higher = more uncertain)
            - normalized_entropy: Entropy normalized by max possible
            - variance: Vote count variance
            - pairwise_disagreement: Pairwise disagreement ratio
            - confidence: Confidence in top action
            - num_candidates: Number of unique candidates
            - total_votes: Total number of votes
        """
        entropy = self.compute_entropy(votes)
        variance = self.compute_variance(votes)
        disagreement = self.compute_pairwise_disagreement(votes)
        confidence = votes.get_confidence()
        
        # Normalize entropy by maximum possible (log2 of num candidates)
        num_candidates = len(votes.candidates)
        max_entropy = math.log2(num_candidates) if num_candidates > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "variance": variance,
            "pairwise_disagreement": disagreement,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,  # Complement of confidence
            "num_candidates": num_candidates,
            "total_votes": votes.total_votes,
        }
    
    def should_scale_up(
        self, 
        stats: Dict[str, float], 
        threshold: float = 0.5
    ) -> bool:
        """Determine if more compute (samples) is needed based on uncertainty.
        
        High uncertainty suggests the model is unsure and would benefit from
        additional reasoning samples to get a more reliable distribution.
        
        Args:
            stats: Uncertainty statistics from get_uncertainty_stats().
            threshold: Uncertainty threshold for scaling up (0-1).
            
        Returns:
            True if more samples should be generated, False otherwise.
        """
        # Use normalized entropy as the primary uncertainty measure
        uncertainty = stats.get("normalized_entropy", 0.0)
        
        # Also consider pairwise disagreement for tie-breaking
        disagreement = stats.get("pairwise_disagreement", 0.0)
        
        # Scale up if either metric meets or exceeds threshold
        return uncertainty >= threshold or disagreement >= threshold
    
    def get_compute_budget(
        self, 
        stats: Dict[str, float],
        min_samples: int = 3,
        max_samples: int = 20
    ) -> int:
        """Dynamically allocate compute budget based on uncertainty.
        
        Higher uncertainty → more samples needed
        Lower uncertainty → fewer samples sufficient
        
        Args:
            stats: Uncertainty statistics from get_uncertainty_stats().
            min_samples: Minimum number of reasoning samples.
            max_samples: Maximum number of reasoning samples.
            
        Returns:
            Recommended number of samples for the current state.
        """
        uncertainty = stats.get("normalized_entropy", 0.5)
        
        # Linear interpolation between min and max based on uncertainty
        # High uncertainty (1.0) → max_samples
        # Low uncertainty (0.0) → min_samples
        budget = min_samples + (max_samples - min_samples) * uncertainty
        
        return int(round(budget))
    
    # Legacy methods for backward compatibility
    
    def estimate(self, action: str, context: Dict[str, Any]) -> float:
        """Estimate uncertainty for an action (legacy method).
        
        Args:
            action: The action to evaluate.
            context: Context information.
            
        Returns:
            Uncertainty score between 0 and 1.
        """
        observation = context.get("observation", action)
        n_samples = context.get("n_samples", self.num_samples)
        
        votes = self.generate_votes(observation, n_samples)
        stats = self.get_uncertainty_stats(votes)
        
        return stats["uncertainty"]
    
    def estimate_action_sequence(self, actions: List[str], context: Dict[str, Any]) -> List[float]:
        """Estimate uncertainty for a sequence of actions (legacy method).
        
        Args:
            actions: List of actions to evaluate.
            context: Context information.
            
        Returns:
            List of uncertainty scores.
        """
        return [self.estimate(action, context) for action in actions]
    
    def calibrate(self, predictions: List[Any], actuals: List[Any]) -> None:
        """Calibrate uncertainty estimates based on past performance.
        
        Args:
            predictions: List of predicted outcomes.
            actuals: List of actual outcomes.
        """
        # TODO: Implement calibration logic using reliability diagrams
        # or temperature scaling
        pass
    
    def should_pause(self, uncertainty: float, threshold: float = 0.7) -> bool:
        """Determine if agent should pause for human input based on uncertainty.
        
        Args:
            uncertainty: Current uncertainty score.
            threshold: Uncertainty threshold for pausing.
            
        Returns:
            True if agent should pause, False otherwise.
        """
        return uncertainty >= threshold
    
    def get_confidence(self, uncertainty: float) -> float:
        """Convert uncertainty to confidence score.
        
        Args:
            uncertainty: Uncertainty score (0-1).
            
        Returns:
            Confidence score (0-1).
        """
        return 1.0 - uncertainty
    
    def reset(self) -> None:
        """Clear uncertainty history."""
        self._history = []
