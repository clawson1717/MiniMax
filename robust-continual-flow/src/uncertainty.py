import math
from collections import Counter
from typing import List, Any, Optional

class UncertaintyEstimator:
    """
    Provides methods to calculate uncertainty statistics for agent steps.
    Inspired by CATTS (Confidence-Aware Test-Time Scaling).
    """

    @staticmethod
    def from_votes(votes: List[Any]) -> float:
        """
        Calculates normalized categorical entropy from a list of votes.
        
        Args:
            votes: A list of responses or vote outcomes.
            
        Returns:
            A normalized uncertainty score between 0.0 and 1.0.
            0.0 means all votes are identical.
            1.0 means all votes are unique (maximum disagreement).
        """
        if not votes:
            return 0.0
        
        n_votes = len(votes)
        if n_votes <= 1:
            return 0.0
            
        counts = Counter(votes)
        entropy = 0.0
        for count in counts.values():
            p = count / n_votes
            if p > 0:
                entropy -= p * math.log(p)
        
        # Normalize by maximum possible entropy for the given number of votes
        # Max entropy occurs when every vote is unique: log(n_votes)
        max_entropy = math.log(n_votes)
        if max_entropy <= 0:
            return 0.0
            
        return entropy / max_entropy

    @staticmethod
    def from_probs(probabilities: List[float]) -> float:
        """
        Calculates normalized entropy from a probability distribution.
        
        Args:
            probabilities: A list of probabilities that sum to 1.0.
            
        Returns:
            A normalized uncertainty score between 0.0 and 1.0.
        """
        if not probabilities:
            return 0.0
            
        n_classes = len(probabilities)
        if n_classes <= 1:
            return 0.0
            
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log(p)
        
        # Max entropy for n classes is log(n)
        max_entropy = math.log(n_classes)
        if max_entropy <= 0:
            return 0.0
            
        return entropy / max_entropy
