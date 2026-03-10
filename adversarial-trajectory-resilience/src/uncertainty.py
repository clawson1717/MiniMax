import math
from typing import List, Dict, Any

class UncertaintyEstimator:
    """
    Estimates uncertainty based on the distribution of votes/outcomes 
    from multiple trajectory branches.
    """

    def __init__(self, high_uncertainty_threshold: float = 1.0):
        """
        Args:
            high_uncertainty_threshold: Threshold above which compute should be scaled.
        """
        self.high_uncertainty_threshold = high_uncertainty_threshold

    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculates Shannon entropy for a list of counts."""
        total = sum(counts)
        if total <= 1:
            return 1.0 if total == 0 else 0.0
        
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def estimate_uncertainty(self, branches: List[Dict[str, Any]]) -> float:
        """
        Estimates uncertainty based on the distribution of outcomes in branches.
        Expects branches to have a key that represents the 'vote' or 'outcome'.
        """
        if not branches:
            return 1.0  # Maximum uncertainty if no data

        outcomes: Dict[Any, int] = {}
        for branch in branches:
            # Try to find a representative outcome for the branch
            outcome = branch.get('outcome', branch.get('result', branch.get('label', 'unknown')))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        counts = list(outcomes.values())
        return self._calculate_entropy(counts)

    def should_scale_compute(self, uncertainty: float) -> bool:
        """
        Decides if more compute (more branches/depth) should be allocated 
        based on the calculated uncertainty.
        """
        return uncertainty >= self.high_uncertainty_threshold
