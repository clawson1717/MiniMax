import numpy as np
from typing import List, Dict

class UncertaintySensor:
    """
    Sensor class for detecting uncertainty and noise in model responses.
    Adapted from DenoiseFlow sensing using entropy and semantic consistency.
    """

    def calculate_entropy(self, probs: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        
        Args:
            probs: Numpy array of probabilities.
            
        Returns:
            Entropy value.
        """
        # Ensure probs is a numpy array
        probs = np.array(probs)
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    def check_semantic_consistency(self, responses: List[str]) -> float:
        """
        Check semantic consistency across multiple responses.
        Simplified version: returns ratio of unique responses to total responses.
        Higher value indicates more inconsistency (more unique responses).
        
        Args:
            responses: List of response strings.
            
        Returns:
            Inconsistency score (0.0 to 1.0).
        """
        if not responses:
            return 0.0
        
        unique_responses = set(responses)
        return len(unique_responses) / len(responses)

    def detect_noise(self, trajectory: List[float], threshold: float = 0.5) -> bool:
        """
        Detect if a trajectory of entropy values indicates high noise.
        
        Args:
            trajectory: List of entropy values over time/steps.
            threshold: Entropy threshold to consider as noisy.
            
        Returns:
            True if noise is detected, False otherwise.
        """
        if not trajectory:
            return False
            
        # Basic noise detection: average entropy exceeds threshold
        avg_entropy = sum(trajectory) / len(trajectory)
        return avg_entropy > threshold
