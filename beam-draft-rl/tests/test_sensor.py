import numpy as np
import unittest
from src.sensor import UncertaintySensor

class TestUncertaintySensor(unittest.TestCase):
    """
    Tests for the UncertaintySensor class.
    """

    def setUp(self):
        self.sensor = UncertaintySensor()

    def test_calculate_entropy(self):
        """
        Test entropy calculation with various probability distributions.
        """
        # Case 1: Uniform distribution (max entropy)
        # 1-bit entropy for 2 outcomes
        probs = np.array([0.5, 0.5])
        self.assertAlmostEqual(self.sensor.calculate_entropy(probs), 1.0)
        
        # Case 2: Deterministic distribution (min entropy)
        probs = np.array([1.0, 0.0])
        self.assertAlmostEqual(self.sensor.calculate_entropy(probs), 0.0)
        
        # Case 3: Mixed distribution
        probs = np.array([0.1, 0.9])
        # - (0.1 * log2(0.1) + 0.9 * log2(0.9)) ~= 0.469
        self.assertAlmostEqual(self.sensor.calculate_entropy(probs), 0.4689955935892812)

    def test_check_semantic_consistency(self):
        """
        Test semantic consistency scoring.
        """
        # Case 1: All responses represent the same result (perfect consistency)
        responses = ["same", "same", "same"]
        # Score should be 1/3 (ratio of unique/total)
        self.assertAlmostEqual(self.sensor.check_semantic_consistency(responses), 1/3)
        
        # Case 2: All responses are unique (total inconsistency)
        responses = ["a", "b", "c"]
        # Score should be 3/3 = 1.0
        self.assertAlmostEqual(self.sensor.check_semantic_consistency(responses), 1.0)
        
        # Case 3: Empty list
        self.assertAlmostEqual(self.sensor.check_semantic_consistency([]), 0.0)

    def test_detect_noise(self):
        """
        Test noise detection based on entropy trajectory.
        """
        # Case 1: Low entropy trajectory (no noise)
        trajectory = [0.1, 0.2, 0.1]
        self.assertFalse(self.sensor.detect_noise(trajectory, threshold=0.5))
        
        # Case 2: High entropy trajectory (noise)
        trajectory = [0.8, 0.9, 0.8]
        self.assertTrue(self.sensor.detect_noise(trajectory, threshold=0.5))
        
        # Case 3: Empty trajectory
        self.assertFalse(self.sensor.detect_noise([]))

if __name__ == '__main__':
    unittest.main()
