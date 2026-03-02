import unittest
import math
from src.uncertainty import UncertaintyEstimator
from src.trajectory import TrajectoryLogger

class TestUncertainty(unittest.TestCase):
    def test_from_votes_consensus(self):
        # All votes are the same
        votes = ["A", "A", "A", "A"]
        uncertainty = UncertaintyEstimator.from_votes(votes)
        self.assertEqual(uncertainty, 0.0)

    def test_from_votes_max_disagreement(self):
        # All votes are different
        votes = ["A", "B", "C", "D"]
        uncertainty = UncertaintyEstimator.from_votes(votes)
        self.assertEqual(uncertainty, 1.0)

    def test_from_votes_partial_disagreement(self):
        # Half and half
        votes = ["A", "A", "B", "B"]
        uncertainty = UncertaintyEstimator.from_votes(votes)
        # H = -(0.5 * log(0.5) + 0.5 * log(0.5)) = log(2)
        # Max H = log(4)
        # Normalized = log(2) / log(4) = 0.5
        self.assertAlmostEqual(uncertainty, 0.5)

    def test_from_probs_certain(self):
        probs = [1.0, 0.0, 0.0]
        uncertainty = UncertaintyEstimator.from_probs(probs)
        self.assertEqual(uncertainty, 0.0)

    def test_from_probs_uncertain(self):
        probs = [1/3, 1/3, 1/3]
        uncertainty = UncertaintyEstimator.from_probs(probs)
        self.assertAlmostEqual(uncertainty, 1.0)

    def test_from_probs_partial(self):
        probs = [0.5, 0.5, 0.0]
        uncertainty = UncertaintyEstimator.from_probs(probs)
        # H = log(2), max H = log(3)
        expected = math.log(2) / math.log(3)
        self.assertAlmostEqual(uncertainty, expected)

    def test_empty_input(self):
        self.assertEqual(UncertaintyEstimator.from_votes([]), 0.0)
        self.assertEqual(UncertaintyEstimator.from_probs([]), 0.0)

    def test_single_input(self):
        self.assertEqual(UncertaintyEstimator.from_votes(["A"]), 0.0)
        self.assertEqual(UncertaintyEstimator.from_probs([1.0]), 0.0)

    def test_trajectory_logger_integration(self):
        logger = TrajectoryLogger()
        votes = ["A", "A", "B", "B"]
        node_id = logger.add_node("test observation", votes=votes)
        
        self.assertIn("uncertainty", logger.nodes[node_id]["metadata"])
        self.assertAlmostEqual(logger.nodes[node_id]["metadata"]["uncertainty"], 0.5)

if __name__ == "__main__":
    unittest.main()
