import unittest
from src.uncertainty import UncertaintyEstimator

class TestUncertaintyEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = UncertaintyEstimator(high_uncertainty_threshold=0.8)

    def test_zero_uncertainty(self):
        """Test with identical outcomes (zero entropy)."""
        branches = [
            {'outcome': 'A'},
            {'outcome': 'A'},
            {'outcome': 'A'}
        ]
        u = self.estimator.estimate_uncertainty(branches)
        self.assertEqual(u, 0.0)
        self.assertFalse(self.estimator.should_scale_compute(u))

    def test_high_uncertainty(self):
        """Test with diverse outcomes (high entropy)."""
        branches = [
            {'outcome': 'A'},
            {'outcome': 'B'},
            {'outcome': 'C'}
        ]
        u = self.estimator.estimate_uncertainty(branches)
        # Entropy of log2(3) = 1.58
        self.assertGreater(u, 1.0)
        self.assertTrue(self.estimator.should_scale_compute(u))

    def test_binary_uncertainty(self):
        """Test with split outcomes."""
        branches = [
            {'outcome': 'A'},
            {'outcome': 'B'}
        ]
        u = self.estimator.estimate_uncertainty(branches)
        # Entropy of 50/50 split is 1.0
        self.assertEqual(u, 1.0)
        self.assertTrue(self.estimator.should_scale_compute(u))

    def test_empty_branches(self):
        """Test behavior with no branches."""
        u = self.estimator.estimate_uncertainty([])
        self.assertEqual(u, 1.0)
        self.assertTrue(self.estimator.should_scale_compute(u))

if __name__ == '__main__':
    unittest.main()
