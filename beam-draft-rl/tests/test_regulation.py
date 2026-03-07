import unittest
from src.regulator import Regulator
from src.corrector import Corrector

class TestRegulation(unittest.TestCase):

    def test_regulator_adjustment(self):
        reg = Regulator(base_temperature=1.0, base_top_p=1.0)
        
        # Test low uncertainty (expect base values)
        low_adj = reg.adjust_sampling(0.0)
        self.assertEqual(low_adj["temperature"], 1.0)
        self.assertEqual(low_adj["top_p"], 1.0)
        
        # Test extreme high uncertainty (expect min values)
        high_adj = reg.adjust_sampling(1.0)
        self.assertEqual(high_adj["temperature"], 0.1)
        self.assertEqual(high_adj["top_p"], 0.5)

        # Test middle ground
        mid_adj = reg.adjust_sampling(0.5)
        self.assertLess(mid_adj["temperature"], 1.0)
        self.assertGreater(mid_adj["temperature"], 0.1)

    def test_corrector_feedback_correct(self):
        cor = Corrector()
        reasoning = "The answer is 42."
        truth = "The answer is 42."
        trajectory = [0.1, 0.2, 0.1] # Low noise
        
        feedback = cor.generate_feedback(trajectory, reasoning, truth)
        self.assertIn("Correct answer", feedback)
        self.assertIn("high confidence", feedback)

    def test_corrector_feedback_incorrect_noisy(self):
        cor = Corrector()
        reasoning = "The answer is 42."
        truth = "The answer is 13."
        trajectory = [0.1, 0.8, 0.2] # Noise spike at step 1
        
        feedback = cor.generate_feedback(trajectory, reasoning, truth)
        self.assertIn("Incorrect answer", feedback)
        self.assertIn("noise spike", feedback)
        self.assertIn("step 1", feedback)

if __name__ == '__main__':
    unittest.main()
