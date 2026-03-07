import json
import os
import unittest

class TestBeamDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.join(os.path.dirname(__file__), "../data/beam_dataset.json")
    
    def test_dataset_exists(self):
        self.assertTrue(os.path.exists(self.dataset_path))
    
    def test_dataset_format(self):
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 50)
        
        for item in data:
            self.assertIn("input", item)
            self.assertIn("ground_truth", item)
            
            inputs = item["input"]
            targets = item["ground_truth"]
            
            self.assertIn("L", inputs)
            self.assertIn("P", inputs)
            self.assertIn("a", inputs)
            
            self.assertIn("R0", targets)
            self.assertIn("RL", targets)
            
            # Physics check: R0 + RL should equal P
            L = inputs["L"]
            P = inputs["P"]
            a = inputs["a"]
            R0 = targets["R0"]
            RL = targets["RL"]
            
            self.assertAlmostEqual(R0 + RL, P, places=4)
            # R0 should be P * (L - a) / L
            self.assertAlmostEqual(R0, P * (L - a) / L, places=4)

if __name__ == "__main__":
    unittest.main()
