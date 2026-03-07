import unittest
import numpy as np
from src.curriculum import CurriculumManager, TrainingStage

class TestCurriculumManager(unittest.TestCase):
    def setUp(self):
        # Small window and threshold to speed up tests
        self.manager = CurriculumManager(
            mastery_threshold=0.8,
            window_size=10,
            min_samples_per_stage=20
        )

    def test_initial_stage(self):
        self.assertEqual(self.manager.get_current_stage(), TrainingStage.FULL_COT)

    def test_update_mastery(self):
        # Adding some history and samples
        self.manager.update_mastery(0.9)
        self.assertEqual(len(self.manager.history), 1)
        self.assertEqual(self.manager.samples_in_current_stage, 1)

    def test_stage_advancement(self):
        # Simulate mastery for the first stage
        # Need at least self.manager.min_samples_per_stage (20)
        # and self.manager.window_size (10)
        for _ in range(20):
            self.manager.update_mastery(0.85) # Above threshold

        # Should have advanced to HYBRID
        self.assertEqual(self.manager.get_current_stage(), TrainingStage.HYBRID_DRAFT)
        self.assertEqual(self.manager.samples_in_current_stage, 0)
        self.assertEqual(len(self.manager.history), 0)

    def test_no_advancement_below_threshold(self):
        # Performance is low, should not advance even with enough samples
        for _ in range(50):
            self.manager.update_mastery(0.5)

        self.assertEqual(self.manager.get_current_stage(), TrainingStage.FULL_COT)

    def test_no_advancement_below_min_samples(self):
        # Performance is high, but not enough samples yet
        for _ in range(5):
            self.manager.update_mastery(0.95)

        self.assertEqual(self.manager.get_current_stage(), TrainingStage.FULL_COT)

    def test_prompt_modification(self):
        original_prompt = "Solve 2+2."
        
        # FULL_COT logic
        self.manager.current_stage = TrainingStage.FULL_COT
        cot_prompt = self.manager.modify_prompt(original_prompt)
        self.assertIn("step-by-step", cot_prompt)
        
        # HYBRID_DRAFT logic
        self.manager.current_stage = TrainingStage.HYBRID_DRAFT
        hybrid_prompt = self.manager.modify_prompt(original_prompt)
        self.assertIn("structured drafting", hybrid_prompt)
        
        # PURE_DRAFT logic
        self.manager.current_stage = TrainingStage.PURE_DRAFT
        pure_prompt = self.manager.modify_prompt(original_prompt)
        self.assertIn("Directly draft", pure_prompt)

    def test_stage_config(self):
        self.manager.current_stage = TrainingStage.FULL_COT
        config = self.manager.get_stage_config()
        self.assertEqual(config["draft_ratio"], 0.0)

        self.manager.current_stage = TrainingStage.PURE_DRAFT
        config = self.manager.get_stage_config()
        self.assertEqual(config["draft_ratio"], 1.0)

if __name__ == "__main__":
    unittest.main()
