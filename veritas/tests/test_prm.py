#!/usr/bin/env python3
"""
Test Suite for PRM

Comprehensive tests for the VERITAS Process Reward Model component.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.prm import PRM, PRMConfig


class TestPRMConfig(unittest.TestCase):
    """Test PRMConfig functionality."""
    
    def test_prm_config_creation(self):
        """Test that PRMConfig can be created."""
        config = PRMConfig(
            model_name="bert-base-uncased",
            learning_rate=2e-5,
            batch_size=32,
            epochs=3,
            use_gpu=True
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 3)
        self.assertTrue(config.use_gpu)
    
    def test_prm_config_defaults(self):
        """Test PRMConfig with default values."""
        config = PRMConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.epochs, 2)
        self.assertFalse(config.use_gpu)


class TestPRM(unittest.TestCase):
    """Test PRM functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PRMConfig(
            model_name="bert-base-uncased",
            learning_rate=2e-5,
            batch_size=16,
            epochs=2,
            use_gpu=False
        )
    
    def test_prm_creation(self):
        """Test that PRM can be created."""
        prm = PRM(config=self.config)
        self.assertIsNotNone(prm)
        self.assertIsInstance(prm.config, PRMConfig)
        self.assertIsNotNone(prm.model)
    
    def test_prm_predict(self):
        """Test PRM prediction returns a score between 0 and 1."""
        prm = PRM(config=self.config)
        score = prm.predict("This is a test message.")
        self.assertTrue(0.0 <= score <= 1.0)
        
        # Test with different inputs
        score2 = prm.predict("Another test message with different content.")
        self.assertTrue(0.0 <= score2 <= 1.0)
        self.assertNotEqual(score, score2)
    
    def test_prm_evaluate_batch(self):
        """Test batch evaluation."""
        prm = PRM(config=self.config)
        texts = ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"]
        scores = prm.evaluate_batch(texts)
        self.assertEqual(len(scores), 5)
        for score in scores:
            self.assertTrue(0.0 <= score <= 1.0)
        
        # Check that scores are different for different texts
        unique_scores = set(scores)
        self.assertTrue(len(unique_scores) > 1)
    
    def test_prm_train(self):
        """Test PRM training returns metrics."""
        prm = PRM(config=self.config)
        
        # Generate some training data
        training_data = []
        for i in range(20):
            text = f"Training example {i}"
            score = 0.5 + (i * 0.03)  # Gradually increasing scores
            training_data.append((text, min(score, 1.0)))
        
        metrics = prm.train(training_data)
        self.assertIsNotNone(metrics)
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
        self.assertIn("val_loss", metrics)
        self.assertIn("val_accuracy", metrics)
        
        # Check that metrics are numbers
        for key in metrics:
            self.assertTrue(isinstance(metrics[key], (int, float)))
    
    def test_prm_save_load(self):
        """Test saving and loading PRM."""
        import tempfile
        import os
        
        prm = PRM(config=self.config)
        
        # Create a temporary directory for saving
        tmpdir = tempfile.mkdtemp()
        test_path = os.path.join(tmpdir, "prm_test.json")
        
        try:
            # Save
            save_success = prm.save(test_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(test_path))
            
            # Load into a new PRM instance
            new_prm = PRM(config=self.config)
            load_success = new_prm.load_from_file(test_path)
            self.assertTrue(load_success)
            
            # Verify that the loaded model can make predictions
            score1 = prm.predict("Test message 1")
            score2 = new_prm.predict("Test message 1")
            self.assertTrue(0.0 <= score1 <= 1.0)
            self.assertTrue(0.0 <= score2 <= 1.0)
            # They should be close but not necessarily identical due to randomness
            self.assertAlmostEqual(score1, score2, places=2)
        finally:
            # Clean up
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
    
    def test_prm_with_custom_model(self):
        """Test PRM with a custom model name."""
        custom_config = PRMConfig(
            model_name="roberta-base",
            learning_rate=1e-4,
            batch_size=8,
            epochs=1,
            use_gpu=False
        )
        prm = PRM(config=custom_config)
        self.assertIsNotNone(prm)
        self.assertEqual(prm.config.model_name, "roberta-base")
    
    def test_prm_gpu_support(self):
        """Test PRM GPU support (even if GPU not available)."""
        gpu_config = PRMConfig(
            model_name="bert-base-uncased",
            learning_rate=2e-5,
            batch_size=16,
            epochs=2,
            use_gpu=True  # Will try to use GPU, but fall back if not available
        )
        prm = PRM(config=gpu_config)
        self.assertIsNotNone(prm)
        # Should still work even if GPU is not available
        score = prm.predict("Test GPU fallback")
        self.assertTrue(0.0 <= score <= 1.0)
    
    def test_prm_batch_consistency(self):
        """Test that batch evaluation is consistent with individual evaluation."""
        prm = PRM(config=self.config)
        
        # Individual predictions
        individual_scores = []
        for i in range(5):
            text = f"Test message {i}"
            score = prm.predict(text)
            individual_scores.append(score)
        
        # Batch prediction
        texts = [f"Test message {i}" for i in range(5)]
        batch_scores = prm.evaluate_batch(texts)
        
        self.assertEqual(len(individual_scores), len(batch_scores))
        for ind_score, batch_score in zip(individual_scores, batch_scores):
            self.assertAlmostEqual(ind_score, batch_score, places=4)
    
    def test_prm_with_empty_input(self):
        """Test PRM with empty input."""
        prm = PRM(config=self.config)
        score = prm.predict("")
        self.assertTrue(0.0 <= score <= 1.0)
        # Empty input should probably get a low score
        self.assertLess(score, 0.5)
    
    def test_prm_with_short_input(self):
        """Test PRM with very short input."""
        prm = PRM(config=self.config)
        score = prm.predict("OK")
        self.assertTrue(0.0 <= score <= 1.0)
        # Very short input should get a moderate score
        self.assertGreaterEqual(score, 0.3)
        self.assertLessEqual(score, 0.7)


class TestPRMIntegrationWithCore(unittest.TestCase):
    """Test integration between PRM and core components."""
    
    def test_full_evaluation_with_prm(self):
        """Test full evaluation with PRM scoring."""
        from veritas.core import ConstitutionalReviewer, ConstitutionalContract, ConstitutionalPrinciples, evaluate_step
        
        # Create reviewer with PRM
        reviewer = ConstitutionalReviewer(use_prm=True)
        
        # Create a contract
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HALLUCINATION],
            prm_threshold=0.5,
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3
        )
        
        # Test with different outputs
        test_cases = [
            {
                "name": "Good output",
                "output": "This is factually correct and well-supported.",
                "expected_pass": True,
                "expected_prm_range": (0.5, 1.0)
            },
            {
                "name": "Bad output",
                "output": "This is factually incorrect and unsupported.",
                "expected_pass": False,
                "expected_prm_range": (0.0, 0.5)
            }
        ]
        
        for test_case in test_cases:
            result = evaluate_step(test_case["output"], contract, logger=None)
            is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
            
            self.assertEqual(is_pass, test_case["expected_pass"])
            self.assertTrue(test_case["expected_prm_range"][0] <= prm_score <= test_case["expected_prm_range"][1])


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)