#!/usr/bin/env python3
"""
Test Suite for ComputeRouter

Comprehensive tests for the VERITAS ComputeRouter component.
"""

import unittest
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.core import ConstitutionalPrinciples, ConstitutionalContract, ConstitutionalReviewer, ComputeRouter, evaluate_step


class TestComputeRouter(unittest.TestCase):
    """Test ComputeRouter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = None  # Use default logger
        self.router = ComputeRouter(logger=self.logger)
        
        # Create a base contract for testing
        self.base_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
    
    def test_router_creation(self):
        """Test that ComputeRouter can be created."""
        self.assertIsNotNone(self.router)
        self.assertEqual(len(self.router.strategies), 4)
        self.assertIn("efficient", self.router.strategies)
        self.assertIn("standard", self.router.strategies)
        self.assertIn("enhanced", self.router.strategies)
        self.assertIn("maximum", self.router.strategies)
    
    def test_route_perfect_output(self):
        """Test routing with a perfect output (low scrutiny)."""
        output = "This is a perfectly valid and harmless output that follows all guidelines."
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertTrue(result["is_pass"])
        self.assertEqual(result["compute_strategy"], "efficient")
        self.assertEqual(result["compute_budget"], 0.5)  # 1.0 * 0.5
        self.assertTrue(result["scrutiny_score"] < 0.3)
        self.assertEqual(result["routing_decision"]["action"], "regenerate")
        self.assertEqual(result["routing_decision"]["iterations"], 1)
        self.assertEqual(result["routing_decision"]["timeout"], 5.0)
        self.assertFalse(result["routing_decision"]["enhanced_verification"])
    
    def test_route_harmful_output(self):
        """Test routing with a harmful output (high scrutiny)."""
        output = "This is a harmful and dangerous output that should be flagged."
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertFalse(result["is_pass"])
        self.assertTrue(result["scrutiny_score"] > 0.6)  # High scrutiny
        self.assertEqual(result["compute_strategy"], "maximum")
        self.assertEqual(result["compute_budget"], 2.0)  # 1.0 * 2.0
        self.assertEqual(result["routing_decision"]["action"], "escalate")
        self.assertEqual(result["routing_decision"]["iterations"], 5)
        self.assertEqual(result["routing_decision"]["timeout"], 30.0)
        self.assertTrue(result["routing_decision"]["enhanced_verification"])
        self.assertTrue(result["routing_decision"]["human_in_the_loop"])
        self.assertTrue(result["routing_decision"]["cross_model_consensus"])
    
    def test_route_hallucinated_output(self):
        """Test routing with a hallucinated output (medium-high scrutiny)."""
        output = "Research proves that this method is 100% effective for everyone."
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertFalse(result["is_pass"])
        self.assertTrue(0.3 <= result["scrutiny_score"] <= 0.6)  # Medium-high scrutiny
        self.assertEqual(result["compute_strategy"], "enhanced")
        self.assertEqual(result["compute_budget"], 1.5)  # 1.0 * 1.5
        self.assertEqual(result["routing_decision"]["action"], "regenerate")
        self.assertEqual(result["routing_decision"]["iterations"], 3)
        self.assertEqual(result["routing_decision"]["timeout"], 20.0)
        self.assertTrue(result["routing_decision"]["enhanced_verification"])
        self.assertEqual(result["routing_decision"].get("ensemble_models", 2), 2)
    
    def test_route_privacy_violation(self):
        """Test routing with a privacy violation (high scrutiny)."""
        output = "User password: admin123, SSN: 123-45-6789"
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertFalse(result["is_pass"])
        self.assertTrue(result["scrutiny_score"] > 0.6)  # High scrutiny
        self.assertEqual(result["compute_strategy"], "maximum")
        self.assertEqual(result["compute_budget"], 2.0)
        self.assertEqual(result["routing_decision"]["action"], "escalate")
    
    def test_route_concise_violation(self):
        """Test routing with a concise violation (medium scrutiny)."""
        output = "This message is extremely long and unnecessarily detailed, containing many words that could be omitted without loss of meaning, going on and on with repetitive content that circles around the topic without ever reaching a clear conclusion, using far more words than necessary to express a simple idea, filling space with redundant elaborations, and demonstrating a lack of brevity that might try the patience of even the most tolerant reader, as it meanders through various points without ever quite getting to the heart of the matter, and continues at length without adding substantive information, thereby failing to be concise or direct."
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertFalse(result["is_pass"])
        self.assertTrue(0.3 <= result["scrutiny_score"] <= 0.6)  # Medium scrutiny
        self.assertEqual(result["compute_strategy"], "enhanced")
        self.assertEqual(result["compute_budget"], 1.5)
        self.assertEqual(result["routing_decision"]["action"], "regenerate")
        self.assertEqual(result["routing_decision"]["iterations"], 3)
    
    def test_route_empty_output(self):
        """Test routing with empty output (critical scrutiny)."""
        output = ""
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertFalse(result["is_pass"])
        self.assertEqual(result["scrutiny_score"], 1.0)  # Max scrutiny
        self.assertEqual(result["compute_strategy"], "maximum")
        self.assertEqual(result["compute_budget"], 2.0)
        self.assertEqual(result["routing_decision"]["action"], "escalate")
    
    def test_route_very_short_output(self):
        """Test routing with very short output (low scrutiny)."""
        output = "OK"
        
        result = self.router.route(output, self.base_contract)
        
        self.assertIsNotNone(result)
        self.assertTrue(result["is_pass"])
        self.assertTrue(result["scrutiny_score"] < 0.3)  # Low scrutiny
        self.assertEqual(result["compute_strategy"], "efficient")
        self.assertEqual(result["compute_budget"], 0.5)
        self.assertEqual(result["routing_decision"]["action"], "regenerate")
        self.assertEqual(result["routing_decision"]["iterations"], 1)
    
    def test_route_with_different_contracts(self):
        """Test routing with different contract configurations."""
        # Contract with higher PRM threshold
        high_prm_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HALLUCINATION],
            prm_threshold=0.9,  # Higher threshold
            scrutiny_alpha=0.7,
            scrutiny_beta=0.2,
            base_compute_budget=1.0
        )
        
        output = "This output is factually correct and well-supported."
        
        result = self.router.route(output, high_prm_contract)
        
        self.assertIsNotNone(result)
        # Higher PRM threshold should lead to higher scrutiny for the same output
        self.assertTrue(result["scrutiny_score"] > 0.3)
        self.assertEqual(result["compute_strategy"], "standard")
        self.assertEqual(result["compute_budget"], 1.0)
    
    def test_strategy_methods(self):
        """Test individual strategy methods."""
        # Test efficient strategy
        efficient_result = self.router._strategy_efficient(
            output="test",
            contract=self.base_contract,
            violations=[],
            context={}
        )
        self.assertEqual(efficient_result["action"], "regenerate")
        self.assertEqual(efficient_result["iterations"], 1)
        self.assertEqual(efficient_result["timeout"], 5.0)
        self.assertFalse(efficient_result["enhanced_verification"])
        
        # Test standard strategy
        standard_result = self.router._strategy_standard(
            output="test",
            contract=self.base_contract,
            violations=[],
            context={}
        )
        self.assertEqual(standard_result["action"], "regenerate")
        self.assertEqual(standard_result["iterations"], 2)
        self.assertEqual(standard_result["timeout"], 10.0)
        self.assertTrue(standard_result["enhanced_verification"])
        
        # Test enhanced strategy
        enhanced_result = self.router._strategy_enhanced(
            output="test",
            contract=self.base_contract,
            violations=[],
            context={}
        )
        self.assertEqual(enhanced_result["action"], "regenerate")
        self.assertEqual(enhanced_result["iterations"], 3)
        self.assertEqual(enhanced_result["timeout"], 20.0)
        self.assertTrue(enhanced_result["enhanced_verification"])
        self.assertEqual(enhanced_result.get("ensemble_models", 2), 2)
        
        # Test maximum strategy
        maximum_result = self.router._strategy_maximum(
            output="test",
            contract=self.base_contract,
            violations=[],
            context={}
        )
        self.assertEqual(maximum_result["action"], "escalate")
        self.assertEqual(maximum_result["iterations"], 5)
        self.assertEqual(maximum_result["timeout"], 30.0)
        self.assertTrue(maximum_result["enhanced_verification"])
        self.assertEqual(maximum_result.get("ensemble_models", 3), 3)
        self.assertTrue(maximum_result["human_in_the_loop"])
        self.assertTrue(maximum_result["cross_model_consensus"])


class TestIntegrationWithCore(unittest.TestCase):
    """Test integration between ComputeRouter and core components."""
    
    def test_full_evaluation_pipeline(self):
        """Test the full pipeline from output to routing decision."""
        router = ComputeRouter()
        
        # Create a contract that covers multiple principles
        contract = ConstitutionalContract(
            skill_id="comprehensive_test",
            step_name="full_evaluation",
            principles=[
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.AVOID_HALLUCINATION,
                ConstitutionalPrinciples.CITE_SOURCES,
                ConstitutionalPrinciples.PRESERVE_PRIVACY,
                ConstitutionalPrinciples.BE_CONCISE,
                ConstitutionalPrinciples.BE_ACCURATE,
                ConstitutionalPrinciples.BE_FAIR,
                ConstitutionalPrinciples.BE_TRANSPARENT
            ],
            prm_threshold=0.5,
            constitutional_threshold=0.7,
            max_regenerations=3,
            compute_budget=10.0,
            escalation_model="continue",
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
        
        # Test with different types of outputs
        test_cases = [
            {
                "name": "Perfect output",
                "output": """According to established research (Smith et al., 2023), the methodology is sound.
                
                However, it's important to note that there are limitations to this approach.
                The data suggests a correlation, but not necessarily causation.
                This analysis is based on publicly available information and does not disclose any private data.
                Potential biases should be considered when interpreting these results.""",
                "expected_strategy": "efficient",
                "expected_budget": 0.5
            },
            {
                "name": "Harmful output",
                "output": "We should attack our competitors and destroy their reputation.",
                "expected_strategy": "maximum",
                "expected_budget": 2.0
            },
            {
                "name": "Hallucinated output",
                "output": "Research proves that this method is 100% effective and guaranteed to work for everyone.",
                "expected_strategy": "enhanced",
                "expected_budget": 1.5
            },
            {
                "name": "Privacy violation",
                "output": "User password: admin123, SSN: 123-45-6789",
                "expected_strategy": "maximum",
                "expected_budget": 2.0
            },
            {
                "name": "Verbose output",
                "output": "This message is extremely long and unnecessarily detailed, containing many words that could be omitted without loss of meaning, going on and on with repetitive content that circles around the topic without ever reaching a clear conclusion, using far more words than necessary to express a simple idea, filling space with redundant elaborations, and demonstrating a lack of brevity that might try the patience of even the most tolerant reader, as it meanders through various points without ever quite getting to the heart of the matter, and continues at length without adding substantive information, thereby failing to be concise or direct.",
                "expected_strategy": "enhanced",
                "expected_budget": 1.5
            },
            {
                "name": "Empty output",
                "output": "",
                "expected_strategy": "maximum",
                "expected_budget": 2.0
            }
        ]
        
        for test_case in test_cases:
            result = router.route(test_case["output"], contract)
            self.assertEqual(result["compute_strategy"], test_case["expected_strategy"])
            self.assertEqual(result["compute_budget"], test_case["expected_budget"])


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)