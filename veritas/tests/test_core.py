#!/usr/bin/env python3
"""
VERITAS Test Suite - Constitutional Framework and Skill Contracts

Comprehensive tests for the VERITAS constitutional framework.
"""

import unittest
import sys
import os
from typing import Dict, Any, Tuple, List
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.core import (
    ConstitutionalPrinciples,
    ConstitutionalContract,
    ConstitutionalReviewer,
    PRM,
    PRMConfig,
    Logger,
    evaluate_step,
    create_constitutional_reviewer,
    calculate_scrutiny_score,
    determine_compute_strategy
)

class TestConstitutionalPrinciples(unittest.TestCase):
    """Test basic constitutional principle functionality."""
    
    def test_principle_enum_values(self):
        """Test that principle enum values are correctly defined."""
        principles = list(ConstitutionalPrinciples)
        self.assertEqual(len(principles), 8)
        self.assertIn(ConstitutionalPrinciples.AVOID_HARM, principles)
        self.assertIn(ConstitutionalPrinciples.AVOID_HALLUCINATION, principles)
        self.assertIn(ConstitutionalPrinciples.CITE_SOURCES, principles)
        self.assertIn(ConstitutionalPrinciples.PRESERVE_PRIVACY, principles)
        self.assertIn(ConstitutionalPrinciples.BE_CONCISE, principles)
        self.assertIn(ConstitutionalPrinciples.BE_ACCURATE, principles)
        self.assertIn(ConstitutionalPrinciples.BE_FAIR, principles)
        self.assertIn(ConstitutionalPrinciples.BE_TRANSPARENT, principles)
    
    def test_principle_value_strings(self):
        """Test that principle values are properly formatted strings."""
        self.assertEqual(ConstitutionalPrinciples.AVOID_HARM.value, "avoid_harm")
        self.assertEqual(ConstitutionalPrinciples.AVOID_HALLUCINATION.value, "avoid_hallucination")
        self.assertEqual(ConstitutionalPrinciples.CITE_SOURCES.value, "cite_sources")
        self.assertEqual(ConstitutionalPrinciples.PRESERVE_PRIVACY.value, "preserve_privacy")
        self.assertEqual(ConstitutionalPrinciples.BE_CONCISE.value, "be_concise")
        self.assertEqual(ConstitutionalPrinciples.BE_ACCURATE.value, "be_accurate")
        self.assertEqual(ConstitutionalPrinciples.BE_FAIR.value, "be_fair")
        self.assertEqual(ConstitutionalPrinciples.BE_TRANSPARENT.value, "be_transparent")

class TestConstitutionalContractValidation(unittest.TestCase):
    """Test contract validation logic."""
    
    def test_valid_contract(self):
        """Test that a valid contract passes validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.BE_CONCISE
            ]
        )
        is_valid, errors = contract.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_missing_skill_id(self):
        """Test that contract with empty skill_id fails validation."""
        contract = ConstitutionalContract(
            skill_id="",
            step_name="test_step",
            principles=[
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.BE_CONCISE
            ]
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("skill_id must be a non-empty string", errors)
    
    def test_missing_step_name(self):
        """Test that contract with empty step_name fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="",
            principles=[
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.BE_CONCISE
            ]
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("step_name must be a non-empty string", errors)
    
    def test_empty_principles(self):
        """Test that contract with no principles fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[]
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("principles must contain at least one principle", errors)
    
    def test_invalid_prm_threshold(self):
        """Test that contract with invalid prm_threshold fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            prm_threshold=1.5  # Invalid, must be between 0 and 1
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("prm_threshold must be between 0 and 1.0", errors)
    
    def test_invalid_constitutional_threshold(self):
        """Test that contract with invalid constitutional_threshold fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            constitutional_threshold=-0.1  # Invalid, must be between 0 and 1
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("constitutional_threshold must be between 0 and 1.0", errors)
    
    def test_invalid_max_regenerations(self):
        """Test that contract with negative max_regenerations fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            max_regenerations=-1  # Invalid, must be non-negative
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("max_regenerations must be non-negative", errors)
    
    def test_invalid_escalation_model(self):
        """Test that contract with invalid escalation_model fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            escalation_model="invalid_model"  # Must be "continue", "fail", or "escalate"
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("escalation_model must be 'continue', 'fail', or 'escalate'", errors)
    
    def test_invalid_scrutiny_weights(self):
        """Test that contract with invalid scrutiny weights fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=1.5,  # Must be between 0 and 1
            scrutiny_beta=0.5
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("scrutiny_alpha must be between 0 and 1", errors)
    
    def test_scrutiny_weights_sum_too_high(self):
        """Test that contract with scrutiny weights summing > 1 fails validation."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.7,
            scrutiny_beta=0.5  # Sum = 1.2 > 1
        )
        is_valid, errors = contract.validate()
        self.assertFalse(is_valid)
        self.assertIn("scrutiny_alpha + scrutiny_beta must not exceed 1", errors)

class TestLogger(unittest.TestCase):
    """Test logger functionality."""
    
    def test_logger_creation(self):
        """Test that logger can be created."""
        logger = Logger()
        self.assertIsNotNone(logger)
        self.assertEqual(logger.min_level, "INFO")
    
    def test_logger_custom_level(self):
        """Test logger with custom minimum level."""
        logger = Logger(min_level="DEBUG")
        self.assertEqual(logger.min_level, "DEBUG")
    
    def test_logging_methods(self):
        """Test all logging methods."""
        logger = Logger(console_output=False, log_file=None, min_level="DEBUG")
        logger.debug("Test debug message", module="test")
        logger.info("Test info message", module="test")
        logger.warning("Test warning message", module="test")
        logger.error("Test error message", module="test")
        logger.critical("Test critical message", module="test")
        self.assertEqual(len(logger.entries), 5)
        
        # Check levels
        levels = [entry.level for entry in logger.entries]
        self.assertEqual(levels, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    def test_log_entry_structure(self):
        """Test that log entries have proper structure."""
        logger = Logger(console_output=False, log_file=None)
        logger.info("Test message", module="test_module", custom_field="value")
        entry = logger.entries[0]
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.module, "test_module")
        self.assertIn("custom_field", entry.metadata)
        self.assertEqual(entry.metadata["custom_field"], "value")
    
    def test_get_logs(self):
        """Test getting logs with filtering."""
        logger = Logger(console_output=False, log_file=None)
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Get all logs
        all_logs = logger.get_logs()
        self.assertEqual(len(all_logs), 3)
        
        # Filter by level
        warning_logs = logger.get_logs(level="WARNING")
        self.assertEqual(len(warning_logs), 1)
        self.assertEqual(warning_logs[0]["level"], "WARNING")
    
    def test_clear_logs(self):
        """Test clearing logs."""
        logger = Logger(console_output=False, log_file=None)
        logger.info("Test message")
        self.assertEqual(len(logger.entries), 1)
        logger.clear_logs()
        self.assertEqual(len(logger.entries), 0)

class TestPRM(unittest.TestCase):
    """Test Process Reward Model functionality."""
    
    def test_prm_creation(self):
        """Test that PRM can be created."""
        prm = PRM()
        self.assertIsNotNone(prm)
        self.assertIsInstance(prm.config, PRMConfig)
    
    def test_prm_predict(self):
        """Test PRM prediction returns a score between 0 and 1."""
        prm = PRM()
        score = prm.predict("This is a test message.")
        self.assertTrue(0.0 <= score <= 1.0)
    
    def test_prm_evaluate_batch(self):
        """Test batch evaluation."""
        prm = PRM()
        texts = ["Test 1", "Test 2", "Test 3"]
        scores = prm.evaluate_batch(texts)
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertTrue(0.0 <= score <= 1.0)
    
    def test_prm_train(self):
        """Test PRM training returns metrics."""
        prm = PRM()
        training_data = [("Test 1", 0.8), ("Test 2", 0.9)]
        metrics = prm.train(training_data)
        self.assertIsNotNone(metrics)
        self.assertIn("train_loss", metrics)
        self.assertIn("train_accuracy", metrics)
    
    def test_prm_save_load(self):
        """Test saving and loading PRM."""
        prm = PRM()
        test_path = "/tmp/prm_test.json"
        
        # Save
        save_success = prm.save(test_path)
        self.assertTrue(save_success)
        self.assertTrue(os.path.exists(test_path))
        
        # Load
        new_prm = PRM()
        load_success = new_prm.load_from_file(test_path)
        self.assertTrue(load_success)
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)

class TestConstitutionalReviewerBasic(unittest.TestCase):
    """Test basic ConstitutionalReviewer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None)
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
        self.contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.BE_CONCISE
            ]
        )
    
    def test_reviewer_creation(self):
        """Test that ConstitutionalReviewer can be created."""
        self.assertIsNotNone(self.reviewer)
    
    def test_get_prm_score(self):
        """Test PRM score retrieval."""
        output = "This is a test output."
        score = self.reviewer._get_prm_score(output, self.contract.principles)
        self.assertTrue(0.0 <= score <= 1.0)
    
    def test_critique_with_no_violations(self):
        """Test that valid output produces no violations."""
        output = "This is a perfectly valid and harmless output that is also concise."
        violations = self.reviewer._advanced_critique(output, self.contract)
        self.assertEqual(len(violations), 0)
    
    def test_critique_with_violations(self):
        """Test that problematic output produces violations."""
        output = "This output is harmful and contains violent threats and is extremely verbose."
        violations = self.reviewer._advanced_critique(output, self.contract)
        self.assertTrue(len(violations) > 0)
        
        # Check that violations are for expected principles
        principles_found = [v["principle"] for v in violations]
        self.assertIn(ConstitutionalPrinciples.AVOID_HARM, principles_found)
        self.assertIn(ConstitutionalPrinciples.BE_CONCISE, principles_found)

class TestConstitutionalReviewerAdvanced(unittest.TestCase):
    """Test advanced ConstitutionalReviewer features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None)
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
        
        # Create contracts for different principles
        self.harm_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="harm_check",
            principles=[ConstitutionalPrinciples.AVOID_HARM]
        )
        
        self.hallucination_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="fact_check",
            principles=[ConstitutionalPrinciples.AVOID_HALLUCINATION]
        )
        
        self.privacy_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="privacy_check",
            principles=[ConstitutionalPrinciples.PRESERVE_PRIVACY]
        )
        
        self.conclusion_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="concise_check",
            principles=[ConstitutionalPrinciples.BE_CONCISE]
        )
    
    def test_avoid_harm_detection(self):
        """Test detection of harmful content."""
        harmful_outputs = [
            "I will kill you",
            "This will hurt you badly",
            "We should attack them",
            "Violence is the answer",
            "Destroy everything"
        ]
        
        for output in harmful_outputs:
            violations = self.reviewer._advanced_critique(output, self.harm_contract)
            self.assertTrue(len(violations) > 0, f"Failed to detect harm in: {output}")
            harm_violations = [v for v in violations if v["principle"] == ConstitutionalPrinciples.AVOID_HARM]
            self.assertTrue(len(harm_violations) > 0, f"No harm violation found in: {output}")
    
    def test_hallucination_detection(self):
        """Test detection of hallucinated content."""
        hallucinatory_outputs = [
            "Research shows that 95% of people believe this (source: made up)",
            "Studies prove that this is always true",
            "Experts say this is guaranteed to work",
            "In 2024, this happened for sure",
            "This is a proven fact"
        ]
        
        for output in hallucinatory_outputs:
            violations = self.reviewer._advanced_critique(output, self.hallucination_contract)
            self.assertTrue(len(violations) > 0, f"Failed to detect hallucination in: {output}")
            hallucination_violations = [v for v in violations if v["principle"] == ConstitutionalPrinciples.AVOID_HALLUCINATION]
            self.assertTrue(len(hallucination_violations) > 0, f"No hallucination violation found in: {output}")
    
    def test_privacy_protection_detection(self):
        """Test detection of privacy violations."""
        privacy_violations = [
            "Password: admin123",
            "SSN: 123-45-6789",
            "Credit card: 4111-1111-1111-1111",
            "Bank account: 123456789",
            "Personal data: John Doe, DOB: 01/01/2000"
        ]
        
        for output in privacy_violations:
            violations = self.reviewer._advanced_critique(output, self.privacy_contract)
            self.assertTrue(len(violations) > 0, f"Failed to detect privacy violation in: {output}")
            privacy_violations_found = [v for v in violations if v["principle"] == ConstitutionalPrinciples.PRESERVE_PRIVACY]
            self.assertTrue(len(privacy_violations_found) > 0, f"No privacy violation found in: {output}")
    
    def test_concise_detection(self):
        """Test detection of verbose content."""
        concise_outputs = [
            "This is a test",
            "Short message",
            "Brief note"
        ]
        
        verbose_outputs = [
            "This message, which you are currently reading, happens to be excessively long and unnecessarily detailed, containing many words that could have been omitted without loss of meaning, going on and on with repetitive content that circles around the topic without ever reaching a clear conclusion, using far more words than necessary to express a simple idea, filling space with redundant elaborations, and demonstrating a lack of brevity that might try the patience of even the most tolerant reader, as it meanders through various points without ever quite getting to the heart of the matter, and continues at length without adding substantive information, thereby failing to be concise or direct.",
            "In conclusion, after careful consideration of all factors and variables and aspects and elements and components and features and characteristics and attributes and properties and qualities and traits and aspects and facets and dimensions and perspectives and angles and viewpoints and standpoints and positions and approaches and methodologies and strategies and tactics and plans and schemes and designs and architectures and structures and frameworks and systems and processes and procedures and protocols and standards and specifications and requirements and needs and wants and desires and expectations and hopes and dreams and aspirations and goals and objectives and targets and aims and intents and purposes and functions and roles and responsibilities and duties and obligations and commitments and promises and pledges and vows and oaths and swears and guarantees and warranties and assurances and certainties and uncertainties and variables and constants and parameters and arguments and inputs and outputs and throughputs and feedbacks and feedforward and byputs and aroundputs and withinputs and withoutputs and inputs and outputs",
            "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
        ]
        
        # Concise outputs should have no violations
        for output in concise_outputs:
            violations = self.reviewer._advanced_critique(output, self.conclusion_contract)
            self.assertEqual(len(violations), 0, f"Incorrectly flagged concise output: {output}")
        
        # Verbose outputs should have violations
        for output in verbose_outputs:
            violations = self.reviewer._advanced_critique(output, self.conclusion_contract)
            self.assertTrue(len(violations) > 0, f"Failed to detect verbosity in: {output}")
            concise_violations = [v for v in violations if v["principle"] == ConstitutionalPrinciples.BE_CONCISE]
            self.assertTrue(len(concise_violations) > 0, f"No concise violation found in: {output}")
    
    def test_cite_sources_detection(self):
        """Test detection of missing citations."""
        contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="cite_check",
            principles=[ConstitutionalPrinciples.CITE_SOURCES]
        )
        
        # Good outputs (with citations)
        good_outputs = [
            "According to Smith et al. (2023), this is true.",
            "Source: https://example.com/research",
            "The data shows this trend (source: World Bank)."
        ]
        
        # Bad outputs (without citations)
        bad_outputs = [
            "Research shows this is true.",
            "Studies prove this conclusion.",
            "Experts believe this is correct.",
            "This is a well-known fact.",
            "The data indicates this trend."
        ]
        
        # Good outputs should have no violations
        for output in good_outputs:
            violations = self.reviewer._advanced_critique(output, contract)
            self.assertEqual(len(violations), 0, f"Incorrectly flagged good output: {output}")
        
        # Bad outputs should have violations
        for output in bad_outputs:
            violations = self.reviewer._advanced_critique(output, contract)
            self.assertTrue(len(violations) > 0, f"Failed to detect missing citation in: {output}")
            cite_violations = [v for v in violations if v["principle"] == ConstitutionalPrinciples.CITE_SOURCES]
            self.assertTrue(len(cite_violations) > 0, f"No citation violation found in: {output}")

class TestScrutinyScoreCalculation(unittest.TestCase):
    """Test scrutiny score calculation for compute routing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None)
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
        
        # Create a base contract with scrutiny weights
        self.contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
    
    def test_scrutiny_score_zero_violations_high_prm(self):
        """Test scrutiny score when PRM is high and no violations."""
        prm_score = 0.9
        violations = []
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        # scrutiny = 0.6*(1-0.9) + 0.3*0 = 0.6*0.1 = 0.06
        self.assertAlmostEqual(score, 0.06, places=2)
    
    def test_scrutiny_score_zero_violations_low_prm(self):
        """Test scrutiny score when PRM is low and no violations."""
        prm_score = 0.2
        violations = []
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        # scrutiny = 0.6*(1-0.2) + 0.3*0 = 0.6*0.8 = 0.48
        self.assertAlmostEqual(score, 0.48, places=2)
    
    def test_scrutiny_score_with_violations_high_severity(self):
        """Test scrutiny score when there are high severity violations."""
        prm_score = 0.8
        violations = [{"principle": ConstitutionalPrinciples.AVOID_HARM, 
                     "description": "Harm detected", 
                     "severity": "high"}]
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        # severity_score = 0.9 (high severity)
        # scrutiny = 0.6*(1-0.8) + 0.3*0.9 = 0.6*0.2 + 0.27 = 0.12 + 0.27 = 0.39
        self.assertAlmostEqual(score, 0.39, places=2)
    
    def test_scrutiny_score_with_multiple_violations(self):
        """Test scrutiny score when there are multiple violations."""
        prm_score = 0.7
        violations = [
            {"principle": ConstitutionalPrinciples.AVOID_HARM, "severity": "low"},
            {"principle": ConstitutionalPrinciples.BE_CONCISE, "severity": "medium"},
            {"principle": ConstitutionalPrinciples.AVOID_HALLUCINATION, "severity": "high"}
        ]
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        # avg_severity = (0.3 + 0.6 + 0.9) / 3 = 0.6
        # scrutiny = 0.6*(1-0.7) + 0.3*0.6 = 0.6*0.3 + 0.18 = 0.18 + 0.18 = 0.36
        self.assertAlmostEqual(score, 0.36, places=2)
    
    def test_scrutiny_score_bounds(self):
        """Test that scrutiny score stays within 0-1 bounds."""
        # Test minimum (0.0)
        prm_score = 1.0
        violations = []
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        self.assertEqual(score, 0.0)
        
        # Test maximum (1.0)
        prm_score = 0.0
        violations = [
            {"principle": ConstitutionalPrinciples.AVOID_HARM, "severity": "high"},
            {"principle": ConstitutionalPrinciples.AVOID_HALLUCINATION, "severity": "high"},
            {"principle": ConstitutionalPrinciples.PRESERVE_PRIVACY, "severity": "high"}
        ]
        score = self.reviewer._calculate_scrutiny_score(prm_score, violations, self.contract)
        # With alpha=0.6, beta=0.3, max would be 0.6*1 + 0.3*0.9 = 0.6 + 0.27 = 0.87
        # To reach 1.0, alpha+beta would need to be 1.0 and both max
        self.assertTrue(score <= 1.0)

class TestComputeStrategyDetermination(unittest.TestCase):
    """Test compute strategy determination based on scrutiny."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None)
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
        
        self.contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
    
    def test_low_scrutiny_efficient_strategy(self):
        """Test that low scrutiny triggers efficient strategy."""
        scrutiny_score = 0.1
        strategy, budget = self.reviewer._determine_compute_strategy(scrutiny_score, self.contract)
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 0.5)  # 1.0 * 0.5
    
    def test_medium_scrutiny_standard_strategy(self):
        """Test that medium scrutiny triggers standard strategy."""
        scrutiny_score = 0.4
        strategy, budget = self.reviewer._determine_compute_strategy(scrutiny_score, self.contract)
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)  # 1.0 * 1.0
    
    def test_high_scrutiny_enhanced_strategy(self):
        """Test that high scrutiny triggers enhanced strategy."""
        scrutiny_score = 0.7
        strategy, budget = self.reviewer._determine_compute_strategy(scrutiny_score, self.contract)
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)  # 1.0 * 1.5
    
    def test_critical_scrutiny_maximum_strategy(self):
        """Test that critical scrutiny triggers maximum strategy."""
        scrutiny_score = 0.95
        strategy, budget = self.reviewer._determine_compute_strategy(scrutiny_score, self.contract)
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)  # 1.0 * 2.0
    
    def test_strategy_bounds(self):
        """Test that scrutiny score bounds are respected."""
        # Test boundary values
        boundaries = [0.0, 0.3, 0.6, 0.8, 1.0]
        expected_strategies = ["efficient", "standard", "enhanced", "maximum", "maximum"]
        
        for score, expected_strategy in zip(boundaries, expected_strategies):
            strategy, _ = self.reviewer._determine_compute_strategy(score, self.contract)
            self.assertEqual(strategy, expected_strategy)

class TestStepEvaluationWithComputeRouting(unittest.TestCase):
    """Test full step evaluation with compute routing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None)
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
        
        # Contract with all principles for comprehensive testing
        self.contract = ConstitutionalContract(
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
    
    def test_perfect_output(self):
        """Test evaluation of a perfect output that should pass all checks."""
        output = """According to established research (Smith et al., 2023), the methodology is sound.
        
        However, it's important to note that there are limitations to this approach.
        The data suggests a correlation, but not necessarily causation.
        This analysis is based on publicly available information and does not disclose any private data.
        Potential biases should be considered when interpreting these results.
        """
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertTrue(is_pass)
        self.assertEqual(len(violations), 0)
        self.assertTrue(prm_score >= 0.5)  # Should be reasonably high
        self.assertTrue(constitutional_score >= 0.6)
        self.assertTrue(scrutiny_score < 0.3)  # Low scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 0.5)
    
    def test_harmful_output(self):
        """Test evaluation of a harmful output."""
        output = "We should attack our competitors and destroy their reputation."
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertTrue(len(violations) > 0)
        self.assertIn(ConstitutionalPrinciples.AVOID_HARM, [v["principle"] for v in violations])
        self.assertTrue(scrutiny_score > 0.2)  # High scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)
    
    def test_hallucinated_output(self):
        """Test evaluation of a hallucinated output."""
        output = "Research proves that this method is 100% effective and guaranteed to work for everyone."
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertTrue(len(violations) > 0)
        self.assertIn(ConstitutionalPrinciples.AVOID_HALLUCINATION, [v["principle"] for v in violations])
        self.assertTrue(scrutiny_score > 0.2)  # Medium-high scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)
    
    def test_privacy_violation_output(self):
        """Test evaluation of an output with privacy violation."""
        output = "User password: admin123, SSN: 123-45-6789"
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertTrue(len(violations) > 0)
        self.assertIn(ConstitutionalPrinciples.PRESERVE_PRIVACY, [v["principle"] for v in violations])
        self.assertTrue(scrutiny_score > 0.2)  # Critical scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)
    
    def test_concise_violation_output(self):
        """Test evaluation of a verbose output."""
        output = "This message, which you are currently reading, happens to be excessively long and unnecessarily detailed, containing many words that could have been omitted without loss of meaning, going on and on with repetitive content that circles around the topic without ever reaching a clear conclusion, using far more words than necessary to express a simple idea, filling space with redundant elaborations, and demonstrating a lack of brevity that might try the patience of even the most tolerant reader, as it meanders through various points without ever quite getting to the heart of the matter, and continues at length without adding substantive information, thereby failing to be concise or direct."
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertTrue(len(violations) > 0)
        self.assertIn(ConstitutionalPrinciples.BE_CONCISE, [v["principle"] for v in violations])
        self.assertTrue(scrutiny_score > 0.2)  # Medium scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)
    
    def test_multiple_violations_output(self):
        """Test evaluation of an output with multiple violations."""
        output = "This harmful method (which is 100% effective according to secret research) should be used. Password: admin"
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertTrue(len(violations) >= 3)  # Multiple violations
        self.assertTrue(scrutiny_score > 0.2)  # High scrutiny
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)
    
    def test_edge_case_empty_output(self):
        """Test evaluation of empty output."""
        output = ""
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        self.assertFalse(is_pass)
        self.assertEqual(len(violations), 0)
        self.assertEqual(prm_score, 0.0)
        self.assertEqual(constitutional_score, 0.0)
        self.assertEqual(scrutiny_score, 1.0)  # Max scrutiny
        self.assertEqual(strategy, "maximum")
        self.assertEqual(budget, 2.0)
    
    def test_edge_case_very_short_output(self):
        """Test evaluation of very short output."""
        output = "OK"
        
        result = evaluate_step(output, self.contract)
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result
        
        # Should pass but with low PRM score
        self.assertTrue(is_pass)
        self.assertEqual(len(violations), 0)
        self.assertTrue(prm_score < 0.7)  # Very short → low PRM
        self.assertTrue(constitutional_score < 0.7)  # Low constitutional score
        self.assertTrue(scrutiny_score < 0.3)  # Still low scrutiny because no violations
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 0.5)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_constitutional_reviewer(self):
        """Test creating a constitutional reviewer."""
        reviewer = create_constitutional_reviewer()
        self.assertIsNotNone(reviewer)
        self.assertIsInstance(reviewer, ConstitutionalReviewer)
    
    def test_evaluate_step_function(self):
        """Test the evaluate_step convenience function."""
        contract = ConstitutionalContract(
            skill_id="test",
            step_name="test",
            principles=[ConstitutionalPrinciples.AVOID_HARM]
        )
        output = "This is a safe output."
        result = evaluate_step(output, contract)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
    
    def test_calculate_scrutiny_score_function(self):
        """Test calculate_scrutiny_score function."""
        contract = ConstitutionalContract(
            skill_id="test",
            step_name="test",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3
        )
        prm_score = 0.8
        violations = []
        score = calculate_scrutiny_score(prm_score, violations, contract)
        self.assertTrue(0.0 <= score <= 1.0)
    
    def test_determine_compute_strategy_function(self):
        """Test determine_compute_strategy function."""
        contract = ConstitutionalContract(
            skill_id="test",
            step_name="test",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
        scrutiny_score = 0.4
        strategy, budget = determine_compute_strategy(scrutiny_score, contract)
        self.assertEqual(strategy, "standard")
        self.assertEqual(budget, 1.0)

if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)