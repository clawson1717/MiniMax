#!/usr/bin/env python3
"""
Comprehensive Integration Test

Tests the complete VERITAS workflow from task decomposition to compute routing.
"""

import unittest
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.core import ConstitutionalPrinciples, ConstitutionalContract, ConstitutionalReviewer, ComputeRouter, evaluate_step
from veritas.task_decomposer import TaskDecomposer, SkillStep, SkillChain, RiskAssessor
from veritas.logger import Logger
from veritas.prm import PRMConfig


class TestVERITASIntegration(unittest.TestCase):
    """Test the complete VERITAS workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(console_output=False, log_file=None, min_level="INFO")
        
        # Create a TaskDecomposer with some registered skills
        self.decomposer = TaskDecomposer(logger=self.logger)
        
        # Register skills with default contracts
        self.decomposer.register_skill(
            skill_id="web_search",
            skill_function=lambda x: "search results",
            default_contract=ConstitutionalContract(
                skill_id="web_search",
                step_name="fetch_information",
                principles=[
                    ConstitutionalPrinciples.AVOID_HALLUCINATION,
                    ConstitutionalPrinciples.CITE_SOURCES,
                    ConstitutionalPrinciples.PRESERVE_PRIVACY
                ],
                prm_threshold=0.6,
                constitutional_threshold=0.7,
                scrutiny_alpha=0.7,
                scrutiny_beta=0.3,
                base_compute_budget=1.0
            )
        )
        
        self.decomposer.register_skill(
            skill_id="summarize",
            skill_function=lambda x: "summary",
            default_contract=ConstitutionalContract(
                skill_id="summarize",
                step_name="create_summary",
                principles=[
                    ConstitutionalPrinciples.AVOID_HALLUCINATION,
                    ConstitutionalPrinciples.BE_CONCISE,
                    ConstitutionalPrinciples.BE_ACCURATE
                ],
                prm_threshold=0.5,
                constitutional_threshold=0.7,
                scrutiny_alpha=0.6,
                scrutiny_beta=0.4,
                base_compute_budget=1.0
            )
        )
        
        self.decomposer.register_skill(
            skill_id="analyze",
            skill_function=lambda x: "analysis",
            default_contract=ConstitutionalContract(
                skill_id="analyze",
                step_name="perform_analysis",
                principles=[
                    ConstitutionalPrinciples.BE_ACCURATE,
                    ConstitutionalPrinciples.BE_FAIR,
                    ConstitutionalPrinciples.BE_TRANSPARENT
                ],
                prm_threshold=0.5,
                constitutional_threshold=0.7,
                scrutiny_alpha=0.5,
                scrutiny_beta=0.5,
                base_compute_budget=1.0
            )
        )
        
        # Create a ComputeRouter
        self.router = ComputeRouter(logger=self.logger)
        
        # Create a ConstitutionalReviewer
        self.reviewer = ConstitutionalReviewer(logger=self.logger)
    
    def test_complete_workflow_low_risk_task(self):
        """Test the complete workflow for a low-risk task."""
        task = "Research and summarize the latest climate change findings"
        
        # Step 1: Decompose the task
        skill_chain = self.decomposer.decompose_task(task)
        self.assertIsNotNone(skill_chain)
        self.assertTrue(len(skill_chain.steps) >= 2)
        
        # Step 2: Verify risk assessment
        self.assertTrue(0.0 <= skill_chain.total_risk_score <= 1.0)
        self.assertTrue(skill_chain.total_estimated_cost > 0.0)
        
        # For a low-risk task, total risk should be < 0.3
        self.assertLess(skill_chain.total_risk_score, 0.3, 
                       f"Expected low risk (<0.3), got {skill_chain.total_risk_score}")
        
        # Step 3: Check that compute budgets were adjusted
        for step in skill_chain.steps:
            self.assertTrue(step.contract.base_compute_budget <= 0.5,  # Should be reduced
                           f"Expected reduced budget (<=0.5), got {step.contract.base_compute_budget}")
        
        # Step 4: Simulate step execution and routing
        # We'll use the first step as an example
        first_step = skill_chain.steps[0]
        output = f"Simulated output for {first_step.skill_id}"
        
        # Get routing decision
        routing_result = self.router.route(output, first_step.contract)
        self.assertIsNotNone(routing_result)
        
        # For low-risk tasks, should use efficient strategy
        self.assertEqual(routing_result["compute_strategy"], "efficient")
        self.assertEqual(routing_result["compute_budget"], 0.5)
        self.assertEqual(routing_result["routing_decision"]["action"], "regenerate")
        self.assertEqual(routing_result["routing_decision"]["iterations"], 1)
        self.assertFalse(routing_result["routing_decision"]["enhanced_verification"])
        
        # Step 5: Verify constitutional review
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = \
            evaluate_step(output, first_step.contract, logger=self.logger)
        
        self.assertTrue(is_pass)
        self.assertEqual(len(violations), 0)
        self.assertTrue(0.0 <= prm_score <= 1.0)
        self.assertTrue(0.0 <= constitutional_score <= 1.0)
        self.assertTrue(0.0 <= scrutiny_score <= 1.0)
        self.assertEqual(strategy, "efficient")
        self.assertEqual(budget, 0.5)
    
    def test_complete_workflow_high_risk_task(self):
        """Test the complete workflow for a high-risk task."""
        task = "Write a persuasive essay arguing for harmful policies"
        
        # Step 1: Decompose the task
        skill_chain = self.decomposer.decompose_task(task)
        self.assertIsNotNone(skill_chain)
        self.assertTrue(len(skill_chain.steps) >= 1)
        
        # Step 2: Verify risk assessment
        self.assertTrue(0.0 <= skill_chain.total_risk_score <= 1.0)
        self.assertTrue(skill_chain.total_estimated_cost > 0.0)
        
        # For a high-risk task, total risk should be > 0.6
        self.assertGreater(skill_chain.total_risk_score, 0.6,
                           f"Expected high risk (>0.6), got {skill_chain.total_risk_score}")
        
        # Step 3: Check that compute budgets were adjusted
        for step in skill_chain.steps:
            self.assertTrue(step.contract.base_compute_budget >= 1.5,  # Should be increased
                           f"Expected increased budget (>=1.5), got {step.contract.base_compute_budget}")
        
        # Step 4: Simulate step execution and routing
        first_step = skill_chain.steps[0]
        output = f"Simulated output for {first_step.skill_id}"
        
        # Get routing decision
        routing_result = self.router.route(output, first_step.contract)
        self.assertIsNotNone(routing_result)
        
        # For high-risk tasks, should use enhanced or maximum strategy
        self.assertIn(routing_result["compute_strategy"], ["enhanced", "maximum"])
        if routing_result["compute_strategy"] == "enhanced":
            self.assertEqual(routing_result["compute_budget"], 1.5)
        else:
            self.assertEqual(routing_result["compute_budget"], 2.0)
        
        # Step 5: Verify constitutional review
        is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = \
            evaluate_step(output, first_step.contract, logger=self.logger)
        
        # High-risk tasks may not pass initially
        self.assertTrue(0.0 <= scrutiny_score <= 1.0)
        self.assertTrue(0.0 <= constitutional_score <= 1.0)
    
    def test_mixed_risk_task(self):
        """Test a task with mixed risk levels."""
        task = "Analyze the economic impact of a controversial policy and write a report"
        
        # Step 1: Decompose the task
        skill_chain = self.decomposer.decompose_task(task)
        self.assertIsNotNone(skill_chain)
        self.assertTrue(len(skill_chain.steps) >= 2)
        
        # Step 2: Verify risk assessment
        self.assertTrue(0.0 <= skill_chain.total_risk_score <= 1.0)
        self.assertTrue(skill_chain.total_estimated_cost > 0.0)
        
        # Should be in the medium risk range
        self.assertGreaterEqual(skill_chain.total_risk_score, 0.3)
        self.assertLess(skill_chain.total_risk_score, 0.8)
        
        # Step 3: Check compute budget adjustment
        # Medium risk should keep standard budgets (factor=1.0)
        for step in skill_chain.steps:
            self.assertAlmostEqual(step.contract.base_compute_budget, 1.0, places=1)
        
        # Step 4: Simulate step execution
        first_step = skill_chain.steps[0]
        output = f"Simulated output for {first_step.skill_id}"
        
        routing_result = self.router.route(output, first_step.contract)
        self.assertIsNotNone(routing_result)
        
        # Medium risk should use standard strategy
        self.assertEqual(routing_result["compute_strategy"], "standard")
        self.assertEqual(routing_result["compute_budget"], 1.0)
    
    def test_edge_case_empty_task(self):
        """Test decomposition of an empty task."""
        task = ""
        
        skill_chain = self.decomposer.decompose_task(task)
        self.assertIsNotNone(skill_chain)
        self.assertTrue(len(skill_chain.steps) >= 1)
        
        # Should still have a generic step
        self.assertEqual(skill_chain.steps[0].skill_id, "generic")
    
    def test_edge_case_very_short_task(self):
        """Test decomposition of a very short task."""
        task = "Do it"
        
        skill_chain = self.decomposer.decompose_task(task)
        self.assertIsNotNone(skill_chain)
        self.assertTrue(len(skill_chain.steps) >= 1)
        
        # Should have at least one step
        self.assertTrue(len(skill_chain.steps) > 0)
    
    def test_task_with_dependencies(self):
        """Test task decomposition with dependencies."""
        # Create a custom decomposer that adds dependencies
        decomposer = TaskDecomposer(logger=self.logger)
        
        # Register skills
        decomposer.register_skill("research", lambda x: "results", 
                                 ConstitutionalContract("research", "step", [ConstitutionalPrinciples.AVOID_HALLUCINATION]))
        decomposer.register_skill("write", lambda x: "draft", 
                                 ConstitutionalContract("write", "step", [ConstitutionalPrinciples.BE_CONCISE]))
        decomposer.register_skill("review", lambda x: "review", 
                                 ConstitutionalContract("review", "step", [ConstitutionalPrinciples.BE_ACCURATE]))
        
        # Manually create a skill chain with dependencies
        chain = SkillChain("test_task")
        chain.steps = [
            SkillStep("research", "step1", ConstitutionalContract("research", "step1", [ConstitutionalPrinciples.AVOID_HALLUCINATION]), 
                     priority=1, dependencies=[]),
            SkillStep("write", "step2", ConstitutionalContract("write", "step2", [ConstitutionalPrinciples.BE_CONCISE]),
                     priority=2, dependencies=["step1"]),
            SkillStep("review", "step3", ConstitutionalContract("review", "step3", [ConstitutionalPrinciples.BE_ACCURATE]),
                     priority=3, dependencies=["step2"])
        ]
        
        # Assess risk
        assessor = RiskAssessor(logger=self.logger)
        chain = assessor.assess_chain_risk(chain)
        
        # Check that dependencies increased risk
        for step in chain.steps[1:]:
            self.assertGreater(step.risk_score, 0.3)  # Dependencies should increase risk
            self.assertGreater(step.estimated_cost, 1.0)  # Higher cost for dependent steps
    
    def test_dynamic_compute_allocation(self):
        """Test dynamic compute allocation based on risk."""
        # Create a task with both low and high risk steps
        decomposer = TaskDecomposer(logger=self.logger)
        
        # Register a low-risk skill
        decomposer.register_skill("low_risk", lambda x: "safe output", 
                                 ConstitutionalContract("low_risk", "step", [ConstitutionalPrinciples.BE_CONCISE],
                                                       scrutiny_alpha=0.3, scrutiny_beta=0.2,
                                                       base_compute_budget=1.0))
        
        # Register a high-risk skill
        decomposer.register_skill("high_risk", lambda x: "dangerous output", 
                                 ConstitutionalContract("high_risk", "step", [ConstitutionalPrinciples.AVOID_HARM],
                                                       scrutiny_alpha=0.7, scrutiny_beta=0.4,
                                                       base_compute_budget=1.0))
        
        # Create a mixed task
        task = "Do something low risk and something high risk"
        chain = decomposer.decompose_task(task)
        self.assertIsNotNone(chain)
        self.assertEqual(len(chain.steps), 2)
        
        # Risk assess
        assessor = RiskAssessor(logger=self.logger)
        chain = assessor.assess_chain_risk(chain)
        
        # Check that low-risk step got reduced budget
        low_risk_step = next(s for s in chain.steps if s.skill_id == "low_risk")
        self.assertLess(low_risk_step.contract.base_compute_budget, 1.0)
        
        # Check that high-risk step got increased budget
        high_risk_step = next(s for s in chain.steps if s.skill_id == "high_risk")
        self.assertGreater(high_risk_step.contract.base_compute_budget, 1.0)
    
    def test_constitutional_scoring_consistency(self):
        """Test that constitutional scoring is consistent across components."""
        # Create a contract
        contract = ConstitutionalContract(
            skill_id="test",
            step_name="test",
            principles=[ConstitutionalPrinciples.AVOID_HALLUCINATION, ConstitutionalPrinciples.BE_CONCISE],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.4,
            base_compute_budget=1.0
        )
        
        # Test with a good output
        good_output = "This is a well-written and factual output that is also concise."
        
        # Get PRM score from reviewer
        reviewer = ConstitutionalReviewer(logger=self.logger)
        prm_score = reviewer._get_prm_score(good_output, contract.principles)
        
        # Get violations from advanced critique
        violations = reviewer._advanced_critique(good_output, contract)
        
        # Calculate constitutional score manually
        violation_penalty = sum(0.1 if v["severity"] == "low" else 0.3 if v["severity"] == "medium" else 0.6 
                               for v in violations)
        constitutional_score = max(0.0, prm_score - violation_penalty)
        
        # Get full evaluation
        is_pass, _, eval_prm_score, eval_constitutional_score, _, _, _ = \
            evaluate_step(good_output, contract, logger=self.logger)
        
        # Check consistency
        self.assertAlmostEqual(prm_score, eval_prm_score, places=2)
        self.assertAlmostEqual(constitutional_score, eval_constitutional_score, places=2)
        self.assertTrue(is_pass)
    
    def test_compute_router_strategy_selection(self):
        """Test that compute router selects appropriate strategies."""
        router = ComputeRouter(logger=self.logger)
        
        # Create contracts with different risk profiles
        low_risk_contract = ConstitutionalContract(
            skill_id="low_risk",
            step_name="step",
            principles=[ConstitutionalPrinciples.BE_CONCISE],
            scrutiny_alpha=0.3,
            scrutiny_beta=0.2,
            base_compute_budget=1.0
        )
        
        medium_risk_contract = ConstitutionalContract(
            skill_id="medium_risk",
            step_name="step",
            principles=[ConstitutionalPrinciples.AVOID_HALLUCINATION],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.3,
            base_compute_budget=1.0
        )
        
        high_risk_contract = ConstitutionalContract(
            skill_id="high_risk",
            step_name="step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.7,
            scrutiny_beta=0.4,
            base_compute_budget=1.0
        )
        
        # Test with outputs that would produce different scrutiny scores
        low_risk_output = "This is safe and concise."
        medium_risk_output = "This is mostly accurate but could be improved."
        high_risk_output = "This is dangerous and harmful."
        
        # Get routing decisions
        low_result = router.route(low_risk_output, low_risk_contract)
        medium_result = router.route(medium_risk_output, medium_risk_contract)
        high_result = router.route(high_risk_output, high_risk_contract)
        
        # Verify strategies
        self.assertEqual(low_result["compute_strategy"], "efficient")
        self.assertEqual(medium_result["compute_strategy"], "standard")
        self.assertEqual(high_result["compute_strategy"], "enhanced")
        
        # For very harmful content, should be maximum
        very_high_risk_output = "This will cause serious harm."
        very_high_result = router.route(very_high_risk_output, high_risk_contract)
        self.assertEqual(very_high_result["compute_strategy"], "maximum")


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)