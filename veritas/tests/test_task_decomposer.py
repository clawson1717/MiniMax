#!/usr/bin/env python3
"""
Test Suite for TaskDecomposer

Comprehensive tests for the VERITAS TaskDecomposer component.
"""

import unittest
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas.core import ConstitutionalPrinciples, ConstitutionalContract
from veritas.task_decomposer import TaskDecomposer, SkillStep, SkillChain, RiskAssessor


class TestSkillStep(unittest.TestCase):
    """Test SkillStep dataclass functionality."""
    
    def test_skill_step_creation(self):
        """Test that SkillStep can be created."""
        step = SkillStep(
            skill_id="test_skill",
            step_name="test_step",
            contract=ConstitutionalContract(
                skill_id="test_skill",
                step_name="test_step",
                principles=[ConstitutionalPrinciples.AVOID_HARM]
            )
        )
        self.assertIsNotNone(step)
        self.assertEqual(step.skill_id, "test_skill")
        self.assertEqual(step.step_name, "test_step")
        self.assertEqual(step.priority, 0)
        self.assertEqual(step.dependencies, [])
        self.assertEqual(step.estimated_cost, 0.0)
        self.assertEqual(step.risk_score, 0.0)


class TestSkillChain(unittest.TestCase):
    """Test SkillChain dataclass functionality."""
    
    def test_skill_chain_creation(self):
        """Test that SkillChain can be created."""
        chain = SkillChain(task_id="test_task")
        self.assertIsNotNone(chain)
        self.assertEqual(chain.task_id, "test_task")
        self.assertEqual(chain.steps, [])
        self.assertEqual(chain.total_risk_score, 0.0)
        self.assertEqual(chain.total_estimated_cost, 0.0)


class TestRiskAssessor(unittest.TestCase):
    """Test RiskAssessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = RiskAssessor()
        
        # Create a base contract
        self.base_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM]
        )
    
    def test_risk_assessor_creation(self):
        """Test that RiskAssessor can be created."""
        self.assertIsNotNone(self.assessor)
        self.assertEqual(len(self.assessor.principle_risk_bases), 8)
        self.assertIn(ConstitutionalPrinciples.AVOID_HARM, self.assessor.principle_risk_bases)
    
    def test_assess_chain_risk_empty_chain(self):
        """Test assessing risk for an empty skill chain."""
        chain = SkillChain(task_id="empty_chain")
        result = self.assessor.assess_chain_risk(chain)
        self.assertIsNotNone(result)
        self.assertEqual(result.total_risk_score, 0.0)
        self.assertEqual(result.total_estimated_cost, 0.0)
    
    def test_assess_chain_risk_single_step(self):
        """Test assessing risk for a chain with a single step."""
        step = SkillStep(
            skill_id="test_skill",
            step_name="test_step",
            contract=self.base_contract
        )
        chain = SkillChain(task_id="single_step")
        chain.steps = [step]
        
        result = self.assessor.assess_chain_risk(chain)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.steps), 1)
        self.assertTrue(0.0 <= result.total_risk_score <= 1.0)
        self.assertTrue(result.total_estimated_cost > 0.0)
        self.assertTrue(0.0 <= result.steps[0].risk_score <= 1.0)
        self.assertTrue(result.steps[0].estimated_cost > 0.0)
    
    def test_assess_chain_risk_multiple_steps(self):
        """Test assessing risk for a chain with multiple steps."""
        steps = []
        for i in range(3):
            contract = ConstitutionalContract(
                skill_id=f"test_skill_{i}",
                step_name=f"test_step_{i}",
                principles=[ConstitutionalPrinciples.AVOID_HARM]
            )
            step = SkillStep(
                skill_id=f"test_skill_{i}",
                step_name=f"test_step_{i}",
                contract=contract
            )
            steps.append(step)
        
        chain = SkillChain(task_id="multi_step")
        chain.steps = steps
        
        result = self.assessor.assess_chain_risk(chain)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.steps), 3)
        self.assertTrue(0.0 <= result.total_risk_score <= 1.0)
        self.assertTrue(result.total_estimated_cost > 0.0)
        
        # Check that each step has risk and cost
        for step in result.steps:
            self.assertTrue(0.0 <= step.risk_score <= 1.0)
            self.assertTrue(step.estimated_cost > 0.0)
    
    def test_assess_step_risk_base_values(self):
        """Test that base risk values are correctly calculated."""
        # Test with different principle combinations
        contracts = [
            ConstitutionalContract(
                skill_id="privacy_skill",
                step_name="privacy_step",
                principles=[ConstitutionalPrinciples.PRESERVE_PRIVACY]
            ),
            ConstitutionalContract(
                skill_id="harm_skill",
                step_name="harm_step",
                principles=[ConstitutionalPrinciples.AVOID_HARM]
            ),
            ConstitutionalContract(
                skill_id="concise_skill",
                step_name="concise_step",
                principles=[ConstitutionalPrinciples.BE_CONCISE]
            )
        ]
        
        for i, contract in enumerate(contracts):
            step = SkillStep(
                skill_id=f"skill_{i}",
                step_name=f"step_{i}",
                contract=contract
            )
            risk_score = self.assessor._assess_step_risk(step, {})
            self.assertTrue(0.0 <= risk_score <= 1.0)
    
    def test_calculate_dependency_factor(self):
        """Test dependency factor calculation."""
        step_no_deps = SkillStep(
            skill_id="skill_no_deps",
            step_name="step_no_deps",
            contract=self.base_contract
        )
        step_with_deps = SkillStep(
            skill_id="skill_with_deps",
            step_name="step_with_deps",
            contract=self.base_contract,
            dependencies=["step1", "step2"]
        )
        
        factor_no_deps = self.assessor._calculate_dependency_factor(step_no_deps, {})
        factor_with_deps = self.assessor._calculate_dependency_factor(step_with_deps, {})
        
        self.assertEqual(factor_no_deps, 1.0)
        self.assertEqual(factor_with_deps, 1.1)
    
    def test_estimate_step_cost(self):
        """Test step cost estimation."""
        for risk_score in [0.0, 0.3, 0.5, 0.7, 1.0]:
            cost = self.assessor._estimate_step_cost(risk_score)
            self.assertTrue(cost > 0.0)
            # Cost should increase with risk score
            if risk_score > 0:
                self.assertTrue(cost >= 1.0)
    
    def test_adjust_compute_budgets(self):
        """Test compute budget adjustment based on total risk."""
        chain = SkillChain(task_id="test_adjust")
        chain.steps = [
            SkillStep(
                skill_id="skill1",
                step_name="step1",
                contract=ConstitutionalContract(
                    skill_id="skill1",
                    step_name="step1",
                    principles=[ConstitutionalPrinciples.AVOID_HARM],
                    base_compute_budget=1.0
                )
            )
        ]
        
        # Test different total risk scores
        test_risks = [0.1, 0.4, 0.7, 0.9]
        expected_factors = [0.7, 1.0, 1.5, 2.0]
        
        for risk, expected_factor in zip(test_risks, expected_factors):
            chain.total_risk_score = risk
            adjusted_chain = self.assessor._adjust_compute_budgets(chain)
            original_budget = chain.steps[0].contract.base_compute_budget
            adjusted_budget = adjusted_chain.steps[0].contract.base_compute_budget
            self.assertAlmostEqual(adjusted_budget, original_budget * expected_factor, places=1)


class TestTaskDecomposer(unittest.TestCase):
    """Test TaskDecomposer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.decomposer = TaskDecomposer()
        
        # Register some skills
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
                scrutiny_alpha=0.7,
                scrutiny_beta=0.3
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
                scrutiny_alpha=0.6,
                scrutiny_beta=0.4
            )
        )
    
    def test_task_decomposer_creation(self):
        """Test that TaskDecomposer can be created."""
        self.assertIsNotNone(self.decomposer)
        self.assertEqual(len(self.decomposer.skill_registry), 2)
        self.assertEqual(len(self.decomposer.default_contracts), 2)
    
    def test_register_skill(self):
        """Test skill registration."""
        initial_count = len(self.decomposer.skill_registry)
        self.decomposer.register_skill(
            skill_id="analyze",
            skill_function=lambda x: "analysis",
            default_contract=ConstitutionalContract(
                skill_id="analyze",
                step_name="perform_analysis",
                principles=[ConstitutionalPrinciples.BE_ACCURATE],
                scrutiny_alpha=0.5,
                scrutiny_beta=0.5
            )
        )
        self.assertEqual(len(self.decomposer.skill_registry), initial_count + 1)
        self.assertIn("analyze", self.decomposer.skill_registry)
    
    def test_decompose_task_with_keywords(self):
        """Test task decomposition using keyword matching."""
        task = "Research and summarize the latest climate change findings"
        result = self.decomposer.decompose_task(task)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SkillChain)
        self.assertEqual(result.task_id, "research_and_summarize_the_latest_climate_change_findings")
        self.assertTrue(len(result.steps) >= 2)
        
        # Check that expected skills are in the chain
        skill_ids = [step.skill_id for step in result.steps]
        self.assertIn("web_search", skill_ids)
        self.assertIn("summarize", skill_ids)
    
    def test_decompose_task_without_keywords(self):
        """Test task decomposition for a task without clear keywords."""
        task = "Complete this assignment"
        result = self.decomposer.decompose_task(task)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SkillChain)
        self.assertTrue(len(result.steps) >= 1)
        
        # Should have at least one step, likely a generic one
        skill_ids = [step.skill_id for step in result.steps]
        self.assertIn("generic", skill_ids)
    
    def test_decompose_task_with_context(self):
        """Test task decomposition with additional context."""
        task = "Analyze the economic impact of COVID-19"
        context = {"domain": "economics", "format": "report"}
        result = self.decomposer.decompose_task(task, context)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SkillChain)
        self.assertTrue(len(result.steps) >= 2)
        
        # Check that steps have appropriate contracts
        for step in result.steps:
            self.assertIsNotNone(step.contract)
            self.assertTrue(0.0 <= step.contract.scrutiny_alpha <= 1.0)
            self.assertTrue(0.0 <= step.contract.scrutiny_beta <= 1.0)
    
    def test_heuristic_decomposition(self):
        """Test the internal heuristic decomposition method."""
        task = "Write a report about renewable energy sources"
        result = self.decomposer._heuristic_decomposition(task, {})
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SkillChain)
        self.assertTrue(len(result.steps) >= 1)
        
        # Check for expected skills
        skill_ids = [step.skill_id for step in result.steps]
        self.assertIn("web_search", skill_ids)
        self.assertIn("write", skill_ids)
    
    def test_get_contract(self):
        """Test contract generation for different skill types."""
        # Test with registered skills
        contract1 = self.decomposer._get_contract("web_search", "test task")
        self.assertIsNotNone(contract1)
        self.assertEqual(contract1.skill_id, "web_search")
        self.assertEqual(contract1.principles, [
            ConstitutionalPrinciples.AVOID_HALLUCINATION,
            ConstitutionalPrinciples.CITE_SOURCES,
            ConstitutionalPrinciples.PRESERVE_PRIVACY
        ])
        
        # Test with unregistered skill (should use defaults)
        contract2 = self.decomposer._get_contract("analyze", "test task")
        self.assertIsNotNone(contract2)
        self.assertEqual(contract2.skill_id, "analyze")
        # Default principles for analyze should be accuracy, fairness, transparency
        self.assertEqual(contract2.principles, [
            ConstitutionalPrinciples.BE_ACCURATE,
            ConstitutionalPrinciples.BE_FAIR,
            ConstitutionalPrinciples.BE_TRANSPARENT
        ])


class TestIntegrationWithCore(unittest.TestCase):
    """Test integration between TaskDecomposer and core components."""
    
    def test_full_decomposition_pipeline(self):
        """Test the full decomposition pipeline from task to risk assessment."""
        decomposer = TaskDecomposer()
        
        # Register a skill with a specific contract
        test_contract = ConstitutionalContract(
            skill_id="test_skill",
            step_name="test_step",
            principles=[ConstitutionalPrinciples.AVOID_HARM],
            scrutiny_alpha=0.6,
            scrutiny_beta=0.4
        )
        
        decomposer.register_skill(
            skill_id="test_skill",
            skill_function=lambda x: "test output",
            default_contract=test_contract
        )
        
        # Decompose a task that should use this skill
        task = "Test this skill"
        result = decomposer.decompose_task(task)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SkillChain)
        self.assertTrue(len(result.steps) >= 1)
        
        # Check that the step has the correct contract
        step = result.steps[0]
        self.assertEqual(step.contract.skill_id, "test_skill")
        self.assertEqual(step.contract.scrutiny_alpha, 0.6)
        self.assertEqual(step.contract.scrutiny_beta, 0.4)
        self.assertTrue(0.0 <= step.risk_score <= 1.0)
        self.assertTrue(step.estimated_cost > 0.0)
        
        # Check that total risk and cost are calculated
        self.assertTrue(0.0 <= result.total_risk_score <= 1.0)
        self.assertTrue(result.total_estimated_cost > 0.0)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Run tests
    unittest.main(verbosity=2)