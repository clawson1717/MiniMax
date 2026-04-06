"""
Task Decomposer for VERITAS

Responsible for decomposing tasks into skill chains and assessing constitutional risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import random
import time
from abc import ABC, abstractmethod

from .logger import Logger, LogLevel
from .core import ConstitutionalContract, ConstitutionalPrinciples, ConstitutionalReviewer


@dataclass
class SkillStep:
    """Represents a single step in a skill chain."""
    skill_id: str
    step_name: str
    contract: ConstitutionalContract
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    risk_score: float = 0.0


@dataclass
class SkillChain:
    """Represents a chain of skill steps for task execution."""
    task_id: str
    steps: List[SkillStep] = field(default_factory=list)
    total_risk_score: float = 0.0
    total_estimated_cost: float = 0.0


class TaskDecomposer:
    """Decomposes tasks into skill chains with constitutional risk assessment."""
    
    def __init__(
        self,
        logger: Optional[Logger] = None,
        default_contracts: Optional[Dict[str, ConstitutionalContract]] = None
    ):
        """
        Initialize the TaskDecomposer.
        
        Args:
            logger: Optional logger instance
            default_contracts: Pre-defined contracts for common skills
        """
        self.logger = logger or Logger(min_level="INFO")
        self.default_contracts = default_contracts or {}
        self.skill_registry: Dict[str, Callable] = {}
        self.risk_assessor = RiskAssessor(logger=self.logger)
        
        self.logger.info("TaskDecomposer initialized", module="task_decomposer")
    
    def register_skill(
        self,
        skill_id: str,
        skill_function: Callable,
        default_contract: Optional[ConstitutionalContract] = None
    ) -> None:
        """
        Register a skill with the decomposer.
        
        Args:
            skill_id: Unique identifier for the skill
            skill_function: The function implementing the skill
            default_contract: Default constitutional contract for this skill
        """
        self.skill_registry[skill_id] = skill_function
        
        if default_contract:
            self.default_contracts[skill_id] = default_contract
        
        self.logger.debug(f"Skill registered: {skill_id}", 
                        module="task_decomposer", 
                        skill_id=skill_id)
    
    def decompose_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[SkillChain]:
        """
        Decompose a task into a skill chain.
        
        Args:
            task_description: Description of the task to decompose
            context: Additional context for decomposition
            
        Returns:
            SkillChain representing the decomposed task, or None if decomposition fails
        """
        if not context:
            context = {}
        
        self.logger.info(f"Decomposing task: {task_description}", 
                       module="task_decomposer", 
                       task_description=task_description)
        
        # Simple heuristic-based decomposition (to be enhanced with LLM)
        # This is a placeholder - in reality, this would use an LLM or rule-based planner
        skill_chain = self._heuristic_decomposition(task_description, context)
        
        if not skill_chain:
            self.logger.error(f"Failed to decompose task: {task_description}", 
                            module="task_decomposer")
            return None
        
        # Assess constitutional risk for the skill chain
        skill_chain = self.risk_assessor.assess_chain_risk(skill_chain, context)
        
        self.logger.info(f"Task decomposed successfully: {task_description}", 
                       module="task_decomposer", 
                       steps=len(skill_chain.steps))
        
        return skill_chain
    
    def _heuristic_decomposition(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> SkillChain:
        """
        Simple heuristic-based task decomposition.
        
        Args:
            task_description: Description of the task
            context: Context for decomposition
            
        Returns:
            SkillChain with initial skill steps
        """
        # This is a basic implementation - in practice, this would use an LLM
        # For now, we'll use simple keyword matching
        
        task_lower = task_description.lower()
        steps = []
        
        # Example decomposition logic
        if "research" in task_lower or "search" in task_lower:
            # Add web search step
            contract = self._get_contract("web_search", task_description)
            steps.append(SkillStep(
                skill_id="web_search",
                step_name="fetch_information",
                contract=contract,
                priority=1
            ))
        
        if "summarize" in task_lower:
            # Add summarization step
            contract = self._get_contract("summarize", task_description)
            steps.append(SkillStep(
                skill_id="summarize",
                step_name="create_summary",
                contract=contract,
                priority=2
            ))
        
        if "analyze" in task_lower or "evaluate" in task_lower:
            # Add analysis step
            contract = self._get_contract("analyze", task_description)
            steps.append(SkillStep(
                skill_id="analyze",
                step_name="perform_analysis",
                contract=contract,
                priority=3
            ))
        
        if "write" in task_lower or "draft" in task_lower:
            # Add writing step
            contract = self._get_contract("write", task_description)
            steps.append(SkillStep(
                skill_id="write",
                step_name="create_document",
                contract=contract,
                priority=4
            ))
        
        # If no steps identified, add a generic step
        if not steps:
            contract = self._get_contract("generic", task_description)
            steps.append(SkillStep(
                skill_id="generic",
                step_name="perform_task",
                contract=contract,
                priority=1
            ))
        
        # Create skill chain
        skill_chain = SkillChain(task_id=task_description.replace(" ", "_")[:50])
        skill_chain.steps = steps
        
        return skill_chain
    
    def _get_contract(
        self,
        skill_id: str,
        task_description: str
    ) -> ConstitutionalContract:
        """
        Get constitutional contract for a skill.
        
        Args:
            skill_id: Skill identifier
            task_description: Task description for context
            
        Returns:
            ConstitutionalContract for the skill
        """
        if skill_id in self.default_contracts:
            return self.default_contracts[skill_id]
        
        # Create a default contract based on skill type
        if skill_id in ["web_search", "fetch"]:
            principles = [
                ConstitutionalPrinciples.AVOID_HALLUCINATION,
                ConstitutionalPrinciples.CITE_SOURCES,
                ConstitutionalPrinciples.PRESERVE_PRIVACY
            ]
            scrutiny_alpha = 0.7  # PRM weight
            scrutiny_beta = 0.3   # Violation weight
        elif skill_id in ["summarize", "write", "draft"]:
            principles = [
                ConstitutionalPrinciples.AVOID_HALLUCINATION,
                ConstitutionalPrinciples.BE_CONCISE,
                ConstitutionalPrinciples.BE_ACCURATE
            ]
            scrutiny_alpha = 0.6
            scrutiny_beta = 0.4
        elif skill_id in ["analyze", "evaluate"]:
            principles = [
                ConstitutionalPrinciples.BE_ACCURATE,
                ConstitutionalPrinciples.BE_FAIR,
                ConstitutionalPrinciples.BE_TRANSPARENT
            ]
            scrutiny_alpha = 0.5
            scrutiny_beta = 0.5
        else:
            # Generic contract
            principles = [
                ConstitutionalPrinciples.AVOID_HARM,
                ConstitutionalPrinciples.AVOID_HALLUCINATION,
                ConstitutionalPrinciples.PRESERVE_PRIVACY
            ]
            scrutiny_alpha = 0.6
            scrutiny_beta = 0.4
        
        return ConstitutionalContract(
            skill_id=skill_id,
            step_name=f"{skill_id}_step",
            principles=principles,
            prm_threshold=0.5,
            constitutional_threshold=0.7,
            max_regenerations=2,
            compute_budget=10.0,
            escalation_model="continue",
            scrutiny_alpha=scrutiny_alpha,
            scrutiny_beta=scrutiny_beta,
            base_compute_budget=1.0
        )


class RiskAssessor:
    """Assesses constitutional risk for skill chains."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """Initialize the RiskAssessor."""
        self.logger = logger or Logger(min_level="INFO")
        
        # Risk multipliers for different severity levels
        self.severity_multipliers = {
            "low": 1.0,
            "medium": 2.0,
            "high": 5.0
        }
        
        # Base risk scores for different principle categories
        self.principle_risk_bases = {
            ConstitutionalPrinciples.AVOID_HARM: 0.9,      # High risk
            ConstitutionalPrinciples.PRESERVE_PRIVACY: 0.8, # High risk
            ConstitutionalPrinciples.AVOID_HALLUCINATION: 0.7, # Medium-high risk
            ConstitutionalPrinciples.CITE_SOURCES: 0.5,     # Medium risk
            ConstitutionalPrinciples.BE_ACCURATE: 0.6,      # Medium risk
            ConstitutionalPrinciples.BE_FAIR: 0.6,          # Medium risk
            ConstitutionalPrinciples.BE_CONCISE: 0.3,       # Low risk
            ConstitutionalPrinciples.BE_TRANSPARENT: 0.4    # Low risk
        }
    
    def assess_chain_risk(
        self,
        skill_chain: SkillChain,
        context: Optional[Dict[str, Any]] = None
    ) -> SkillChain:
        """
        Assess constitutional risk for an entire skill chain.
        
        Args:
            skill_chain: SkillChain to assess
            context: Additional context for risk assessment
            
        Returns:
            SkillChain with updated risk scores and compute budgets
        """
        if not context:
            context = {}
        
        total_risk_score = 0.0
        total_estimated_cost = 0.0
        
        self.logger.debug(f"Assessing risk for skill chain: {skill_chain.task_id}", 
                        module="risk_assessor")
        
        # Assess each step in the chain
        for step in skill_chain.steps:
            step_risk_score = self._assess_step_risk(step, context)
            step.estimated_cost = self._estimate_step_cost(step_risk_score)
            step.risk_score = step_risk_score
            
            total_risk_score += step_risk_score
            total_estimated_cost += step.estimated_cost
        
        # Update chain totals
        skill_chain.total_risk_score = total_risk_score
        skill_chain.total_estimated_cost = total_estimated_cost
        
        # Adjust compute budgets based on total risk
        skill_chain = self._adjust_compute_budgets(skill_chain)
        
        self.logger.info(
            f"Risk assessment complete: total_risk={total_risk_score:.2f}, "
            f"total_cost={total_estimated_cost:.2f}",
            module="risk_assessor",
            task_id=skill_chain.task_id
        )
        
        return skill_chain
    
    def _assess_step_risk(
        self,
        step: SkillStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Assess constitutional risk for a single skill step.
        
        Args:
            step: SkillStep to assess
            context: Additional context for assessment
            
        Returns:
            Risk score between 0 and 1
        """
        # Start with base risk based on principles
        base_risk = 0.0
        for principle in step.contract.principles:
            if principle in self.principle_risk_bases:
                base_risk += self.principle_risk_bases[principle]
        
        # Normalize base risk (max 1.0 if all high-risk principles)
        base_risk = base_risk / len(self.principle_risk_bases)
        
        # Adjust based on step priority and dependencies
        priority_factor = 1.0 + (step.priority * 0.1)
        dependency_factor = self._calculate_dependency_factor(step, context)
        
        # Combine factors
        risk_score = base_risk * priority_factor * dependency_factor
        
        # Add randomness for exploration (to be replaced with proper uncertainty estimation)
        risk_score += random.uniform(0, 0.1)
        
        # Clamp to 0-1 range
        risk_score = max(0.0, min(1.0, risk_score))
        
        self.logger.debug(
            f"Step risk assessment: {step.step_name} -> risk={risk_score:.2f}",
            module="risk_assessor",
            step_name=step.step_name,
            risk_score=risk_score
        )
        
        return risk_score
    
    def _calculate_dependency_factor(self, step: SkillStep, context: Dict[str, Any]) -> float:
        """
        Calculate risk factor based on dependencies.
        
        Args:
            step: SkillStep to assess
            context: Additional context
            
        Returns:
            Dependency factor (0.8-1.2)
        """
        if not step.dependencies:
            return 1.0
        
        # If there are dependencies, risk is slightly higher due to propagation
        return 1.1
    
    def _estimate_step_cost(self, risk_score: float) -> float:
        """
        Estimate compute cost for a step based on its risk score.
        
        Args:
            risk_score: Risk score (0-1)
            
        Returns:
            Estimated compute cost in arbitrary units
        """
        # Higher risk → higher estimated cost
        base_cost = 1.0
        risk_multiplier = 1.0 + (risk_score * 2.0)  # Risk score 0→1 becomes 1→3 multiplier
        
        return base_cost * risk_multiplier
    
    def _adjust_compute_budgets(self, skill_chain: SkillChain) -> SkillChain:
        """
        Adjust compute budgets based on total risk score.
        
        Args:
            skill_chain: SkillChain to adjust
            
        Returns:
            SkillChain with adjusted compute budgets
        """
        total_risk = skill_chain.total_risk_score
        
        # Adjust based on total risk
        if total_risk < 0.3:
            # Low risk - reduce budgets slightly
            adjustment_factor = 0.7
        elif total_risk < 0.6:
            # Medium risk - keep as is
            adjustment_factor = 1.0
        elif total_risk < 0.8:
            # High risk - increase budgets
            adjustment_factor = 1.5
        else:
            # Critical risk - significantly increase budgets
            adjustment_factor = 2.0
        
        # Apply adjustment to each step's base compute budget
        for step in skill_chain.steps:
            step.contract.base_compute_budget *= adjustment_factor
        
        return skill_chain