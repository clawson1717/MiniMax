from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import random
import math
import re
import time
from typing_extensions import override

from .logger import Logger, LogLevel
from .prm import PRM, PRMConfig


class ConstitutionalPrinciples(Enum):
    """Constitutional principles for AI behavior validation."""

    AVOID_HARM = "avoid_harm"
    """Avoid causing harm to users, property, or the environment."""

    AVOID_HALLUCINATION = "avoid_hallucination"
    """Avoid generating false or fabricated information."""

    CITE_SOURCES = "cite_sources"
    """Properly attribute sources and provide citations when appropriate."""

    PRESERVE_PRIVACY = "preserve_privacy"
    """Protect user privacy and confidential information."""

    BE_CONCISE = "be_concise"
    """Communicate clearly and without unnecessary verbosity."""

    BE_ACCURATE = "be_accurate"
    """Ensure information is factually correct and verifiable."""

    BE_FAIR = "be_fair"
    """Present balanced viewpoints and avoid bias."""

    BE_TRANSPARENT = "be_transparent"
    """Disclose limitations, uncertainties, and reasoning methods."""


@dataclass
class ConstitutionalContract:
    """Contract that defines constitutional requirements for a skill step."""

    skill_id: str
    """Unique identifier for the skill."""

    step_name: str
    """Name of the skill step."""

    principles: List[ConstitutionalPrinciples]
    """List of constitutional principles to enforce."""

    prm_threshold: float = 0.5
    """Threshold for PRM (Policy Reinforcement Model) confidence."""

    constitutional_threshold: float = 0.7
    """Minimum constitutional score required to pass."""

    max_regenerations: int = 3
    """Maximum number of output regenerations allowed."""

    compute_budget: Optional[float] = None
    """Optional compute budget in compute units."""

    escalation_model: str = "continue"
    """Escalation model to use when constitutional violations are detected."""

    # Advanced fields for compute routing
    scrutiny_alpha: float = 0.6
    """Weight for PRM confidence in scrutiny score (0-1)."""
    
    scrutiny_beta: float = 0.4
    """Weight for violation severity in scrutiny score (0-1)."""
    
    base_compute_budget: float = 1.0
    """Base compute budget for this step."""

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the contract parameters.

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []

        if not self.skill_id:
            errors.append("skill_id must be a non-empty string")

        if not self.step_name:
            errors.append("step_name must be a non-empty string")

        if not self.principles:
            errors.append("principles must contain at least one principle")

        if self.prm_threshold < 0 or self.prm_threshold > 1.0:
            errors.append("prm_threshold must be between 0 and 1.0")

        if self.constitutional_threshold < 0 or self.constitutional_threshold > 1.0:
            errors.append("constitutional_threshold must be between 0 and 1.0")

        if self.max_regenerations < 0:
            errors.append("max_regenerations must be non-negative")

        if self.escalation_model not in ["continue", "fail", "escalate"]:
            errors.append("escalation_model must be 'continue', 'fail', or 'escalate'")

        if not (0 <= self.scrutiny_alpha <= 1):
            errors.append("scrutiny_alpha must be between 0 and 1")
        
        if not (0 <= self.scrutiny_beta <= 1):
            errors.append("scrutiny_beta must be between 0 and 1")
        
        if self.scrutiny_alpha + self.scrutiny_beta > 1:
            errors.append("scrutiny_alpha + scrutiny_beta must not exceed 1")

        return len(errors) == 0, errors


class ConstitutionalReviewer:
    """Reviews skill outputs against constitutional principles with advanced compute routing."""

    def __init__(
        self,
        critique_model: str = "bert-base-uncased",
        critique_model_size: str = "base",
        use_prm: bool = True,
        logger: Optional[Logger] = None,
        prm_config: Optional[PRMConfig] = None
    ):
        """Initialize the ConstitutionalReviewer.

        Args:
            critique_model: Name of the critique model to use.
            critique_model_size: Size of the critique model.
            use_prm: Whether to use PRM for scoring.
            logger: Optional logger instance.
            prm_config: Optional PRM configuration.
        """
        self.critique_model = critique_model
        self.critique_model_size = critique_model_size
        self.use_prm = use_prm
        self.logger = logger or Logger()
        self.prm_config = prm_config or PRMConfig()
        self.critique_cache: Dict[str, Any] = {}
        self.prm_instances: Dict[str, PRM] = {}
        
        # Initialize PRM instance
        self._init_prm()

    def _init_prm(self) -> None:
        """Initialize or reinitialize the PRM instance."""
        try:
            prm_id = f"{self.critique_model}_{self.critique_model_size}"
            if prm_id not in self.prm_instances:
                self.prm_instances[prm_id] = PRM(
                    config=self.prm_config,
                    pretrained=True
                )
            self.prm = self.prm_instances[prm_id]
        except Exception as e:
            self.logger.warning(f"Failed to initialize PRM: {e}")
            self.prm = None

    def _get_prm_score(
        self, 
        output: str, 
        principles: List[ConstitutionalPrinciples],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get PRM score for the output against principles.

        Args:
            output: The output to evaluate.
            principles: List of principles to check.
            context: Optional additional context for evaluation.

        Returns:
            PRM confidence score between 0 and 1.
        """
        if not self.use_prm or self.prm is None:
            return 1.0

        # Create a concatenated text for PRM evaluation with context
        principle_texts = [
            f"The output should {principle.value.replace('_', ' ')}."
            for principle in principles
        ]
        
        context_text = ""
        if context:
            context_text = f"Context: {context.get('task', '')} "
        
        text_to_evaluate = f"{context_text}{output} {' '.join(principle_texts)}"

        try:
            score = self.prm.predict(text_to_evaluate, context=context)
            return max(0.0, min(1.0, score))
        except Exception as e:
            self.logger.warning(f"PRM evaluation failed: {e}")
            return 0.0

    def _advanced_critique(
        self, 
        output: str, 
        contract: ConstitutionalContract,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Advanced critique using heuristic patterns and severity assessment.

        Args:
            output: The output to evaluate.
            contract: The constitutional contract.
            context: Optional additional context.

        Returns:
            List of violation dictionaries with principle, description, and severity.
        """
        violations = []
        output_lower = output.lower()
        
        # Helper function to add violation with severity
        def add_violation(
            principle: ConstitutionalPrinciples,
            description: str,
            severity: str = "medium"
        ):
            violations.append({
                "principle": principle,
                "description": description,
                "severity": severity,
                "timestamp": time.time()
            })
        
        # Principle-specific checks with improved logic
        for principle in contract.principles:
            # AVOID_HARM: Check for harmful content
            if principle == ConstitutionalPrinciples.AVOID_HARM:
                harm_indicators = [
                    "kill", "hurt", "damage", "attack", "destroy",
                    "harm", "violence", "threat", "abuse", "discriminate"
                ]
                if any(indicator in output_lower for indicator in harm_indicators):
                    add_violation(
                        principle,
                        "Output contains potentially harmful language",
                        "high"
                    )
            
            # AVOID_HALLUCINATION: Check for unsupported claims
            elif principle == ConstitutionalPrinciples.AVOID_HALLUCINATION:
                # Check for absolute claims without evidence
                if re.search(r"\b(always|never|absolutely|definitely|proven fact|guarantee|guaranteed|certain)\b", output_lower):
                    add_violation(
                        principle,
                        "Output makes absolute claims without supporting evidence",
                        "medium"
                    )
                
                # Check for fabricated details
                if re.search(r"\b(research|study|experts)\b (show|prove|demonstrate|indicate|suggest|say)\b", output_lower) and "source" not in output_lower:
                    add_violation(
                        principle,
                        "Output references research or experts but doesn't cite sources",
                        "medium"
                    )
                
                # Check for excessive specificity without justification
                if re.search(r"\b(exactly|precisely|in 2023|in 2024)\b", output_lower) and len(output.split()) < 20:
                    add_violation(
                        principle,
                        "Output contains specific dates/numbers without sufficient context",
                        "low"
                    )
            
            # CITE_SOURCES: Check for proper attribution
            elif principle == ConstitutionalPrinciples.CITE_SOURCES:
                # Check for explicit citations
                if "according to" in output_lower or "source:" in output_lower:
                    pass  # Good
                # Check for parenthetical citations (e.g., Smith et al. (2023), (Jones, 2022))
                elif re.search(r"\b[A-Z][a-z]+ et al\.? \(\d{4}\)\b", output) or re.search(r"\(\d{4}\)", output):
                    pass  # Good
                # Check for mentions of research/studies without citation
                elif ("research" in output_lower or "study" in output_lower or "experts" in output_lower) and "source" not in output_lower:
                    add_violation(
                        principle,
                        "Output references research or experts but fails to provide citations",
                        "medium"
                    )
                # Check for factual claims without attribution
                elif re.search(r"\b(fact:|\btrue that|incorrect to say)\b", output_lower):
                    add_violation(
                        principle,
                        "Output presents factual claims as absolute without attribution",
                        "medium"
                    )
            
            # PRESERVE_PRIVACY: Check for private information
            elif principle == ConstitutionalPrinciples.PRESERVE_PRIVACY:
                privacy_indicators = [
                    "password", "ssn", "social security", "credit card",
                    "bank account", "private key", "personal data",
                    "confidential", "classified", "secret"
                ]
                if any(indicator in output_lower for indicator in privacy_indicators):
                    add_violation(
                        principle,
                        "Output may contain private or confidential information",
                        "high"
                    )
                
                # Check for PII patterns
                if re.search(r"\b([A-Z]{3}\d{6}|\d{3}-\d{2}-\d{4})\b", output):  # SSN/Employee ID patterns
                    add_violation(
                        principle,
                        "Output contains patterns matching personally identifiable information",
                        "high"
                    )
            
            # BE_CONCISE: Check for verbosity
            elif principle == ConstitutionalPrinciples.BE_CONCISE:
                word_count = len(output.split())
                sentence_count = len(output.split(".")) if "." in output else 1
                
                # Different thresholds for different severity levels
                if word_count > 300:
                    add_violation(
                        principle,
                        "Output is extremely long; consider breaking into sections",
                        "high"
                    )
                elif word_count > 200:
                    add_violation(
                        principle,
                        "Output is excessively verbose with low information density",
                        "medium"
                    )
                elif word_count > 100:
                    add_violation(
                        principle,
                        "Output could be more concise",
                        "low"
                    )
            
            # BE_ACCURATE: Check for factual accuracy and verifiability
            elif principle == ConstitutionalPrinciples.BE_ACCURATE:
                # Check for unverified claims
                unverified_patterns = [
                    r"\b(allegedly|reportedly|it is said)\b",
                    r"\b(some people|many believe)\b"
                ]
                if not any(re.search(pattern, output_lower) for pattern in unverified_patterns):
                    if "according to" not in output_lower and "source" not in output_lower:
                        if len(output) > 50 and re.search(r"\b(claim|assert|maintain)\b", output_lower):
                            add_violation(
                                principle,
                                "Output makes assertions without attribution or verification",
                                "medium"
                            )
            
            # BE_FAIR: Check for bias and balance
            elif principle == ConstitutionalPrinciples.BE_FAIR:
                # Check for absolute language that might indicate bias
                if re.search(r"\b(clearly|obviously|undoubtedly|of course)\b", output_lower, re.IGNORECASE):
                    add_violation(
                        principle,
                        "Output uses absolute language that may indicate bias",
                        "low"
                    )
                
                # Check for lack of opposing viewpoints
                if "however" not in output_lower and "on the other hand" not in output_lower:
                    if len(output) > 100 and "debate" in output_lower:
                        add_violation(
                            principle,
                        "Output discusses a debate but presents only one side",
                            "medium"
                        )
            
            # BE_TRANSPARENT: Check for transparency about limitations
            elif principle == ConstitutionalPrinciples.BE_TRANSPARENT:
                if "uncertainty" not in output_lower and "limitation" not in output_lower:
                    if "might be" in output_lower or "could be" in output_lower:
                        continue  # Already expresses uncertainty
                    if len(output) > 50 and not re.search(r"\b(likely|possibly|probably)\b", output_lower):
                        add_violation(
                            principle,
                            "Output presents information without acknowledging uncertainties",
                            "low"
                        )

        return violations

    def _calculate_scrutiny_score(
        self,
        prm_score: float,
        violations: List[Dict[str, Any]],
        contract: ConstitutionalContract
    ) -> float:
        """Calculate scrutiny score for compute routing.
        
        Args:
            prm_score: PRM confidence score (0-1)
            violations: List of detected violations
            contract: Constitutional contract
            
        Returns:
            Scrutiny score (0-1) used for routing decisions
        """
        # Calculate average violation severity (0-1 scale)
        severity_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        if violations:
            avg_severity = sum(severity_map.get(v["severity"], 0.6) for v in violations) / len(violations)
        else:
            avg_severity = 0.0
        
        # Combine PRM confidence and violation severity using contract weights
        scrutiny = (
            contract.scrutiny_alpha * (1.0 - prm_score) +  # Lower PRM → higher scrutiny
            contract.scrutiny_beta * avg_severity  # More severe violations → higher scrutiny
        )
        
        return max(0.0, min(1.0, scrutiny))

    def _determine_compute_strategy(
        self,
        scrutiny_score: float,
        contract: ConstitutionalContract
    ) -> Tuple[str, float]:
        """Determine compute strategy based on scrutiny score.
        
        Args:
            scrutiny_score: Scrutiny score (0-1)
            contract: Constitutional contract
            
        Returns:
            Tuple of (strategy_name, compute_budget)
        """
        if scrutiny_score < 0.3:
            # Low scrutiny - lean compute
            return "efficient", contract.base_compute_budget * 0.5
        elif scrutiny_score < 0.6:
            # Medium scrutiny - standard compute
            return "standard", contract.base_compute_budget
        elif scrutiny_score < 0.8:
            # High scrutiny - enhanced compute
            return "enhanced", contract.base_compute_budget * 1.5
        else:
            # Critical scrutiny - maximum compute
            return "maximum", contract.base_compute_budget * 2.0

    def check_step(
        self, 
        output: str, 
        contract: ConstitutionalContract,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[Dict[str, Any]], float, float, float, str, float]:
        """
        Check a skill step output against its constitutional contract with compute routing.

        Args:
            output: The output from the skill step.
            contract: The constitutional contract to enforce.
            context: Optional context for evaluation (task, previous steps, etc.)

        Returns:
            Tuple: (is_pass, violations, prm_score, constitutional_score, 
                   scrutiny_score, compute_strategy, compute_budget)
        """
        start_time = time.time()
        
        # Validate inputs
        if not output:
            # Empty output gets maximum scrutiny
            scrutiny_score = 1.0
            strategy, budget = self._determine_compute_strategy(scrutiny_score, contract)
            return False, [], 0.0, 0.0, scrutiny_score, strategy, budget
        
        if not context:
            context = {}

        # Log the check
        self.logger.debug(f"Checking step {contract.step_name}", 
                        module="core", 
                        contract_id=f"{contract.skill_id}:{contract.step_name}")

        # Get PRM score
        prm_score = self._get_prm_score(output, contract.principles, context)

        # Advanced critique with severity assessment
        violations = self._advanced_critique(output, contract, context)

        # Calculate constitutional score (weighted combination)
        # Base score on PRM, reduced by violations
        violation_penalty = sum(
            0.1 if v["severity"] == "low" else 0.3 if v["severity"] == "medium" else 0.6
            for v in violations
        )
        constitutional_score = max(0.0, prm_score - violation_penalty)

        # Calculate scrutiny score for compute routing
        scrutiny_score = self._calculate_scrutiny_score(prm_score, violations, contract)

        # Determine compute strategy and budget
        compute_strategy, compute_budget = self._determine_compute_strategy(
            scrutiny_score, contract
        )

        # Check thresholds with enhanced logic
        is_pass = (
            constitutional_score >= contract.constitutional_threshold and
            len(violations) == 0 or (scrutiny_score < 0.3 and constitutional_score >= 0.5)
        )

        # Check compute budget if specified
        elapsed = time.time() - start_time
        if contract.compute_budget and elapsed > contract.compute_budget:
            self.logger.warning(
                f"Compute budget exceeded for {contract.skill_id}. "
                f"Elapsed: {elapsed:.2f}s, Budget: {contract.compute_budget}s",
                module="core",
                budget_exceeded=True
            )

        # Log results
        self.logger.info(
            f"Step {contract.step_name} complete: "
            f"PRM={prm_score:.2f}, "
            f"Constitutional={constitutional_score:.2f}, "
            f"Violations={len(violations)}, "
            f"Scrutiny={scrutiny_score:.2f}, "
            f"Strategy={compute_strategy}",
            module="core",
            step_name=contract.step_name
        )

        return (
            is_pass,
            violations,
            prm_score,
            constitutional_score,
            scrutiny_score,
            compute_strategy,
            compute_budget
        )


class ComputeRouter:
    """Routes steps to appropriate compute strategies based on scrutiny."""

    def __init__(self, logger: Optional[Logger] = None):
        """Initialize the ComputeRouter."""
        self.logger = logger or Logger()
        self.strategies = {
            "efficient": self._strategy_efficient,
            "standard": self._strategy_standard,
            "enhanced": self._strategy_enhanced,
            "maximum": self._strategy_maximum
        }

    def route(
        self,
        output: str,
        contract: ConstitutionalContract,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route a step to appropriate compute strategy.
        
        Args:
            output: Step output to evaluate
            contract: Constitutional contract
            context: Evaluation context
            
        Returns:
            Dictionary with routing decision and metrics
        """
        reviewer = ConstitutionalReviewer(logger=self.logger)
        
        # Get full evaluation
        (
            is_pass,
            violations,
            prm_score,
            constitutional_score,
            scrutiny_score,
            strategy,
            compute_budget
        ) = reviewer.check_step(output, contract, context)
        
        # Execute routing strategy
        if strategy in self.strategies:
            result = self.strategies[strategy](
                output=output,
                contract=contract,
                violations=violations,
                context=context
            )
        else:
            # Default to standard strategy
            result = self._strategy_standard(
                output=output,
                contract=contract,
                violations=violations,
                context=context
            )
        
        return {
            "is_pass": is_pass,
            "violations": violations,
            "prm_score": prm_score,
            "constitutional_score": constitutional_score,
            "scrutiny_score": scrutiny_score,
            "compute_strategy": strategy,
            "compute_budget": compute_budget,
            "routing_decision": result,
            "timestamp": time.time()
        }

    def _strategy_efficient(self, **kwargs) -> Dict[str, Any]:
        """Efficient compute strategy: minimal processing."""
        return {
            "action": "regenerate",
            "iterations": 1,
            "timeout": 5.0,
            "enhanced_verification": False
        }

    def _strategy_standard(self, **kwargs) -> Dict[str, Any]:
        """Standard compute strategy: normal processing."""
        return {
            "action": "regenerate",
            "iterations": 2,
            "timeout": 10.0,
            "enhanced_verification": True
        }

    def _strategy_enhanced(self, **kwargs) -> Dict[str, Any]:
        """Enhanced compute strategy: additional verification."""
        return {
            "action": "regenerate",
            "iterations": 3,
            "timeout": 20.0,
            "enhanced_verification": True,
            "ensemble_models": 2
        }

    def _strategy_maximum(self, **kwargs) -> Dict[str, Any]:
        """Maximum compute strategy: extensive verification."""
        return {
            "action": "escalate",
            "iterations": 5,
            "timeout": 30.0,
            "enhanced_verification": True,
            "ensemble_models": 3,
            "human_in_the_loop": True,
            "cross_model_consensus": True
        }


# Utility functions
def create_constitutional_reviewer(
    critique_model: str = "bert-base-uncased",
    critique_model_size: str = "base",
    use_prm: bool = True,
    logger: Optional[Logger] = None,
    prm_config: Optional[PRMConfig] = None
) -> ConstitutionalReviewer:
    """Create a ConstitutionalReviewer instance."""
    return ConstitutionalReviewer(
        critique_model=critique_model,
        critique_model_size=critique_model_size,
        use_prm=use_prm,
        logger=logger,
        prm_config=prm_config
    )


def evaluate_step(
    output: str,
    contract: ConstitutionalContract,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[Logger] = None
) -> Tuple[bool, List[Dict[str, Any]], float, float, float, str, float]:
    """Convenience function for step evaluation."""
    reviewer = ConstitutionalReviewer(logger=logger)
    return reviewer.check_step(output, contract, context)


def calculate_scrutiny_score(
    prm_score: float,
    violations: List[Dict[str, Any]],
    contract: ConstitutionalContract
) -> float:
    """Calculate scrutiny score for compute routing."""
    reviewer = ConstitutionalReviewer()
    return reviewer._calculate_scrutiny_score(prm_score, violations, contract)


def determine_compute_strategy(
    scrutiny_score: float,
    contract: ConstitutionalContract
) -> Tuple[str, float]:
    """Determine compute strategy based on scrutiny."""
    reviewer = ConstitutionalReviewer()
    return reviewer._determine_compute_strategy(scrutiny_score, contract)