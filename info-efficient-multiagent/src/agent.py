"""
Base Reasoning Agent

A simple reasoning agent that can generate responses and have its
information capacity measured. Designed for multi-agent systems where
capacity-aware compute allocation is needed.
"""

import json
import random
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import time

from .capacity import InformationCapacityEstimator, CapacityResult


@dataclass
class AgentResponse:
    """A single response from an agent."""
    
    content: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentConfig:
    """Configuration for a reasoning agent."""
    
    agent_id: str
    model_name: str = "base-agent"
    max_tokens: int = 256
    temperature: float = 0.7
    seed: Optional[int] = None
    specializations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "specializations": self.specializations,
        }


class ReasoningAgent:
    """
    A base reasoning agent that generates responses and tracks capacity.
    
    This agent is designed for use in multi-agent systems where agents
    need to be compared by their information capacity. It supports:
    
    - Response generation with configurable strategies
    - Information capacity measurement
    - Specialization tracking (domains the agent excels in)
    - Response history for analysis
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        model: Optional[Callable] = None,
        capacity_estimator: Optional[InformationCapacityEstimator] = None,
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Agent configuration. Uses defaults if not provided.
            model: Optional callable model for response generation.
                   Signature: model(prompt: str, **kwargs) -> str
            capacity_estimator: Estimator for measuring information capacity.
        """
        self.config = config or AgentConfig(agent_id="default-agent")
        self.model = model
        self.capacity_estimator = capacity_estimator or InformationCapacityEstimator()
        
        self.response_history: List[AgentResponse] = []
        self.capacity_history: List[CapacityResult] = []
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    @property
    def agent_id(self) -> str:
        """Get the agent's ID."""
        return self.config.agent_id
    
    @property
    def specializations(self) -> List[str]:
        """Get the agent's specializations."""
        return self.config.specializations
    
    def generate(
        self,
        prompt: str,
        num_responses: int = 1,
        **kwargs,
    ) -> List[AgentResponse]:
        """
        Generate one or more responses to a prompt.
        
        Args:
            prompt: The input prompt/question.
            num_responses: Number of responses to generate.
            **kwargs: Additional arguments passed to the model.
            
        Returns:
            List of AgentResponse objects.
        """
        responses = []
        
        for _ in range(num_responses):
            if self.model is not None:
                # Use provided model
                content = self.model(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                )
            else:
                # Use mock generation for testing
                content = self._mock_generate(prompt)
            
            response = AgentResponse(
                content=content,
                confidence=self._estimate_confidence(content),
                metadata={"prompt": prompt[:100]},
            )
            responses.append(response)
            self.response_history.append(response)
        
        return responses
    
    def _mock_generate(self, prompt: str) -> str:
        """
        Mock response generation for testing.
        
        Generates responses based on prompt keywords to simulate
        domain-specific knowledge.
        """
        prompt_lower = prompt.lower()
        
        # Generate domain-specific mock responses
        templates = {
            "math": [
                "The solution involves applying the fundamental theorem. "
                "We can derive the answer by following the standard procedure.",
                "To solve this, we need to identify the key variables and "
                "apply the relevant mathematical operations step by step.",
                "Mathematical analysis shows that the solution exists and "
                "can be computed using established methods.",
            ],
            "science": [
                "Based on scientific principles, the phenomenon can be "
                "explained through established theories and empirical evidence.",
                "Scientific analysis reveals that multiple factors contribute "
                "to the observed behavior, following known physical laws.",
                "The scientific method allows us to systematically investigate "
                "and derive conclusions from the available evidence.",
            ],
            "reasoning": [
                "Logical analysis of the premises leads to the conclusion "
                "that the argument follows valid inference patterns.",
                "By applying deductive reasoning, we can trace the logical "
                "connections and arrive at a sound conclusion.",
                "Critical thinking reveals that multiple perspectives should "
                "be considered before drawing a final conclusion.",
            ],
            "general": [
                "This is a comprehensive topic that requires careful analysis. "
                "Multiple factors should be considered in formulating a response.",
                "The question touches on several important aspects that merit "
                "detailed examination and thoughtful consideration.",
                "A thorough analysis would involve examining the context, "
                "identifying key elements, and synthesizing relevant information.",
            ],
        }
        
        # Select template based on prompt keywords
        for domain, domain_templates in templates.items():
            if domain in prompt_lower:
                return random.choice(domain_templates)
        
        # Fall back to general template
        return random.choice(templates["general"])
    
    def _estimate_confidence(self, response: str) -> float:
        """
        Estimate confidence based on response characteristics.
        
        Longer, more detailed responses typically indicate higher confidence.
        """
        if not response:
            return 0.0
        
        # Base confidence on length and structure
        word_count = len(response.split())
        
        # Scale: 10 words = 0.5, 50+ words = 1.0
        base_confidence = min(1.0, 0.3 + word_count / 50)
        
        # Boost for structured responses
        if any(marker in response for marker in ["because", "therefore", "thus", "since"]):
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def measure_capacity(
        self,
        responses: Optional[List[str]] = None,
        context: Optional[str] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> CapacityResult:
        """
        Measure the agent's information capacity.
        
        Args:
            responses: Responses to measure. Uses history if not provided.
            context: Task context for the measurement.
            reference_responses: Reference responses for comparison.
            
        Returns:
            CapacityResult with capacity in bits.
        """
        if responses is None:
            responses = [r.content for r in self.response_history[-10:]]
        
        result = self.capacity_estimator.estimate_capacity(
            responses=responses,
            context=context,
            reference_responses=reference_responses,
        )
        
        self.capacity_history.append(result)
        return result
    
    def get_current_capacity(self) -> float:
        """Get the most recent capacity measurement in bits."""
        if not self.capacity_history:
            return 0.0
        return self.capacity_history[-1].capacity_bits
    
    def get_average_capacity(self, n: int = 5) -> float:
        """Get average capacity over the last n measurements."""
        if not self.capacity_history:
            return 0.0
        
        recent = self.capacity_history[-n:]
        return sum(r.capacity_bits for r in recent) / len(recent)
    
    def clear_history(self) -> None:
        """Clear response and capacity history."""
        self.response_history.clear()
        self.capacity_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to dictionary."""
        return {
            "config": self.config.to_dict(),
            "response_count": len(self.response_history),
            "capacity_measurements": len(self.capacity_history),
            "current_capacity": self.get_current_capacity(),
            "average_capacity": self.get_average_capacity(),
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save agent state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReasoningAgent":
        """Load agent state from file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        config = AgentConfig(**data["config"])
        return cls(config=config)


class SpecializedAgent(ReasoningAgent):
    """
    A reasoning agent with specific domain specializations.
    
    Specialized agents have higher capacity in their domains
    and generate more detailed responses for relevant prompts.
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        model: Optional[Callable] = None,
        capacity_estimator: Optional[InformationCapacityEstimator] = None,
        domain_knowledge: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize a specialized agent.
        
        Args:
            config: Agent configuration with specializations set.
            model: Optional callable model.
            capacity_estimator: Capacity estimator.
            domain_knowledge: Dict mapping domain to knowledge snippets.
        """
        super().__init__(config, model, capacity_estimator)
        self.domain_knowledge = domain_knowledge or {}
    
    def _mock_generate(self, prompt: str) -> str:
        """Generate with domain-specific knowledge enhancement."""
        prompt_lower = prompt.lower()
        
        # Check if prompt matches a specialization
        for domain in self.config.specializations:
            if domain.lower() in prompt_lower:
                # Generate detailed response using domain knowledge
                knowledge = self.domain_knowledge.get(domain, [])
                if knowledge:
                    base_response = super()._mock_generate(prompt)
                    knowledge_snippet = random.choice(knowledge)
                    return f"{base_response}\n\nSpecialized insight: {knowledge_snippet}"
        
        return super()._mock_generate(prompt)
    
    def measure_capacity(
        self,
        responses: Optional[List[str]] = None,
        context: Optional[str] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> CapacityResult:
        """Measure capacity with domain boosting."""
        result = super().measure_capacity(responses, context, reference_responses)
        
        # Boost capacity if context matches specializations
        if context:
            for domain in self.config.specializations:
                if domain.lower() in context.lower():
                    # Domain expertise increases effective capacity
                    result.capacity_bits *= 1.2
                    break
        
        return result


def create_agent(
    agent_id: str = "agent-1",
    specializations: Optional[List[str]] = None,
    model: Optional[Callable] = None,
    **kwargs,
) -> ReasoningAgent:
    """
    Factory function to create a reasoning agent.
    
    Args:
        agent_id: Unique identifier for the agent.
        specializations: List of domain specializations.
        model: Optional callable model for response generation.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured ReasoningAgent instance.
    """
    config = AgentConfig(
        agent_id=agent_id,
        specializations=specializations or [],
        **kwargs,
    )
    
    if specializations:
        return SpecializedAgent(config=config, model=model)
    
    return ReasoningAgent(config=config, model=model)


def create_agent_pool(
    num_agents: int = 5,
    specializations_list: Optional[List[List[str]]] = None,
    model: Optional[Callable] = None,
) -> List[ReasoningAgent]:
    """
    Create a pool of agents with diverse specializations.
    
    Args:
        num_agents: Number of agents to create.
        specializations_list: Optional list of specializations per agent.
        model: Optional callable model for all agents.
        
    Returns:
        List of ReasoningAgent instances.
    """
    agents = []
    
    if specializations_list is None:
        # Create diverse specializations
        all_domains = ["math", "science", "reasoning", "language", "coding"]
        specializations_list = []
        for i in range(num_agents):
            # Each agent gets 1-2 random specializations
            n_specs = random.randint(1, 2)
            specs = random.sample(all_domains, min(n_specs, len(all_domains)))
            specializations_list.append(specs)
    
    for i, specs in enumerate(specializations_list):
        agent = create_agent(
            agent_id=f"agent-{i+1}",
            specializations=specs,
            model=model,
        )
        agents.append(agent)
    
    return agents
