"""
Information Capacity Estimator for agent ranking.

Implements methods to estimate an agent's information capacity in bits,
supporting entropy-based and mutual information-based estimation.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import Counter


@dataclass
class CapacityResult:
    """
    Result of a capacity estimation.
    
    Attributes:
        capacity_bits: Estimated information capacity in bits
        method: Estimation method used (entropy, mutual_info, combined)
        confidence: Confidence score for the estimate (0.0 to 1.0)
        timestamp: Unix timestamp when the estimate was computed
    """
    capacity_bits: float
    method: str
    confidence: float
    timestamp: float
    
    def __post_init__(self):
        """Validate the result after initialization."""
        if self.capacity_bits < 0:
            raise ValueError("capacity_bits must be non-negative")
        if self.method not in ("entropy", "mutual_info", "combined"):
            raise ValueError(f"Invalid method: {self.method}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


class CapacityEstimator:
    """
    Estimates the information capacity of agents for test-time ensemble weighting.
    
    Information capacity represents how much useful information an agent can provide
    for a given task. This is measured in bits using entropy-based and mutual
    information-based methods.
    
    Example:
        >>> estimator = CapacityEstimator()
        >>> result = estimator.estimate_capacity(agent, task_context)
        >>> print(f"Capacity: {result.capacity_bits:.2f} bits")
    """
    
    def __init__(
        self,
        default_method: str = "combined",
        num_samples: int = 10,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the capacity estimator.
        
        Args:
            default_method: Default estimation method (entropy, mutual_info, combined)
            num_samples: Number of samples to use for estimation
            random_seed: Optional random seed for reproducibility
        """
        if default_method not in ("entropy", "mutual_info", "combined"):
            raise ValueError(f"Invalid method: {default_method}")
        
        self.default_method = default_method
        self.num_samples = num_samples
        self.random_seed = random_seed
        
    def estimate_capacity(
        self,
        agent: Any,
        task_context: Dict[str, Any],
        method: Optional[str] = None
    ) -> CapacityResult:
        """
        Estimate an agent's information capacity for a given task.
        
        Args:
            agent: An agent object that can generate responses
            task_context: Dictionary containing task-related information
            method: Estimation method (overrides default)
            
        Returns:
            CapacityResult with capacity estimate in bits
        """
        method = method or self.default_method
        
        if method == "entropy":
            capacity, confidence = self._estimate_entropy(agent, task_context)
        elif method == "mutual_info":
            capacity, confidence = self._estimate_mutual_info(agent, task_context)
        else:  # combined
            ent_cap, ent_conf = self._estimate_entropy(agent, task_context)
            mi_cap, mi_conf = self._estimate_mutual_info(agent, task_context)
            # Weighted combination
            capacity = 0.5 * ent_cap + 0.5 * mi_cap
            confidence = 0.5 * ent_conf + 0.5 * mi_conf
        
        return CapacityResult(
            capacity_bits=capacity,
            method=method,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _estimate_entropy(
        self,
        agent: Any,
        task_context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Estimate capacity using entropy of response distribution.
        
        Higher entropy indicates more diverse responses, which can indicate
        higher information capacity.
        
        Args:
            agent: Agent object
            task_context: Task context dictionary
            
        Returns:
            Tuple of (capacity_bits, confidence)
        """
        responses = self._sample_responses(agent, task_context)
        
        if not responses:
            return 0.0, 0.0
        
        # Discretize responses into tokens for entropy calculation
        token_counts: Counter = Counter()
        total_tokens = 0
        
        for response in responses:
            # Simple tokenization by whitespace
            tokens = str(response).lower().split()
            token_counts.update(tokens)
            total_tokens += len(tokens)
        
        if total_tokens == 0:
            return 0.0, 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            if count > 0:
                p = count / total_tokens
                entropy -= p * math.log2(p)
        
        # Confidence based on number of samples and response diversity
        unique_tokens = len(token_counts)
        confidence = min(1.0, (len(responses) / self.num_samples) * (unique_tokens / max(total_tokens, 1)))
        
        return entropy, confidence
    
    def _estimate_mutual_info(
        self,
        agent: Any,
        task_context: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Estimate capacity using mutual information between responses and task context.
        
        Higher mutual information indicates responses are more informative
        about the task, suggesting higher task-relevant capacity.
        
        Args:
            agent: Agent object
            task_context: Task context dictionary
            
        Returns:
            Tuple of (capacity_bits, confidence)
        """
        responses = self._sample_responses(agent, task_context)
        
        if not responses:
            return 0.0, 0.0
        
        # Extract task-relevant features from context
        task_features = self._extract_task_features(task_context)
        
        # Calculate response entropy
        response_entropy = self._calculate_response_entropy(responses)
        
        # Calculate task-response mutual information
        # Simplified: use overlap between response tokens and task keywords
        task_keywords = set()
        if "keywords" in task_context:
            task_keywords = set(k.lower() for k in task_context["keywords"])
        if "query" in task_context:
            task_keywords.update(str(task_context["query"]).lower().split())
        
        # Count task-relevant tokens in responses
        relevant_count = 0
        total_tokens = 0
        
        for response in responses:
            tokens = set(str(response).lower().split())
            relevant_count += len(tokens & task_keywords)
            total_tokens += len(tokens)
        
        if total_tokens == 0:
            return 0.0, 0.0
        
        # Mutual information approximation
        relevance_ratio = relevant_count / total_tokens
        mi_estimate = response_entropy * relevance_ratio
        
        # Confidence based on response consistency and sample size
        confidence = min(1.0, (len(responses) / self.num_samples) * relevance_ratio)
        
        return mi_estimate, confidence
    
    def _sample_responses(
        self,
        agent: Any,
        task_context: Dict[str, Any]
    ) -> List[Any]:
        """
        Sample multiple responses from an agent.
        
        Args:
            agent: Agent object with generate() or similar method
            task_context: Task context for generation
            
        Returns:
            List of sampled responses
        """
        responses = []
        
        # Try different agent interfaces
        for _ in range(self.num_samples):
            try:
                # Try generate method
                if hasattr(agent, 'generate'):
                    response = agent.generate(task_context)
                # Try __call__ method
                elif callable(agent):
                    response = agent(task_context)
                # Try respond method
                elif hasattr(agent, 'respond'):
                    response = agent.respond(task_context)
                else:
                    # If agent has no response method, use string representation
                    response = str(agent)
                
                responses.append(response)
            except Exception:
                # If sampling fails, continue with fewer samples
                continue
        
        return responses
    
    def _extract_task_features(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant features from task context.
        
        Args:
            task_context: Task context dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if "query" in task_context:
            features["query_length"] = len(str(task_context["query"]))
        
        if "keywords" in task_context:
            features["num_keywords"] = len(task_context["keywords"])
        
        if "domain" in task_context:
            features["domain"] = task_context["domain"]
        
        return features
    
    def _calculate_response_entropy(self, responses: List[Any]) -> float:
        """
        Calculate entropy of a set of responses.
        
        Args:
            responses: List of response strings/objects
            
        Returns:
            Entropy in bits
        """
        if not responses:
            return 0.0
        
        # Convert responses to string representation
        response_strs = [str(r) for r in responses]
        
        # Count unique responses
        response_counts = Counter(response_strs)
        total = len(response_strs)
        
        # Calculate entropy
        entropy = 0.0
        for count in response_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def rank_agents(
        self,
        agents: List[Tuple[str, Any]],
        task: Dict[str, Any],
        method: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank agents by their estimated information capacity.
        
        Args:
            agents: List of (agent_id, agent_object) tuples
            task: Task context dictionary
            method: Estimation method (overrides default)
            
        Returns:
            List of (agent_id, capacity_bits) tuples sorted descending by capacity
        """
        if not agents:
            return []
        
        method = method or self.default_method
        capacities = []
        
        for agent_id, agent in agents:
            result = self.estimate_capacity(agent, task, method=method)
            capacities.append((agent_id, result.capacity_bits))
        
        # Sort by capacity descending
        capacities.sort(key=lambda x: x[1], reverse=True)
        
        return capacities
    
    def rank_agents_with_results(
        self,
        agents: List[Tuple[str, Any]],
        task: Dict[str, Any],
        method: Optional[str] = None
    ) -> List[Tuple[str, CapacityResult]]:
        """
        Rank agents and return full CapacityResult objects.
        
        Args:
            agents: List of (agent_id, agent_object) tuples
            task: Task context dictionary
            method: Estimation method (overrides default)
            
        Returns:
            List of (agent_id, CapacityResult) tuples sorted descending by capacity
        """
        if not agents:
            return []
        
        method = method or self.default_method
        results = []
        
        for agent_id, agent in agents:
            result = self.estimate_capacity(agent, task, method=method)
            results.append((agent_id, result))
        
        # Sort by capacity descending
        results.sort(key=lambda x: x[1].capacity_bits, reverse=True)
        
        return results
