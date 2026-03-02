"""
Information Capacity Estimator

Measures each agent's information capacity in bits based on the metric from
Faustino (2026) - "Information-theoretic analysis of world models".

The information capacity represents how much task-relevant knowledge an agent
possesses, measured in bits. Higher capacity indicates better knowledge about
the environment/task.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class CapacityResult:
    """Result of an information capacity measurement."""
    
    capacity_bits: float
    entropy_bits: float
    mutual_info_bits: float
    response_diversity: float
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class InformationCapacityEstimator:
    """
    Estimates the information capacity of an agent in bits.
    
    Based on the insight that an agent's world model capacity can be measured
    by quantifying how much information the agent's policy captures about
    the environment.
    
    Methods:
    - Entropy-based: Measures uncertainty in agent's output distribution
    - Mutual Information: Measures dependency between context and responses
    - Compression-based: Measures how well responses can be compressed
    """
    
    def __init__(
        self,
        method: str = "combined",
        num_samples: int = 10,
        base: float = 2.0,  # Log base (2 = bits)
    ):
        """
        Initialize the estimator.
        
        Args:
            method: Estimation method ("entropy", "mutual_info", "compression", "combined")
            num_samples: Number of samples to use for estimation
            base: Logarithm base (2 for bits, e for nats)
        """
        self.method = method
        self.num_samples = num_samples
        self.base = base
    
    def estimate_capacity(
        self,
        responses: List[str],
        context: Optional[str] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> CapacityResult:
        """
        Estimate the information capacity from a list of agent responses.
        
        Args:
            responses: List of response strings from the agent
            context: Optional task context string
            reference_responses: Optional reference/ground truth responses
            
        Returns:
            CapacityResult with capacity in bits and other metrics
        """
        if not responses:
            return CapacityResult(
                capacity_bits=0.0,
                entropy_bits=0.0,
                mutual_info_bits=0.0,
                response_diversity=0.0,
                confidence=0.0,
                metadata={"error": "No responses provided"}
            )
        
        # Compute individual metrics
        entropy = self._compute_entropy(responses)
        diversity = self._compute_diversity(responses)
        
        # Compute mutual information if context provided
        mutual_info = 0.0
        if context and reference_responses:
            mutual_info = self._compute_mutual_information(
                responses, context, reference_responses
            )
        
        # Compute final capacity based on method
        if self.method == "entropy":
            capacity = entropy
        elif self.method == "mutual_info":
            capacity = mutual_info
        elif self.method == "compression":
            capacity = self._compute_compression_capacity(responses)
        else:  # combined
            # Weighted combination: entropy captures uncertainty, diversity captures knowledge
            # Higher entropy + higher diversity = higher capacity
            capacity = 0.4 * entropy + 0.3 * diversity * 10 + 0.3 * mutual_info
        
        # Confidence based on number of samples
        confidence = min(1.0, len(responses) / self.num_samples)
        
        return CapacityResult(
            capacity_bits=capacity,
            entropy_bits=entropy,
            mutual_info_bits=mutual_info,
            response_diversity=diversity,
            confidence=confidence,
            metadata={
                "method": self.method,
                "num_responses": len(responses),
                "has_context": context is not None,
            }
        )
    
    def _compute_entropy(self, responses: List[str]) -> float:
        """
        Compute the entropy of the response distribution.
        
        Higher entropy indicates more diverse/uncertain responses,
        which can indicate either broad knowledge or lack of focused knowledge.
        """
        if not responses:
            return 0.0
        
        # Tokenize responses and count token frequencies
        all_tokens = []
        for response in responses:
            tokens = response.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return 0.0
        
        # Count token frequencies
        counter = Counter(all_tokens)
        total = len(all_tokens)
        
        # Compute entropy
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p, self.base)
        
        return entropy
    
    def _compute_diversity(self, responses: List[str]) -> float:
        """
        Compute response diversity (unique responses / total responses).
        
        Higher diversity indicates the agent can generate varied responses,
        suggesting broader knowledge.
        """
        if not responses:
            return 0.0
        
        unique = len(set(responses))
        total = len(responses)
        
        return unique / total
    
    def _compute_mutual_information(
        self,
        responses: List[str],
        context: str,
        reference_responses: List[str],
    ) -> float:
        """
        Compute mutual information between responses and reference.
        
        Higher MI indicates responses are more aligned with what's expected,
        suggesting better task-relevant knowledge.
        """
        if not responses or not reference_responses:
            return 0.0
        
        # Simple approach: measure overlap in tokens
        response_tokens = set()
        for r in responses:
            response_tokens.update(r.lower().split())
        
        reference_tokens = set()
        for r in reference_responses:
            reference_tokens.update(r.lower().split())
        
        if not response_tokens or not reference_tokens:
            return 0.0
        
        # Jaccard similarity as proxy for mutual information
        intersection = len(response_tokens & reference_tokens)
        union = len(response_tokens | reference_tokens)
        
        if union == 0:
            return 0.0
        
        # Scale to bits (rough approximation)
        similarity = intersection / union
        mi = -math.log(1 - similarity + 1e-10, self.base)
        
        return mi
    
    def _compute_compression_capacity(self, responses: List[str]) -> float:
        """
        Compute capacity based on compression ratio.
        
        More compressible responses suggest patterns/structure in knowledge.
        Less compressible suggests more random/unstructured knowledge.
        """
        if not responses:
            return 0.0
        
        # Concatenate all responses
        combined = " ".join(responses)
        
        # Simple compression estimate using run-length encoding concept
        # Count repeated patterns
        original_len = len(combined)
        
        # Count unique character patterns
        char_counts = Counter(combined)
        
        # Theoretical minimum bits to encode
        entropy = 0.0
        for count in char_counts.values():
            p = count / original_len
            if p > 0:
                entropy -= p * math.log(p, self.base)
        
        # Capacity is entropy * length (total information content)
        capacity = entropy * original_len / 100  # Scale down for reasonable values
        
        return capacity
    
    def compare_agents(
        self,
        agent_responses: Dict[str, List[str]],
        context: Optional[str] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> Dict[str, CapacityResult]:
        """
        Compare information capacity across multiple agents.
        
        Args:
            agent_responses: Dict mapping agent_id to list of responses
            context: Optional task context
            reference_responses: Optional reference responses
            
        Returns:
            Dict mapping agent_id to CapacityResult, sorted by capacity
        """
        results = {}
        
        for agent_id, responses in agent_responses.items():
            results[agent_id] = self.estimate_capacity(
                responses=responses,
                context=context,
                reference_responses=reference_responses,
            )
        
        return results
    
    def rank_agents_by_capacity(
        self,
        agent_responses: Dict[str, List[str]],
        context: Optional[str] = None,
        reference_responses: Optional[List[str]] = None,
    ) -> List[tuple]:
        """
        Rank agents by their information capacity (highest first).
        
        Returns:
            List of (agent_id, capacity_bits) tuples, sorted descending
        """
        results = self.compare_agents(agent_responses, context, reference_responses)
        
        ranked = [
            (agent_id, result.capacity_bits)
            for agent_id, result in results.items()
        ]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)


def create_estimator(method: str = "combined") -> InformationCapacityEstimator:
    """Factory function to create an estimator."""
    return InformationCapacityEstimator(method=method)
