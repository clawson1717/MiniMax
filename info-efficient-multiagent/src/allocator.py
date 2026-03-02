"""
Capacity-Aware Compute Allocator

Adapts CATTS-style test-time scaling to allocate compute based on
information capacity rather than uncertainty. The key insight is that
higher-capacity agents receive MORE compute budget (inverse of typical
uncertainty-based scaling), since they possess more task-relevant knowledge.

Based on:
- CATTS (Lee et al., 2026): Agentic Test-Time Scaling
- Information-theoretic analysis (Faustino, 2026): World model capacity
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AllocationStrategy(Enum):
    """Strategy for compute allocation."""
    CAPACITY_PROPORTIONAL = "capacity_proportional"  # Linear with capacity
    CAPACITY_THRESHOLD = "capacity_threshold"  # Step function based on thresholds
    CAPACITY_EXPONENTIAL = "capacity_exponential"  # Exponential scaling
    ADAPTIVE = "adaptive"  # Combines capacity with task complexity


@dataclass
class AllocationResult:
    """Result of compute allocation."""
    
    agent_id: str
    compute_budget: float
    capacity_bits: float
    allocation_ratio: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocatorConfig:
    """Configuration for the compute allocator."""
    
    base_compute: float = 1.0
    max_compute: float = 10.0
    min_compute: float = 0.5
    capacity_threshold_low: float = 2.0  # bits
    capacity_threshold_high: float = 8.0  # bits
    scaling_factor: float = 1.0
    strategy: AllocationStrategy = AllocationStrategy.CAPACITY_PROPORTIONAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_compute": self.base_compute,
            "max_compute": self.max_compute,
            "min_compute": self.min_compute,
            "capacity_threshold_low": self.capacity_threshold_low,
            "capacity_threshold_high": self.capacity_threshold_high,
            "scaling_factor": self.scaling_factor,
            "strategy": self.strategy.value,
        }


class CapacityAwareAllocator:
    """
    Allocates compute budget based on information capacity.
    
    Unlike uncertainty-based scaling (CATTS) where high uncertainty
    triggers more compute, this allocator gives more compute to
    higher-capacity agents. The rationale:
    
    - High-capacity agents have more task-relevant knowledge
    - Investing compute in them yields better returns
    - Low-capacity agents may not benefit from more compute
    
    Allocation strategies:
    1. Capacity Proportional: compute = base * (1 + capacity_factor)
    2. Capacity Threshold: step function based on capacity tiers
    3. Capacity Exponential: compute = base * exp(capacity * factor)
    4. Adaptive: considers both capacity and task complexity
    """
    
    def __init__(self, config: Optional[AllocatorConfig] = None):
        """
        Initialize the allocator.
        
        Args:
            config: Allocator configuration. Uses defaults if not provided.
        """
        self.config = config or AllocatorConfig()
        self.allocation_history: List[AllocationResult] = []
    
    def allocate(
        self,
        agent_id: str,
        capacity_bits: float,
        task_complexity: Optional[float] = None,
    ) -> AllocationResult:
        """
        Allocate compute budget for an agent based on its capacity.
        
        Args:
            agent_id: Identifier for the agent.
            capacity_bits: Information capacity in bits.
            task_complexity: Optional task complexity factor (0-1).
            
        Returns:
            AllocationResult with compute budget and metadata.
        """
        strategy = self.config.strategy
        
        if strategy == AllocationStrategy.CAPACITY_PROPORTIONAL:
            compute = self._allocate_proportional(capacity_bits)
        elif strategy == AllocationStrategy.CAPACITY_THRESHOLD:
            compute = self._allocate_threshold(capacity_bits)
        elif strategy == AllocationStrategy.CAPACITY_EXPONENTIAL:
            compute = self._allocate_exponential(capacity_bits)
        elif strategy == AllocationStrategy.ADAPTIVE:
            compute = self._allocate_adaptive(capacity_bits, task_complexity)
        else:
            compute = self.config.base_compute
        
        # Clamp to min/max bounds
        compute = max(self.config.min_compute, min(self.config.max_compute, compute))
        
        # Calculate allocation ratio
        allocation_ratio = compute / self.config.base_compute
        
        result = AllocationResult(
            agent_id=agent_id,
            compute_budget=compute,
            capacity_bits=capacity_bits,
            allocation_ratio=allocation_ratio,
            strategy=strategy.value,
            metadata={
                "task_complexity": task_complexity,
                "base_compute": self.config.base_compute,
            }
        )
        
        self.allocation_history.append(result)
        return result
    
    def _allocate_proportional(self, capacity_bits: float) -> float:
        """
        Proportional allocation: compute scales linearly with capacity.
        
        Formula: compute = base * (1 + capacity / threshold_high)
        """
        factor = 1 + (capacity_bits / self.config.capacity_threshold_high)
        return self.config.base_compute * factor * self.config.scaling_factor
    
    def _allocate_threshold(self, capacity_bits: float) -> float:
        """
        Threshold-based allocation: step function based on capacity tiers.
        
        Tiers:
        - Low capacity (< threshold_low): min_compute
        - Medium capacity (threshold_low to threshold_high): base_compute
        - High capacity (> threshold_high): max_compute
        """
        if capacity_bits < self.config.capacity_threshold_low:
            return self.config.min_compute
        elif capacity_bits > self.config.capacity_threshold_high:
            return self.config.max_compute
        else:
            # Linear interpolation between base and max for medium
            range_bits = self.config.capacity_threshold_high - self.config.capacity_threshold_low
            position = (capacity_bits - self.config.capacity_threshold_low) / range_bits
            return self.config.base_compute + position * (self.config.max_compute - self.config.base_compute)
    
    def _allocate_exponential(self, capacity_bits: float) -> float:
        """
        Exponential allocation: compute grows exponentially with capacity.
        
        Formula: compute = base * exp(capacity * factor / threshold_high)
        """
        exponent = capacity_bits * self.config.scaling_factor / self.config.capacity_threshold_high
        return self.config.base_compute * math.exp(exponent)
    
    def _allocate_adaptive(
        self,
        capacity_bits: float,
        task_complexity: Optional[float] = None,
    ) -> float:
        """
        Adaptive allocation: considers both capacity and task complexity.
        
        For complex tasks, even medium-capacity agents get more compute.
        For simple tasks, only high-capacity agents get extra compute.
        """
        base_from_capacity = self._allocate_proportional(capacity_bits)
        
        if task_complexity is None:
            return base_from_capacity
        
        # Adjust based on task complexity
        # Complex tasks (high complexity) need more compute from all agents
        # Simple tasks (low complexity) can be handled by low-capacity agents
        complexity_factor = 0.5 + task_complexity  # Range: 0.5 to 1.5
        
        return base_from_capacity * complexity_factor
    
    def allocate_batch(
        self,
        agents: Dict[str, float],  # agent_id -> capacity_bits
        task_complexity: Optional[float] = None,
        normalize: bool = True,
    ) -> Dict[str, AllocationResult]:
        """
        Allocate compute budget for multiple agents.
        
        Args:
            agents: Dict mapping agent_id to capacity in bits.
            task_complexity: Optional task complexity factor.
            normalize: If True, normalize budgets to sum to max_compute * n_agents.
            
        Returns:
            Dict mapping agent_id to AllocationResult.
        """
        results = {}
        
        for agent_id, capacity_bits in agents.items():
            results[agent_id] = self.allocate(
                agent_id=agent_id,
                capacity_bits=capacity_bits,
                task_complexity=task_complexity,
            )
        
        if normalize and len(results) > 1:
            # Normalize so total compute equals n_agents * base_compute
            total_compute = sum(r.compute_budget for r in results.values())
            target_total = len(results) * self.config.base_compute
            
            if total_compute > 0:
                scale_factor = target_total / total_compute
                for agent_id, result in results.items():
                    result.compute_budget *= scale_factor
                    result.allocation_ratio = result.compute_budget / self.config.base_compute
        
        return results
    
    def rank_by_allocation(
        self,
        agents: Dict[str, float],
        task_complexity: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank agents by their compute allocation (highest first).
        
        Args:
            agents: Dict mapping agent_id to capacity in bits.
            task_complexity: Optional task complexity factor.
            
        Returns:
            List of (agent_id, compute_budget) tuples, sorted descending.
        """
        results = self.allocate_batch(agents, task_complexity, normalize=False)
        
        ranked = [
            (agent_id, result.compute_budget)
            for agent_id, result in results.items()
        ]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_total_compute(self) -> float:
        """Get total compute allocated across all history."""
        return sum(r.compute_budget for r in self.allocation_history)
    
    def get_average_allocation(self) -> float:
        """Get average compute allocation."""
        if not self.allocation_history:
            return 0.0
        return self.get_total_compute() / len(self.allocation_history)
    
    def clear_history(self) -> None:
        """Clear allocation history."""
        self.allocation_history.clear()


class MultiAgentComputePool:
    """
    Manages compute allocation across a pool of agents.
    
    Tracks agent capacities, manages budgets, and ensures
    fair allocation based on capacity rankings.
    """
    
    def __init__(
        self,
        allocator: Optional[CapacityAwareAllocator] = None,
        total_budget: float = 100.0,
    ):
        """
        Initialize the compute pool.
        
        Args:
            allocator: Compute allocator instance.
            total_budget: Total compute budget for the pool.
        """
        self.allocator = allocator or CapacityAwareAllocator()
        self.total_budget = total_budget
        self.agent_capacities: Dict[str, float] = {}
        self.current_allocations: Dict[str, float] = {}
    
    def register_agent(self, agent_id: str, capacity_bits: float) -> None:
        """Register an agent with its capacity."""
        self.agent_capacities[agent_id] = capacity_bits
    
    def update_capacity(self, agent_id: str, capacity_bits: float) -> None:
        """Update an agent's capacity."""
        self.agent_capacities[agent_id] = capacity_bits
    
    def allocate_budget(
        self,
        task_complexity: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Allocate budget across all registered agents.
        
        Returns:
            Dict mapping agent_id to compute budget.
        """
        if not self.agent_capacities:
            return {}
        
        results = self.allocator.allocate_batch(
            agents=self.agent_capacities,
            task_complexity=task_complexity,
            normalize=True,
        )
        
        # Scale to total budget
        total_allocated = sum(r.compute_budget for r in results.values())
        if total_allocated > 0:
            scale = self.total_budget / (total_allocated * len(results))
            self.current_allocations = {
                agent_id: r.compute_budget * scale
                for agent_id, r in results.items()
            }
        else:
            # Equal allocation if no capacity info
            equal_share = self.total_budget / len(self.agent_capacities)
            self.current_allocations = {
                agent_id: equal_share
                for agent_id in self.agent_capacities
            }
        
        return self.current_allocations
    
    def get_agent_budget(self, agent_id: str) -> float:
        """Get the current budget for an agent."""
        return self.current_allocations.get(agent_id, 0.0)
    
    def get_high_capacity_agents(self, threshold: float = 5.0) -> List[str]:
        """Get list of agents with capacity above threshold."""
        return [
            agent_id for agent_id, capacity in self.agent_capacities.items()
            if capacity >= threshold
        ]


def create_allocator(
    base_compute: float = 1.0,
    max_compute: float = 10.0,
    strategy: str = "capacity_proportional",
    **kwargs,
) -> CapacityAwareAllocator:
    """
    Factory function to create a capacity-aware allocator.
    
    Args:
        base_compute: Base compute budget.
        max_compute: Maximum compute budget.
        strategy: Allocation strategy name.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured CapacityAwareAllocator instance.
    """
    strategy_enum = AllocationStrategy(strategy)
    
    config = AllocatorConfig(
        base_compute=base_compute,
        max_compute=max_compute,
        strategy=strategy_enum,
        **kwargs,
    )
    
    return CapacityAwareAllocator(config=config)
