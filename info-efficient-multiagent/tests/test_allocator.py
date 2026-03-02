"""
Tests for Capacity-Aware Compute Allocator
"""

import pytest
import math
from unittest.mock import Mock

from src.allocator import (
    AllocationStrategy,
    AllocationResult,
    AllocatorConfig,
    CapacityAwareAllocator,
    MultiAgentComputePool,
    create_allocator,
)


class TestAllocatorConfig:
    """Tests for AllocatorConfig."""
    
    def test_init_defaults(self):
        """Test default configuration."""
        config = AllocatorConfig()
        
        assert config.base_compute == 1.0
        assert config.max_compute == 10.0
        assert config.min_compute == 0.5
        assert config.capacity_threshold_low == 2.0
        assert config.capacity_threshold_high == 8.0
        assert config.strategy == AllocationStrategy.CAPACITY_PROPORTIONAL
    
    def test_init_custom(self):
        """Test custom configuration."""
        config = AllocatorConfig(
            base_compute=2.0,
            max_compute=20.0,
            strategy=AllocationStrategy.CAPACITY_THRESHOLD,
        )
        
        assert config.base_compute == 2.0
        assert config.max_compute == 20.0
        assert config.strategy == AllocationStrategy.CAPACITY_THRESHOLD
    
    def test_to_dict(self):
        """Test serialization."""
        config = AllocatorConfig(base_compute=3.0)
        d = config.to_dict()
        
        assert d["base_compute"] == 3.0
        assert "strategy" in d


class TestAllocationResult:
    """Tests for AllocationResult."""
    
    def test_init(self):
        """Test result initialization."""
        result = AllocationResult(
            agent_id="agent-1",
            compute_budget=5.0,
            capacity_bits=4.0,
            allocation_ratio=1.25,
            strategy="capacity_proportional",
        )
        
        assert result.agent_id == "agent-1"
        assert result.compute_budget == 5.0
        assert result.capacity_bits == 4.0
        assert result.allocation_ratio == 1.25
    
    def test_metadata_default(self):
        """Test metadata defaults to empty dict."""
        result = AllocationResult(
            agent_id="agent-1",
            compute_budget=5.0,
            capacity_bits=4.0,
            allocation_ratio=1.25,
            strategy="test",
        )
        
        assert result.metadata == {}


class TestCapacityAwareAllocator:
    """Tests for CapacityAwareAllocator."""
    
    def test_init(self):
        """Test allocator initialization."""
        allocator = CapacityAwareAllocator()
        
        assert allocator.config is not None
        assert allocator.allocation_history == []
    
    def test_allocate_proportional_low_capacity(self):
        """Test proportional allocation with low capacity."""
        config = AllocatorConfig(strategy=AllocationStrategy.CAPACITY_PROPORTIONAL)
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=1.0)
        
        # Low capacity should get less than base * 2
        assert result.compute_budget < 2.0
        assert result.agent_id == "agent-1"
    
    def test_allocate_proportional_high_capacity(self):
        """Test proportional allocation with high capacity."""
        config = AllocatorConfig(strategy=AllocationStrategy.CAPACITY_PROPORTIONAL)
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=10.0)
        
        # High capacity should get more compute
        assert result.compute_budget > 1.0
    
    def test_allocate_threshold_low(self):
        """Test threshold allocation - low tier."""
        config = AllocatorConfig(
            strategy=AllocationStrategy.CAPACITY_THRESHOLD,
            capacity_threshold_low=2.0,
            capacity_threshold_high=8.0,
            min_compute=0.5,
            base_compute=2.0,
            max_compute=10.0,
        )
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=1.0)
        
        # Below threshold_low should get min_compute
        assert result.compute_budget == 0.5
    
    def test_allocate_threshold_medium(self):
        """Test threshold allocation - medium tier."""
        config = AllocatorConfig(
            strategy=AllocationStrategy.CAPACITY_THRESHOLD,
            capacity_threshold_low=2.0,
            capacity_threshold_high=8.0,
            min_compute=0.5,
            base_compute=2.0,
            max_compute=10.0,
        )
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=5.0)
        
        # Between thresholds should interpolate
        assert 2.0 <= result.compute_budget <= 10.0
    
    def test_allocate_threshold_high(self):
        """Test threshold allocation - high tier."""
        config = AllocatorConfig(
            strategy=AllocationStrategy.CAPACITY_THRESHOLD,
            capacity_threshold_low=2.0,
            capacity_threshold_high=8.0,
            min_compute=0.5,
            base_compute=2.0,
            max_compute=10.0,
        )
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=10.0)
        
        # Above threshold_high should get max_compute
        assert result.compute_budget == 10.0
    
    def test_allocate_exponential(self):
        """Test exponential allocation."""
        config = AllocatorConfig(
            strategy=AllocationStrategy.CAPACITY_EXPONENTIAL,
            capacity_threshold_high=8.0,
        )
        allocator = CapacityAwareAllocator(config)
        
        result_low = allocator.allocate("agent-1", capacity_bits=1.0)
        result_high = allocator.allocate("agent-2", capacity_bits=6.0)
        
        # Higher capacity should get exponentially more
        assert result_high.compute_budget > result_low.compute_budget
    
    def test_allocate_adaptive_with_complexity(self):
        """Test adaptive allocation with task complexity."""
        config = AllocatorConfig(strategy=AllocationStrategy.ADAPTIVE)
        allocator = CapacityAwareAllocator(config)
        
        result_simple = allocator.allocate("agent-1", capacity_bits=5.0, task_complexity=0.2)
        result_complex = allocator.allocate("agent-2", capacity_bits=5.0, task_complexity=0.9)
        
        # Complex task should get more compute for same capacity
        assert result_complex.compute_budget > result_simple.compute_budget
    
    def test_allocate_respects_max_compute(self):
        """Test that allocation respects max_compute bound."""
        config = AllocatorConfig(
            max_compute=5.0,
            strategy=AllocationStrategy.CAPACITY_PROPORTIONAL,
        )
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=100.0)
        
        assert result.compute_budget <= 5.0
    
    def test_allocate_respects_min_compute(self):
        """Test that allocation respects min_compute bound."""
        config = AllocatorConfig(
            min_compute=1.0,
            strategy=AllocationStrategy.CAPACITY_PROPORTIONAL,
        )
        allocator = CapacityAwareAllocator(config)
        
        result = allocator.allocate("agent-1", capacity_bits=0.1)
        
        assert result.compute_budget >= 1.0
    
    def test_allocation_history(self):
        """Test that allocations are tracked in history."""
        allocator = CapacityAwareAllocator()
        
        allocator.allocate("agent-1", 5.0)
        allocator.allocate("agent-2", 3.0)
        
        assert len(allocator.allocation_history) == 2
    
    def test_allocate_batch(self):
        """Test batch allocation."""
        allocator = CapacityAwareAllocator()
        
        agents = {
            "agent-1": 8.0,
            "agent-2": 4.0,
            "agent-3": 2.0,
        }
        
        results = allocator.allocate_batch(agents)
        
        assert len(results) == 3
        assert "agent-1" in results
        assert all(isinstance(r, AllocationResult) for r in results.values())
    
    def test_allocate_batch_normalize(self):
        """Test batch allocation with normalization."""
        allocator = CapacityAwareAllocator()
        
        agents = {
            "agent-1": 8.0,
            "agent-2": 4.0,
        }
        
        results = allocator.allocate_batch(agents, normalize=True)
        
        # Normalized so total = n_agents * base_compute
        total = sum(r.compute_budget for r in results.values())
        assert abs(total - 2.0) < 0.1  # 2 agents * base_compute of 1.0
    
    def test_rank_by_allocation(self):
        """Test ranking agents by allocation."""
        allocator = CapacityAwareAllocator()
        
        agents = {
            "low": 2.0,
            "high": 10.0,
            "medium": 5.0,
        }
        
        ranked = allocator.rank_by_allocation(agents)
        
        # Should be sorted descending by compute
        assert ranked[0][0] == "high"
        assert ranked[-1][0] == "low"
    
    def test_get_total_compute(self):
        """Test getting total compute."""
        allocator = CapacityAwareAllocator()
        
        allocator.allocate("agent-1", 5.0)
        allocator.allocate("agent-2", 3.0)
        
        total = allocator.get_total_compute()
        assert total > 0
    
    def test_get_average_allocation(self):
        """Test getting average allocation."""
        allocator = CapacityAwareAllocator()
        
        allocator.allocate("agent-1", 5.0)
        allocator.allocate("agent-2", 5.0)
        
        avg = allocator.get_average_allocation()
        assert avg > 0
    
    def test_clear_history(self):
        """Test clearing allocation history."""
        allocator = CapacityAwareAllocator()
        
        allocator.allocate("agent-1", 5.0)
        allocator.clear_history()
        
        assert len(allocator.allocation_history) == 0


class TestMultiAgentComputePool:
    """Tests for MultiAgentComputePool."""
    
    def test_init(self):
        """Test pool initialization."""
        pool = MultiAgentComputePool()
        
        assert pool.allocator is not None
        assert pool.agent_capacities == {}
    
    def test_register_agent(self):
        """Test registering an agent."""
        pool = MultiAgentComputePool()
        
        pool.register_agent("agent-1", 5.0)
        
        assert "agent-1" in pool.agent_capacities
        assert pool.agent_capacities["agent-1"] == 5.0
    
    def test_update_capacity(self):
        """Test updating agent capacity."""
        pool = MultiAgentComputePool()
        
        pool.register_agent("agent-1", 5.0)
        pool.update_capacity("agent-1", 8.0)
        
        assert pool.agent_capacities["agent-1"] == 8.0
    
    def test_allocate_budget(self):
        """Test budget allocation across pool."""
        pool = MultiAgentComputePool(total_budget=100.0)
        
        pool.register_agent("agent-1", 8.0)
        pool.register_agent("agent-2", 4.0)
        pool.register_agent("agent-3", 2.0)
        
        allocations = pool.allocate_budget()
        
        assert len(allocations) == 3
        # Higher capacity should get more budget
        assert allocations["agent-1"] > allocations["agent-3"]
    
    def test_get_agent_budget(self):
        """Test getting agent's budget."""
        pool = MultiAgentComputePool()
        
        pool.register_agent("agent-1", 5.0)
        pool.allocate_budget()
        
        budget = pool.get_agent_budget("agent-1")
        assert budget > 0
    
    def test_get_agent_budget_unknown(self):
        """Test getting budget for unknown agent."""
        pool = MultiAgentComputePool()
        
        budget = pool.get_agent_budget("unknown")
        assert budget == 0.0
    
    def test_get_high_capacity_agents(self):
        """Test getting high capacity agents."""
        pool = MultiAgentComputePool()
        
        pool.register_agent("low", 2.0)
        pool.register_agent("high", 8.0)
        pool.register_agent("medium", 5.0)
        
        high_agents = pool.get_high_capacity_agents(threshold=5.0)
        
        assert "high" in high_agents
        assert "medium" in high_agents
        assert "low" not in high_agents


class TestFactoryFunction:
    """Tests for create_allocator factory."""
    
    def test_create_allocator_defaults(self):
        """Test creating allocator with defaults."""
        allocator = create_allocator()
        
        assert allocator.config.base_compute == 1.0
        assert allocator.config.strategy == AllocationStrategy.CAPACITY_PROPORTIONAL
    
    def test_create_allocator_custom(self):
        """Test creating allocator with custom config."""
        allocator = create_allocator(
            base_compute=2.0,
            max_compute=20.0,
            strategy="capacity_threshold",
        )
        
        assert allocator.config.base_compute == 2.0
        assert allocator.config.max_compute == 20.0
        assert allocator.config.strategy == AllocationStrategy.CAPACITY_THRESHOLD
    
    def test_create_allocator_invalid_strategy(self):
        """Test creating allocator with invalid strategy."""
        with pytest.raises(ValueError):
            create_allocator(strategy="invalid_strategy")


class TestAllocationComparison:
    """Tests comparing different allocation strategies."""
    
    def test_high_capacity_gets_more_than_low(self):
        """Test that high-capacity agents get more compute."""
        allocator = CapacityAwareAllocator()
        
        high_result = allocator.allocate("high", capacity_bits=10.0)
        low_result = allocator.allocate("low", capacity_bits=1.0)
        
        assert high_result.compute_budget > low_result.compute_budget
    
    def test_inverse_of_uncertainty_based(self):
        """Test that capacity-based allocation is inverse of uncertainty-based.
        
        In uncertainty-based (CATTS), high uncertainty = more compute.
        In capacity-based, high capacity = more compute.
        """
        config = AllocatorConfig(strategy=AllocationStrategy.CAPACITY_PROPORTIONAL)
        allocator = CapacityAwareAllocator(config)
        
        # High capacity (low uncertainty) -> MORE compute
        high_cap = allocator.allocate("agent-1", capacity_bits=10.0)
        
        # Low capacity (high uncertainty) -> LESS compute
        low_cap = allocator.allocate("agent-2", capacity_bits=1.0)
        
        # This is the inverse of uncertainty-based scaling
        assert high_cap.compute_budget > low_cap.compute_budget
    
    def test_strategies_produce_different_results(self):
        """Test that different strategies produce different allocations."""
        capacity = 5.0
        
        results = {}
        for strategy in AllocationStrategy:
            allocator = create_allocator(strategy=strategy.value)
            results[strategy.value] = allocator.allocate("agent-1", capacity).compute_budget
        
        # At least some strategies should produce different values
        unique_values = set(results.values())
        assert len(unique_values) > 1
