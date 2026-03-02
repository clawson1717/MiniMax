import pytest
from src.allocator import ComputeAllocator

def test_should_scale():
    allocator = ComputeAllocator(threshold=0.5)
    
    assert allocator.should_scale(0.3) is False
    assert allocator.should_scale(0.5) is False
    assert allocator.should_scale(0.6) is True
    assert allocator.should_scale(1.0) is True

def test_allocate_no_scaling():
    allocator = ComputeAllocator(threshold=0.5)
    base = 10
    
    # Uncertainty below threshold
    assert allocator.allocate(base, 0.4) == 10
    # Uncertainty at threshold
    assert allocator.allocate(base, 0.5) == 10

def test_allocate_scaling():
    allocator = ComputeAllocator(threshold=0.5)
    base = 10
    
    # Uncertainty halfway between threshold (0.5) and max (1.0) -> 0.75
    # multiplier = 1 + (0.75 - 0.5) / (1.0 - 0.5) = 1 + 0.25 / 0.5 = 1.5
    # scaled = 10 * 1.5 = 15
    assert allocator.allocate(base, 0.75) == 15
    
    # Uncertainty at max (1.0)
    # multiplier = 1 + (1.0 - 0.5) / 1.0 - 0.5 = 1 + 1 = 2
    # scaled = 10 * 2 = 20
    assert allocator.allocate(base, 1.0) == 20

def test_allocate_from_votes():
    allocator = ComputeAllocator(threshold=0.3)
    base = 5
    
    # All votes same -> uncertainty 0.0
    votes_uniform = ["A", "A", "A"]
    assert allocator.allocate_from_votes(base, votes_uniform) == 5
    
    # All votes different -> uncertainty 1.0
    # multiplier = 1 + (1.0 - 0.3) / (1.0 - 0.3) = 2.0
    # scaled = 5 * 2 = 10
    votes_diverse = ["A", "B", "C"]
    assert allocator.allocate_from_votes(base, votes_diverse) == 10

def test_custom_threshold():
    allocator = ComputeAllocator(threshold=0.8)
    base = 10
    
    assert allocator.should_scale(0.7) is False
    assert allocator.allocate(base, 0.7) == 10
    
    # Uncertainty 0.9 is halfway between 0.8 and 1.0
    # Multiplier = 1.5
    assert allocator.allocate(base, 0.9) == 15
