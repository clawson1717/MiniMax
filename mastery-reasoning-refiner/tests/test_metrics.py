import torch
import pytest
from src.metrics import InformationTheoreticMetrics

def test_world_model_capacity_uniform():
    # Uniform distribution = max entropy = 0 bits of specific info
    # 4 actions, max entropy = log2(4) = 2.0
    policy_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    capacity = InformationTheoreticMetrics.calculate_world_model_capacity(policy_probs, env_states=10)
    assert capacity == pytest.approx(0.0, abs=1e-5)

def test_world_model_capacity_deterministic():
    # Deterministic distribution = 0 entropy = max info bits
    # 4 actions, max entropy = 2.0. Info per state = 2.0 - 0 = 2.0.
    # 10 states * 2.0 = 20 bits
    policy_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    capacity = InformationTheoreticMetrics.calculate_world_model_capacity(policy_probs, env_states=10)
    assert capacity == pytest.approx(20.0, abs=1e-5)

def test_semantic_information_gain():
    initial = "The cat sat."
    refined = "The calico cat sat comfortably on the velvet mat."
    
    gain = InformationTheoreticMetrics.calculate_semantic_information_gain(initial, refined)
    
    # Refined is longer and more complex, should have positive gain
    assert gain > 0
    
    # Same string should have zero or near-zero gain (depending on normalization)
    gain_same = InformationTheoreticMetrics.calculate_semantic_information_gain(initial, initial)
    assert gain_same == pytest.approx(0.0)

def test_semantic_information_gain_empty():
    assert InformationTheoreticMetrics.calculate_semantic_information_gain("", "something") == 0.0
    assert InformationTheoreticMetrics.calculate_semantic_information_gain("something", "") == 0.0
