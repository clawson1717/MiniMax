import pytest
import numpy as np
from src.diffusion import DiffusionPolicy

def test_diffusion_policy_initialization():
    policy = DiffusionPolicy(action_dim=2, horizon=4)
    assert policy.action_dim == 2
    assert policy.horizon == 4
    assert policy.num_diffusion_steps == 5

def test_sample_action_shape():
    action_dim = 3
    horizon = 2
    policy = DiffusionPolicy(action_dim=action_dim, horizon=horizon)
    context = {"state": "test_state"}
    
    action = policy.sample_action(context)
    
    assert action.shape == (horizon, action_dim)
    assert isinstance(action, np.ndarray)

def test_sample_action_reproducibility():
    # Since it's random, we just check it runs without error for now
    # and produces different results if noise is significant
    policy = DiffusionPolicy()
    context = {}
    
    action1 = policy.sample_action(context)
    action2 = policy.sample_action(context)
    
    assert not np.array_equal(action1, action2)

def test_online_update_placeholder():
    policy = DiffusionPolicy()
    # Should not raise error
    policy.update_online({"dummy": "data"})
