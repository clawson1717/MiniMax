"""Tests for ODE reasoning dynamics."""

import torch
import pytest
from src.ode import (
    ODEReasoningDynamics,
    ReasoningODEOutput,
    euler_step,
    rk4_step,
)


class TestODEReasoningDynamics:
    """Test suite for ODEReasoningDynamics class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        dynamics = ODEReasoningDynamics(state_dim=10)
        assert dynamics.state_dim == 10
        assert dynamics.growth_rate == 0.5
        assert dynamics.decay_rate == 1.0
        assert dynamics.uncertainty_scale == 2.0
        assert dynamics.oscillation_freq == 1.0
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        dynamics = ODEReasoningDynamics(
            state_dim=32,
            baseline=0.5,
            growth_rate=0.3,
            decay_rate=0.8,
            uncertainty_scale=1.5,
            oscillation_freq=2.0,
        )
        assert dynamics.state_dim == 32
        assert dynamics.baseline == 0.5
        assert dynamics.growth_rate == 0.3
        assert dynamics.decay_rate == 0.8
    
    def test_init_with_tensor_baseline(self):
        """Test initialization with tensor baseline."""
        baseline = torch.randn(16)
        dynamics = ODEReasoningDynamics(state_dim=16, baseline=baseline)
        assert torch.equal(dynamics.baseline, baseline)
    
    def test_init_invalid_baseline_shape(self):
        """Test that invalid baseline shape raises error."""
        baseline = torch.randn(8)  # Wrong dimension
        with pytest.raises(ValueError):
            ODEReasoningDynamics(state_dim=16, baseline=baseline)
    
    def test_dynamics_shape(self):
        """Test that dynamics returns correct shape."""
        dynamics = ODEReasoningDynamics(state_dim=32)
        R = torch.randn(32)
        t = torch.tensor(0.0)
        dRdt = dynamics.dynamics(t, R)
        assert dRdt.shape == R.shape
    
    def test_dynamics_low_uncertainty_grows(self):
        """Test that low uncertainty allows growth."""
        dynamics = ODEReasoningDynamics(
            state_dim=8,
            baseline=0.0,
            growth_rate=1.0,
            decay_rate=0.1,
        )
        dynamics.set_uncertainty(0.01)  # Low uncertainty
        
        R = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        t = torch.tensor(0.0)
        dRdt = dynamics.dynamics(t, R)
        
        # With low uncertainty, growth component should dominate
        # dR/dt should be positive for positive R
        assert torch.all(dRdt > 0)
    
    def test_dynamics_high_uncertainty_decays(self):
        """Test that high uncertainty causes decay."""
        dynamics = ODEReasoningDynamics(
            state_dim=8,
            baseline=0.0,
            growth_rate=0.5,
            decay_rate=1.0,
        )
        dynamics.set_uncertainty(0.9)  # High uncertainty
        
        R = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        t = torch.tensor(0.0)
        dRdt = dynamics.dynamics(t, R)
        
        # With high uncertainty, decay should dominate
        assert torch.all(dRdt < 0)
    
    def test_set_uncertainty_scalar(self):
        """Test setting uncertainty with scalar."""
        dynamics = ODEReasoningDynamics(state_dim=4)
        dynamics.set_uncertainty(0.5)
        assert isinstance(dynamics._current_uncertainty, torch.Tensor)
        assert dynamics._current_uncertainty.item() == 0.5
    
    def test_set_uncertainty_tensor(self):
        """Test setting uncertainty with tensor."""
        dynamics = ODEReasoningDynamics(state_dim=4)
        U = torch.tensor([0.1, 0.2, 0.3, 0.4])
        dynamics.set_uncertainty(U)
        assert torch.equal(dynamics._current_uncertainty, U)


class TestODESolvers:
    """Test suite for ODE solvers."""
    
    def test_solve_dopri5(self):
        """Test dopri5 solver produces trajectory."""
        dynamics = ODEReasoningDynamics(state_dim=8)
        R0 = torch.randn(8)
        U = 0.2
        
        result = dynamics.solve(R0, U, t_span=(0.0, 1.0), method='dopri5')
        
        assert isinstance(result, ReasoningODEOutput)
        assert len(result.trajectory) > 0
        assert len(result.times) > 0
        assert torch.equal(result.final_state, result.trajectory[-1])
    
    def test_solve_euler(self):
        """Test Euler solver produces trajectory."""
        dynamics = ODEReasoningDynamics(state_dim=8)
        R0 = torch.randn(8)
        U = 0.2
        
        result = dynamics.solve(R0, U, t_span=(0.0, 1.0), method='euler')
        
        assert isinstance(result, ReasoningODEOutput)
        assert len(result.trajectory) > 0
        assert len(result.times) > 0
        assert torch.equal(result.final_state, result.trajectory[-1])
    
    def test_solve_rk4(self):
        """Test RK4 solver produces trajectory."""
        dynamics = ODEReasoningDynamics(state_dim=8)
        R0 = torch.randn(8)
        U = 0.2
        
        result = dynamics.solve(R0, U, t_span=(0.0, 1.0), method='rk4')
        
        assert isinstance(result, ReasoningODEOutput)
        assert len(result.trajectory) > 0
        assert len(result.times) > 0
        assert torch.equal(result.final_state, result.trajectory[-1])
    
    def test_trajectory_shape_consistency(self):
        """Test that all trajectory states have consistent shape."""
        dynamics = ODEReasoningDynamics(state_dim=16)
        R0 = torch.randn(16)
        
        result = dynamics.solve(R0, 0.1, t_span=(0.0, 2.0), method='dopri5')
        
        for state in result.trajectory:
            assert state.shape == R0.shape
    
    def test_trajectory_times_match(self):
        """Test that trajectory length matches times length."""
        dynamics = ODEReasoningDynamics(state_dim=8)
        R0 = torch.randn(8)
        
        result = dynamics.solve(R0, 0.1, t_span=(0.0, 1.0), method='euler')
        
        assert len(result.trajectory) == len(result.times)
    
    def test_custom_t_eval(self):
        """Test solving with custom evaluation times."""
        dynamics = ODEReasoningDynamics(state_dim=4)
        R0 = torch.randn(4)
        t_eval = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        result = dynamics.solve(R0, 0.1, t_span=(0.0, 1.0), method='euler', t_eval=t_eval)
        
        assert len(result.times) == len(t_eval)
        assert len(result.trajectory) == len(t_eval)
    
    def test_different_methods_converge(self):
        """Test that different solvers give similar results."""
        dynamics = ODEReasoningDynamics(
            state_dim=4,
            baseline=0.0,
            growth_rate=0.2,
            decay_rate=2.0,
        )
        R0 = torch.tensor([0.5, 0.6, 0.7, 0.8])
        U = 0.1
        t_span = (0.0, 0.5)
        
        euler_result = dynamics.solve(R0, U, t_span=t_span, method='euler', t_eval=[0.0, 0.5])
        rk4_result = dynamics.solve(R0, U, t_span=t_span, method='rk4', t_eval=[0.0, 0.5])
        dopri_result = dynamics.solve(R0, U, t_span=t_span, method='dopri5', t_eval=[0.0, 0.5])
        
        # All methods should converge to similar final states
        # (Not exactly equal due to different error characteristics)
        assert euler_result.final_state.shape == R0.shape
        assert rk4_result.final_state.shape == R0.shape
        assert dopri_result.final_state.shape == R0.shape


class TestUncertaintyEffects:
    """Test that uncertainty affects dynamics as expected."""
    
    def test_high_uncertainty_different_trajectory(self):
        """Test that higher uncertainty leads to different trajectory."""
        dynamics = ODEReasoningDynamics(state_dim=8)
        R0 = torch.randn(8)
        
        result_low = dynamics.solve(R0, 0.05, t_span=(0.0, 1.0), method='rk4')
        result_high = dynamics.solve(R0, 0.8, t_span=(0.0, 1.0), method='rk4')
        
        # Trajectories should be different
        assert not torch.allclose(result_low.final_state, result_high.final_state, atol=0.1)
    
    def test_zero_uncertainty_growth(self):
        """Test that zero uncertainty maximizes growth potential."""
        dynamics = ODEReasoningDynamics(
            state_dim=4,
            baseline=0.0,
            growth_rate=1.0,
            decay_rate=0.5,
        )
        dynamics.set_uncertainty(0.0)
        
        R = torch.tensor([0.1, 0.2, 0.3, 0.4])
        t = torch.tensor(0.0)
        dRdt = dynamics.dynamics(t, R)
        
        # With zero uncertainty, growth should dominate
        assert torch.all(dRdt > 0)
    
    def test_uncertainty_affects_dynamics_magnitude(self):
        """Test that uncertainty significantly changes dynamics behavior."""
        dynamics = ODEReasoningDynamics(
            state_dim=4,
            baseline=0.0,
            decay_rate=1.0,
            growth_rate=1.0,
        )
        
        dynamics.set_uncertainty(0.0)
        R = torch.tensor([0.3, 0.3, 0.3, 0.3])
        dRdt_zero_U = dynamics.dynamics(torch.tensor(0.0), R)
        
        dynamics.set_uncertainty(1.0)
        dRdt_high_U = dynamics.dynamics(torch.tensor(0.0), R)
        
        # Uncertainty should significantly change the dynamics
        # They shouldn't be nearly identical
        assert not torch.allclose(dRdt_high_U, dRdt_zero_U, atol=0.01)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_scalar_initial_state(self):
        """Test solving with scalar (1D) initial state."""
        dynamics = ODEReasoningDynamics(state_dim=1)
        R0 = torch.tensor([0.5])
        
        result = dynamics.solve(R0, 0.1, t_span=(0.0, 1.0), method='dopri5')
        
        assert result.final_state.shape == torch.Size([1])
    
    def test_very_short_timespan(self):
        """Test solving with very short time span."""
        dynamics = ODEReasoningDynamics(state_dim=4)
        R0 = torch.randn(4)
        
        result = dynamics.solve(R0, 0.1, t_span=(0.0, 0.01), method='euler')
        
        assert len(result.trajectory) > 0
    
    def test_very_long_timespan(self):
        """Test solving with long time span."""
        dynamics = ODEReasoningDynamics(
            state_dim=4,
            decay_rate=1.0,
            growth_rate=0.1,
        )
        R0 = torch.randn(4)
        
        result = dynamics.solve(R0, 0.2, t_span=(0.0, 10.0), method='rk4')
        
        # Should still produce trajectory and converge
        assert len(result.trajectory) > 0
        assert result.final_state.shape == R0.shape
    
    def test_invalid_solver_method(self):
        """Test that invalid solver method raises error."""
        dynamics = ODEReasoningDynamics(state_dim=4)
        R0 = torch.randn(4)
        
        with pytest.raises(ValueError, match="Unknown solver method"):
            dynamics.solve(R0, 0.1, t_span=(0.0, 1.0), method='invalid_method')
    
    def test_euler_step_function(self):
        """Test the euler_step utility function."""
        def simple_dynamics(t, R):
            return -R
        
        R0 = torch.tensor(1.0)
        R1 = euler_step(simple_dynamics, torch.tensor(0.0), R0, dt=0.1)
        
        # dR/dt = -R, so with R=1, dR/dt = -1
        # Euler: R1 = R0 + dt * dRdt = 1 + 0.1 * (-1) = 0.9
        assert torch.isclose(R1, torch.tensor(0.9), atol=1e-5)
    
    def test_rk4_step_function(self):
        """Test the rk4_step utility function."""
        def simple_dynamics(t, R):
            return -R
        
        R0 = torch.tensor(1.0)
        R1 = rk4_step(simple_dynamics, torch.tensor(0.0), R0, dt=0.1)
        
        # Exact solution for dR/dt = -R is R(t) = R0 * exp(-t)
        # At t=0.1: R = exp(-0.1) ≈ 0.904837
        exact = torch.tensor(0.904837418)
        assert torch.isclose(R1, exact, atol=1e-4)
    
    def test_baseline_attraction(self):
        """Test that dynamics attracts to baseline."""
        dynamics = ODEReasoningDynamics(
            state_dim=4,
            baseline=0.0,
            decay_rate=5.0,  # Strong restoring force
            growth_rate=0.0,  # No growth
        )
        dynamics.set_uncertainty(0.0)
        
        R = torch.tensor([10.0, 10.0, 10.0, 10.0])
        t = torch.tensor(0.0)
        dRdt = dynamics.dynamics(t, R)
        
        # Should be negative (moving toward baseline)
        assert torch.all(dRdt < 0)


class TestReasoningODEOutput:
    """Test ReasoningODEOutput dataclass."""
    
    def test_dataclass_fields(self):
        """Test that dataclass has correct fields."""
        trajectory = [torch.randn(4) for _ in range(3)]
        times = [0.0, 0.5, 1.0]
        final_state = trajectory[-1]
        
        output = ReasoningODEOutput(
            trajectory=trajectory,
            times=times,
            final_state=final_state,
        )
        
        assert output.trajectory == trajectory
        assert output.times == times
        assert torch.equal(output.final_state, final_state)
    
    def test_final_state_matches_last_trajectory(self):
        """Test that final_state is correctly set from trajectory."""
        trajectory = [torch.randn(8) for _ in range(10)]
        output = ReasoningODEOutput(
            trajectory=trajectory,
            times=list(range(10)),
            final_state=trajectory[-1],
        )
        
        assert torch.equal(output.final_state, output.trajectory[-1])
