"""ODE-based reasoning dynamics for continuous-time reasoning."""

from dataclasses import dataclass
from typing import List, Literal, Union
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


@dataclass
class ReasoningODEOutput:
    """
    Output from ODE-based reasoning dynamics simulation.
    
    Attributes:
        trajectory: List of reasoning state tensors at each time point
        times: List of timestamps corresponding to each trajectory point
        final_state: The final reasoning state tensor
    """
    trajectory: List[torch.Tensor]
    times: List[float]
    final_state: torch.Tensor


class ODEReasoningDynamics(nn.Module):
    """
    ODE-based dynamics for reasoning state evolution.
    
    Models how reasoning evolves over time given uncertainty:
    - dR/dt = f(R, U)
    - Grows when reasoning is consistent (low uncertainty)
    - Decays or oscillates when uncertainty is high
    - Has a stable equilibrium at baseline reasoning state
    
    Args:
        state_dim: Dimension of the reasoning state vector
        baseline: Baseline/equilibrium reasoning state (default: zeros)
        growth_rate: Base growth rate for low uncertainty (default: 0.5)
        decay_rate: Base decay rate (default: 1.0)
        uncertainty_scale: Scale factor for uncertainty effect (default: 2.0)
        oscillation_freq: Frequency for oscillation component (default: 1.0)
    """
    
    def __init__(
        self,
        state_dim: int,
        baseline: Union[float, torch.Tensor] = 0.0,
        growth_rate: float = 0.5,
        decay_rate: float = 1.0,
        uncertainty_scale: float = 2.0,
        oscillation_freq: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.uncertainty_scale = uncertainty_scale
        self.oscillation_freq = oscillation_freq
        
        # Handle baseline
        if isinstance(baseline, (int, float)):
            self.baseline = baseline
        else:
            self.baseline = baseline
            if baseline.shape != (state_dim,):
                raise ValueError(f"Baseline shape {baseline.shape} must match state_dim {state_dim}")
    
    def _get_baseline_tensor(self, R: torch.Tensor) -> torch.Tensor:
        """Get baseline as a tensor matching R's shape."""
        if isinstance(self.baseline, torch.Tensor):
            return self.baseline.to(R.device)
        return torch.full_like(R, self.baseline)
    
    def dynamics(self, t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Compute the dynamics dR/dt = f(R, U).
        
        The dynamics function models:
        - Attraction toward baseline state
        - Growth when reasoning is consistent (low uncertainty)
        - Decay/oscillation when uncertainty is high
        
        Args:
            t: Current time (scalar tensor)
            R: Current reasoning state tensor
            
        Returns:
            dR/dt: Derivative of reasoning state
        """
        # Get uncertainty - use the stored uncertainty if available
        # For ODE solving, we need to access this from the module's registered buffers
        U = getattr(self, '_current_uncertainty', torch.tensor(0.1))
        
        # Convert U to tensor if needed
        if not isinstance(U, torch.Tensor):
            U = torch.tensor(U, dtype=R.dtype, device=R.device)
        
        # Ensure U is scalar-like (expand to match R if needed)
        if U.numel() == 1:
            U = U.expand_as(R)
        
        # Compute deviation from baseline
        baseline = self._get_baseline_tensor(R)
        deviation = R - baseline
        
        # Dynamics components:
        # 1. Restoring force toward baseline ( Hooke's law )
        restoring = -self.decay_rate * deviation
        
        # 2. Growth component - positive feedback that drives growth
        # This is modulated by inverse uncertainty (low uncertainty = more growth)
        uncertainty_factor = torch.exp(-self.uncertainty_scale * U)
        growth = self.growth_rate * uncertainty_factor * (1.0 - torch.abs(deviation))
        
        # 3. Oscillation component - high uncertainty causes oscillation
        oscillation = self.oscillation_freq * torch.sin(R)
        
        # Combine: dR/dt = restoring + growth - oscillation*damping
        damping = self.uncertainty_scale * U
        dRdt = restoring + growth - damping * oscillation
        
        return dRdt
    
    def set_uncertainty(self, U: Union[float, torch.Tensor]):
        """Set the uncertainty for dynamics evaluation."""
        if isinstance(U, torch.Tensor):
            self._current_uncertainty = U.detach().clone()
        else:
            self._current_uncertainty = torch.tensor(U)
    
    def solve(
        self,
        R0: torch.Tensor,
        U: Union[float, torch.Tensor],
        t_span: tuple,
        method: Literal['euler', 'rk4', 'dopri5'] = 'dopri5',
        t_eval: Union[list, torch.Tensor, None] = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ) -> ReasoningODEOutput:
        """
        Solve the ODE for reasoning dynamics.
        
        Args:
            R0: Initial reasoning state tensor
            U: Uncertainty tensor or scalar
            t_span: Tuple of (t_start, t_end)
            method: Solver method - 'euler', 'rk4', or 'dopri5'
            t_eval: Times at which to evaluate the solution (optional)
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            
        Returns:
            ReasoningODEOutput with trajectory, times, and final_state
        """
        self.set_uncertainty(U)
        
        # Ensure R0 is a tensor with gradient
        if not isinstance(R0, torch.Tensor):
            R0 = torch.tensor(R0, dtype=torch.float32)
        R0 = R0.float().requires_grad_(True)
        
        # Time span
        t_start, t_end = t_span
        
        # Map method names to torchdiffeq options
        solver_options = {}
        if method == 'euler':
            solver_options['method'] = 'euler'
        elif method == 'rk4':
            solver_options['method'] = 'midpoint'
            # Note: torchdiffeq doesn't have explicit rk4, midpoint is a reasonable substitute
            # for explicit control; we implement proper RK4 below if needed
        elif method == 'dopri5':
            solver_options['method'] = 'dopri5'
            solver_options['rtol'] = rtol
            solver_options['atol'] = atol
        else:
            raise ValueError(f"Unknown solver method: {method}. Use 'euler', 'rk4', or 'dopri5'.")
        
        # For proper RK4, we need custom implementation
        if method == 'rk4':
            trajectory, times = self._solve_rk4(R0, t_start, t_end, t_eval)
        else:
            # Use torchdiffeq for euler and dopri5
            if t_eval is not None:
                if not isinstance(t_eval, torch.Tensor):
                    t_eval = torch.tensor(t_eval, dtype=torch.float32)
            else:
                # Default time points
                t_eval = torch.linspace(t_start, t_end, 100)
            
            # Make R0 batched if needed for torchdiffeq
            R0_batch = R0.unsqueeze(0) if R0.dim() == 1 else R0
            
            solution = odeint(
                self,
                R0_batch,
                t_eval,
                **solver_options
            )
            
            # Convert solution to list of tensors
            if solution.dim() == 3:
                # Batch dimension present
                trajectory = [solution[i, 0] if solution.shape[1] == 1 else solution[i] for i in range(len(t_eval))]
            else:
                trajectory = [solution[i] for i in range(len(t_eval))]
            
            times = t_eval.tolist()
        
        return ReasoningODEOutput(
            trajectory=trajectory,
            times=times,
            final_state=trajectory[-1]
        )
    
    def _solve_rk4(
        self,
        R0: torch.Tensor,
        t_start: float,
        t_end: float,
        t_eval: Union[list, torch.Tensor, None],
    ) -> tuple:
        """
        Custom RK4 implementation for ODE solving.
        
        Args:
            R0: Initial state
            t_start: Start time
            t_end: End time
            t_eval: Evaluation times
            
        Returns:
            Tuple of (trajectory, times)
        """
        if t_eval is None:
            n_steps = 100
            t_eval = torch.linspace(t_start, t_end, n_steps)
        elif not isinstance(t_eval, torch.Tensor):
            t_eval = torch.tensor(t_eval, dtype=torch.float32)
        
        dt = (t_end - t_start) / (len(t_eval) - 1)
        
        trajectory = [R0.detach().clone()]
        R = R0.detach().clone().requires_grad_(True)
        
        for i, t in enumerate(t_eval[:-1]):
            # RK4 steps
            k1 = self.dynamics(t, R)
            k2 = self.dynamics(t + dt/2, R + dt/2 * k1)
            k3 = self.dynamics(t + dt/2, R + dt/2 * k2)
            k4 = self.dynamics(t + dt, R + dt * k3)
            
            R = R + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(R.detach().clone())
        
        return trajectory, t_eval.tolist()
    
    def _solve_euler(
        self,
        R0: torch.Tensor,
        t_start: float,
        t_end: float,
        t_eval: Union[list, torch.Tensor, None],
    ) -> tuple:
        """
        Custom Euler implementation for ODE solving.
        
        Args:
            R0: Initial state
            t_start: Start time
            t_end: End time
            t_eval: Evaluation times
            
        Returns:
            Tuple of (trajectory, times)
        """
        if t_eval is None:
            n_steps = 100
            t_eval = torch.linspace(t_start, t_end, n_steps)
        elif not isinstance(t_eval, torch.Tensor):
            t_eval = torch.tensor(t_eval, dtype=torch.float32)
        
        dt = (t_end - t_start) / (len(t_eval) - 1)
        
        trajectory = [R0.detach().clone()]
        R = R0.detach().clone().requires_grad_(True)
        
        for i, t in enumerate(t_eval[:-1]):
            dRdt = self.dynamics(t, R)
            R = R + dt * dRdt
            trajectory.append(R.detach().clone())
        
        return trajectory, t_eval.tolist()
    
    def forward(self, t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for use with torchdiffeq ODE solvers.
        
        Args:
            t: Current time
            R: Current state
            
        Returns:
            Derivative dR/dt
        """
        return self.dynamics(t, R)


def euler_step(
    dynamics_fn,
    t: torch.Tensor,
    R: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Single Euler method step.
    
    Args:
        dynamics_fn: Function computing dR/dt = f(t, R)
        t: Current time
        R: Current state
        dt: Time step
        
    Returns:
        Next state R + dt * f(t, R)
    """
    return R + dt * dynamics_fn(t, R)


def rk4_step(
    dynamics_fn,
    t: torch.Tensor,
    R: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Single RK4 method step.
    
    Args:
        dynamics_fn: Function computing dR/dt = f(t, R)
        t: Current time
        R: Current state
        dt: Time step
        
    Returns:
        Next state computed with RK4
    """
    k1 = dynamics_fn(t, R)
    k2 = dynamics_fn(t + dt/2, R + dt/2 * k1)
    k3 = dynamics_fn(t + dt/2, R + dt/2 * k2)
    k4 = dynamics_fn(t + dt, R + dt * k3)
    return R + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
