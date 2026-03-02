"""
Diffusion-Based Coordinator

Implements OMAD-style diffusion policies for coordinating multi-agent responses
with entropy augmentation. Based on concepts from multi-agent diffusion models
where each agent contributes to a shared trajectory.

Key features:
- Multi-agent diffusion where each agent contributes to a shared trajectory
- Entropy regularization to prevent mode collapse
- Temperature-controlled noise injection
- Coordination across agents with capacity-weighted influence
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .environment import MultiAgentEnvironment, AgentRole, MessageType
from .capacity import InformationCapacityEstimator, CapacityResult


class DenoisingStage(Enum):
    """Stages in the denoising process."""
    INIT = "init"
    DIFFUSION = "diffusion"
    REFINEMENT = "refinement"
    COMPLETE = "complete"


@dataclass
class DiffusionState:
    """State of a diffusion process."""
    
    # Current collective response representation (as embedding-like vector)
    response_vector: np.ndarray
    
    # Current denoising step (0 = pure noise, max_steps = clean)
    current_step: int
    
    # Total denoising steps
    total_steps: int
    
    # Current temperature
    temperature: float
    
    # Current entropy
    entropy: float
    
    # Agent contributions at this step
    agent_contributions: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Stage of denoising
    stage: DenoisingStage = DenoisingStage.INIT
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorConfig:
    """Configuration for the diffusion coordinator."""
    
    # Number of denoising steps
    num_steps: int = 10
    
    # Initial temperature (noise level)
    initial_temperature: float = 1.0
    
    # Final temperature (noise level at end)
    final_temperature: float = 0.1
    
    # Entropy regularization coefficient
    entropy_coef: float = 0.1
    
    # Minimum capacity weight threshold
    min_capacity_weight: float = 0.01
    
    # Response vector dimension (simulated embedding size)
    vector_dim: int = 64
    
    # Convergence threshold for early stopping
    convergence_threshold: float = 0.001
    
    # Maximum iterations for refinement
    max_refinement_iterations: int = 5


class DiffusionCoordinator:
    """
    Coordinates multi-agent response generation using diffusion-style denoising.
    
    The coordinator manages a collective response that is iteratively refined
    through a denoising process. Each agent contributes based on their
    information capacity, with entropy augmentation to ensure diverse outputs.
    
    Based on OMAD (Offline Multi-Agent Diffusion) principles:
    - Agents contribute to shared trajectories
    - Capacity-weighted influence on collective output
    - Entropy regularization prevents mode collapse
    - Temperature schedule controls noise injection
    """
    
    def __init__(
        self,
        environment: MultiAgentEnvironment,
        config: Optional[CoordinatorConfig] = None,
        capacity_estimator: Optional[InformationCapacityEstimator] = None,
    ):
        """
        Initialize the diffusion coordinator.
        
        Args:
            environment: The multi-agent environment to coordinate
            config: Coordinator configuration
            capacity_estimator: Estimator for measuring agent capacities
        """
        self.environment = environment
        self.config = config or CoordinatorConfig()
        self.capacity_estimator = capacity_estimator or InformationCapacityEstimator()
        
        # Track active diffusion processes
        self.active_states: Dict[str, DiffusionState] = {}
        
        # Track agent capacity weights
        self._capacity_weights: Dict[str, float] = {}
        
        # Random state for reproducibility
        self._rng = np.random.default_rng()
        
        # Update capacity weights from environment
        self._update_capacity_weights()
    
    def _update_capacity_weights(self) -> None:
        """Update capacity weights from environment agent states."""
        total_capacity = 0.0
        
        for agent_id, state in self.environment.agent_states.items():
            total_capacity += max(state.capacity, self.config.min_capacity_weight)
        
        if total_capacity > 0:
            for agent_id, state in self.environment.agent_states.items():
                raw_weight = max(state.capacity, self.config.min_capacity_weight)
                self._capacity_weights[agent_id] = raw_weight / total_capacity
        else:
            # Equal weights if no capacity info
            n_agents = len(self.environment.agents)
            if n_agents > 0:
                for agent_id in self.environment.agents:
                    self._capacity_weights[agent_id] = 1.0 / n_agents
    
    def get_capacity_weight(self, agent_id: str) -> float:
        """Get the capacity weight for an agent."""
        return self._capacity_weights.get(agent_id, 0.0)
    
    def get_temperature_schedule(self) -> np.ndarray:
        """
        Generate temperature schedule for denoising.
        
        Returns:
            Array of temperatures from initial to final over num_steps
        """
        # Linear interpolation from initial to final temperature
        alphas = np.linspace(0, 1, self.config.num_steps)
        temperatures = (
            self.config.initial_temperature * (1 - alphas) +
            self.config.final_temperature * alphas
        )
        return temperatures
    
    def initialize_diffusion(
        self,
        task_id: str,
        initial_prompt: Optional[str] = None,
    ) -> DiffusionState:
        """
        Initialize a new diffusion process for a task.
        
        Args:
            task_id: Unique identifier for this diffusion process
            initial_prompt: Optional initial prompt/context
            
        Returns:
            Initial DiffusionState
        """
        # Start with random noise vector (pure noise state)
        initial_vector = self._rng.standard_normal(self.config.vector_dim)
        
        state = DiffusionState(
            response_vector=initial_vector,
            current_step=0,
            total_steps=self.config.num_steps,
            temperature=self.config.initial_temperature,
            entropy=self._compute_vector_entropy(initial_vector),
            stage=DenoisingStage.INIT,
            metadata={"task_id": task_id, "initial_prompt": initial_prompt},
        )
        
        self.active_states[task_id] = state
        return state
    
    def _compute_vector_entropy(self, vector: np.ndarray) -> float:
        """Compute entropy of a vector (measure of disorder/uncertainty)."""
        # Normalize to probability distribution
        abs_vec = np.abs(vector)
        total = np.sum(abs_vec)
        
        if total == 0:
            return 0.0
        
        probs = abs_vec / total
        
        # Compute entropy (avoiding log(0))
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)
    
    def _inject_noise(
        self,
        vector: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """
        Inject temperature-controlled noise into a vector.
        
        Higher temperature = more noise.
        Lower temperature = less noise.
        """
        noise = self._rng.standard_normal(vector.shape)
        return vector + temperature * noise
    
    def _compute_entropy_augmentation(
        self,
        vector: np.ndarray,
        entropy_coef: float,
    ) -> np.ndarray:
        """
        Compute entropy augmentation for diversity.
        
        Encourages exploration in high-entropy regions to prevent
        mode collapse in the collective response.
        """
        # Gradient towards higher entropy
        current_entropy = self._compute_vector_entropy(vector)
        max_entropy = np.log2(len(vector))  # Maximum possible entropy
        
        # Augmentation strength based on distance from max entropy
        entropy_gap = max_entropy - current_entropy
        augmentation = entropy_coef * entropy_gap * self._rng.standard_normal(vector.shape)
        
        return augmentation
    
    def collect_agent_contributions(
        self,
        state: DiffusionState,
    ) -> Dict[str, np.ndarray]:
        """
        Collect contributions from all agents for current diffusion step.
        
        Each agent's contribution is weighted by their capacity and
        their response to the current state.
        
        Args:
            state: Current diffusion state
            
        Returns:
            Dict mapping agent_id to contribution vector
        """
        contributions = {}
        
        for agent_id, agent in self.environment.agents.items():
            if agent is None:
                continue
            
            # Get capacity weight
            weight = self.get_capacity_weight(agent_id)
            
            # Generate contribution based on agent's capacity
            # Higher capacity = more focused contribution
            # Lower capacity = more noisy contribution
            noise_scale = 1.0 - weight
            
            # Simulate agent "processing" the current state
            # In a real implementation, this would call the agent's model
            contribution = self._simulate_agent_contribution(
                state.response_vector,
                weight,
                noise_scale,
            )
            
            contributions[agent_id] = contribution * weight
        
        return contributions
    
    def _simulate_agent_contribution(
        self,
        current_vector: np.ndarray,
        weight: float,
        noise_scale: float,
    ) -> np.ndarray:
        """
        Simulate an agent's contribution to the diffusion process.
        
        Higher capacity agents contribute more focused (less noisy) updates.
        """
        # Create a contribution that moves towards a "clean" signal
        # This simulates the denoising direction
        noise = self._rng.standard_normal(current_vector.shape)
        
        # Higher weight = more focused contribution towards denoising
        # Lower weight = more random contribution
        contribution = -current_vector * weight * 0.1 + noise * noise_scale * 0.1
        
        return contribution
    
    def aggregate_contributions(
        self,
        contributions: Dict[str, np.ndarray],
        temperature: float,
    ) -> np.ndarray:
        """
        Aggregate agent contributions into a collective update.
        
        Uses capacity-weighted averaging with temperature-scaled noise.
        
        Args:
            contributions: Agent contribution vectors
            temperature: Current temperature for noise scaling
            
        Returns:
            Aggregated update vector
        """
        if not contributions:
            return np.zeros(self.config.vector_dim)
        
        # Weighted average of contributions
        total_weight = sum(self._capacity_weights.get(aid, 0.0) for aid in contributions)
        
        if total_weight == 0:
            return np.zeros(self.config.vector_dim)
        
        aggregated = np.zeros(self.config.vector_dim)
        for agent_id, contribution in contributions.items():
            weight = self._capacity_weights.get(agent_id, 0.0) / total_weight
            aggregated += weight * contribution
        
        # Scale by temperature
        aggregated *= (1.0 - temperature * 0.5)
        
        return aggregated
    
    def diffusion_step(
        self,
        state: DiffusionState,
        temperature: float,
    ) -> DiffusionState:
        """
        Perform a single diffusion (denoising) step.
        
        Args:
            state: Current diffusion state
            temperature: Temperature for this step
            
        Returns:
            Updated DiffusionState
        """
        # Update capacity weights
        self._update_capacity_weights()
        
        # Collect contributions from agents
        contributions = self.collect_agent_contributions(state)
        
        # Aggregate contributions
        update = self.aggregate_contributions(contributions, temperature)
        
        # Apply entropy augmentation
        entropy_aug = self._compute_entropy_augmentation(
            state.response_vector,
            self.config.entropy_coef,
        )
        
        # Update response vector
        new_vector = state.response_vector + update + entropy_aug
        
        # Inject temperature-scaled noise
        new_vector = self._inject_noise(new_vector, temperature * 0.1)
        
        # Create new state
        new_state = DiffusionState(
            response_vector=new_vector,
            current_step=state.current_step + 1,
            total_steps=state.total_steps,
            temperature=temperature,
            entropy=self._compute_vector_entropy(new_vector),
            agent_contributions=contributions,
            stage=DenoisingStage.DIFFUSION if state.current_step < state.total_steps - 1 else DenoisingStage.REFINEMENT,
            metadata=state.metadata.copy(),
        )
        
        # Update active state
        task_id = state.metadata.get("task_id", "")
        if task_id in self.active_states:
            self.active_states[task_id] = new_state
        
        return new_state
    
    def refine(
        self,
        state: DiffusionState,
        iterations: Optional[int] = None,
    ) -> DiffusionState:
        """
        Refine the diffusion result with additional iterations.
        
        Args:
            state: Current diffusion state
            iterations: Number of refinement iterations (default from config)
            
        Returns:
            Refined DiffusionState
        """
        iterations = iterations or self.config.max_refinement_iterations
        
        current_state = state
        current_state.stage = DenoisingStage.REFINEMENT
        
        for i in range(iterations):
            # Small temperature for refinement
            refine_temp = self.config.final_temperature * 0.5
            
            # Get contributions
            contributions = self.collect_agent_contributions(current_state)
            
            # Smaller updates during refinement
            update = self.aggregate_contributions(contributions, refine_temp) * 0.1
            
            # Check for convergence
            update_norm = np.linalg.norm(update)
            if update_norm < self.config.convergence_threshold:
                break
            
            # Apply small update
            new_vector = current_state.response_vector + update
            current_state = DiffusionState(
                response_vector=new_vector,
                current_step=current_state.current_step,
                total_steps=current_state.total_steps,
                temperature=refine_temp,
                entropy=self._compute_vector_entropy(new_vector),
                agent_contributions=contributions,
                stage=DenoisingStage.REFINEMENT,
                metadata=current_state.metadata.copy(),
            )
        
        # Mark as complete
        current_state.stage = DenoisingStage.COMPLETE
        
        # Update active state
        task_id = state.metadata.get("task_id", "")
        if task_id in self.active_states:
            self.active_states[task_id] = current_state
        
        return current_state
    
    def run_diffusion(
        self,
        task_id: str,
        initial_prompt: Optional[str] = None,
    ) -> DiffusionState:
        """
        Run the complete diffusion process from init to completion.
        
        Args:
            task_id: Unique identifier for this diffusion
            initial_prompt: Optional initial prompt
            
        Returns:
            Final DiffusionState
        """
        # Initialize
        state = self.initialize_diffusion(task_id, initial_prompt)
        
        # Get temperature schedule
        temperatures = self.get_temperature_schedule()
        
        # Run diffusion steps
        for temp in temperatures:
            state = self.diffusion_step(state, temp)
        
        # Refine
        state = self.refine(state)
        
        return state
    
    def run_parallel_diffusion(
        self,
        task_ids: List[str],
        initial_prompts: Optional[List[str]] = None,
    ) -> Dict[str, DiffusionState]:
        """
        Run diffusion processes for multiple tasks in parallel.
        
        Args:
            task_ids: List of task identifiers
            initial_prompts: Optional list of initial prompts
            
        Returns:
            Dict mapping task_id to final DiffusionState
        """
        if initial_prompts is None:
            initial_prompts = [None] * len(task_ids)
        
        results = {}
        
        # Initialize all processes
        for task_id, prompt in zip(task_ids, initial_prompts):
            self.initialize_diffusion(task_id, prompt)
        
        # Run all steps for all tasks
        temperatures = self.get_temperature_schedule()
        
        for temp in temperatures:
            for task_id in task_ids:
                if task_id in self.active_states:
                    state = self.active_states[task_id]
                    self.diffusion_step(state, temp)
        
        # Refine all
        for task_id in task_ids:
            if task_id in self.active_states:
                state = self.active_states[task_id]
                results[task_id] = self.refine(state)
        
        return results
    
    def get_state(self, task_id: str) -> Optional[DiffusionState]:
        """Get the current state for a task."""
        return self.active_states.get(task_id)
    
    def clear_state(self, task_id: str) -> bool:
        """Clear a completed diffusion state."""
        if task_id in self.active_states:
            del self.active_states[task_id]
            return True
        return False
    
    def vector_to_response(self, state: DiffusionState) -> Dict[str, Any]:
        """
        Convert a diffusion state vector to a response representation.
        
        In a real implementation, this would decode the latent representation
        into actual text. Here we provide a structured summary.
        
        Args:
            state: Final diffusion state
            
        Returns:
            Dict containing response information
        """
        # Analyze the final vector
        vector = state.response_vector
        
        # Compute quality metrics
        final_entropy = state.entropy
        vector_norm = float(np.linalg.norm(vector))
        max_amplitude = float(np.max(np.abs(vector)))
        
        # Determine dominant agents
        agent_influences = {
            aid: float(np.linalg.norm(contrib))
            for aid, contrib in state.agent_contributions.items()
        }
        
        return {
            "task_id": state.metadata.get("task_id", "unknown"),
            "status": state.stage.value,
            "steps_completed": state.current_step,
            "final_temperature": state.temperature,
            "final_entropy": final_entropy,
            "vector_norm": vector_norm,
            "max_amplitude": max_amplitude,
            "agent_influences": agent_influences,
            "total_contributors": len(state.agent_contributions),
        }
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get statistics about the coordinator."""
        return {
            "num_active_diffusions": len(self.active_states),
            "num_agents": len(self.environment.agents),
            "capacity_weights": dict(self._capacity_weights),
            "config": {
                "num_steps": self.config.num_steps,
                "initial_temperature": self.config.initial_temperature,
                "final_temperature": self.config.final_temperature,
                "entropy_coef": self.config.entropy_coef,
            },
        }


def create_coordinator(
    environment: Optional[MultiAgentEnvironment] = None,
    num_steps: int = 10,
    entropy_coef: float = 0.1,
) -> DiffusionCoordinator:
    """
    Factory function to create a diffusion coordinator.
    
    Args:
        environment: Multi-agent environment (creates default if None)
        num_steps: Number of denoising steps
        entropy_coef: Entropy regularization coefficient
        
    Returns:
        Configured DiffusionCoordinator
    """
    if environment is None:
        from .environment import create_collaboration_environment
        environment = create_collaboration_environment()
    
    config = CoordinatorConfig(
        num_steps=num_steps,
        entropy_coef=entropy_coef,
    )
    
    return DiffusionCoordinator(environment=environment, config=config)


if __name__ == "__main__":
    # Demo usage
    from .environment import create_collaboration_environment
    
    env = create_collaboration_environment(num_workers=3)
    coordinator = create_coordinator(environment=env, num_steps=5)
    
    print("Coordinator Stats:")
    print(coordinator.get_coordinator_stats())
    
    # Run a diffusion process
    state = coordinator.run_diffusion("demo_task", "Solve this problem")
    
    print(f"\nDiffusion completed:")
    print(f"  Steps: {state.current_step}")
    print(f"  Final entropy: {state.entropy:.4f}")
    print(f"  Final temperature: {state.temperature:.4f}")
    
    # Convert to response
    response = coordinator.vector_to_response(state)
    print(f"\nResponse: {response}")
