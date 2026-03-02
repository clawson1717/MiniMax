from typing import Any, Dict, List, Optional, Callable
from src.trajectory import TrajectoryLogger
from src.uncertainty import UncertaintyEstimator
from src.allocator import ComputeAllocator
from src.pruner import TrajectoryPruner
from src.monitor import FailureMonitor

class NavigationAgent:
    """
    A multi-step navigation agent that orchestrates graph pruning, 
    scaling, and monitoring logic for complex multi-hop tasks.
    """

    def __init__(self, 
                 env: Any,
                 model: Callable[[str, int], Any],
                 base_compute: int = 3,
                 uncertainty_threshold: float = 0.5):
        """
        Initializes the agent.

        Args:
            env: The environment to interact with. Should have a step(action) method.
            model: A callable that takes a prompt and a compute budget, returns a list of votes.
            base_compute: The initial number of votes/reasoning steps to take.
            uncertainty_threshold: The threshold for scaling and pruning.
        """
        self.env = env
        self.model = model
        self.base_compute = base_compute
        
        self.logger = TrajectoryLogger()
        self.allocator = ComputeAllocator(threshold=uncertainty_threshold)
        self.pruner = TrajectoryPruner(self.logger)
        self.monitor = FailureMonitor(fatigue_threshold=0.7, hijacking_threshold=0.5)
        self.uncertainty_threshold = uncertainty_threshold

    def run(self, goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Executes an agent loop to achieve the specified goal.

        Args:
            goal: The target goal for the agent.
            max_steps: Maximum number of steps to execute.

        Returns:
            A dictionary containing the final trajectory and status.
        """
        self.monitor.set_initial_goal(goal)
        
        # Initial observation
        observation = self.env.reset()
        current_node_id = self.logger.add_node(observation, metadata={"goal": goal})
        
        for step in range(max_steps):
            # 1. Estimate uncertainty and allocate compute (scaling)
            # To estimate uncertainty before a full action, we can do a preliminary small-scale vote
            preliminary_votes = self.model(f"Goal: {goal}. Observation: {observation}. What is the next action?", self.base_compute)
            uncertainty_score = UncertaintyEstimator.from_votes(preliminary_votes)
            
            compute_budget = self.allocator.allocate(self.base_compute, uncertainty_score)
            
            # 2. Decide the next action based on the pruned state and goal
            # In a real scenario, we might pass the pruned trajectory to the model
            clean_trajectory = self.pruner.get_clean_trajectory()
            prompt = f"Goal: {goal}. Current Observation: {observation}. History: {clean_trajectory}. Decide next action."
            
            final_votes = self.model(prompt, compute_budget)
            # Simple majority vote for the action
            action = max(set(final_votes), key=final_votes.count)
            
            # 3. Execute action
            next_observation, reward, done, info = self.env.step(action)
            
            # 4. Log the action/observation to the trajectory graph
            next_node_id = self.logger.add_node(
                next_observation, 
                metadata={"uncertainty": uncertainty_score, "compute_budget": compute_budget},
                votes=final_votes
            )
            self.logger.add_edge(current_node_id, next_node_id, action)
            current_node_id = next_node_id
            observation = next_observation

            # 5. Check for failure modes (fatigue/hijacking)
            status = self.monitor.check_status(self.logger.to_dict(), current_uncertainty=uncertainty_score)
            if status["alerts"]:
                print(f"Step {step}: {status['alerts']}")
                # In a robust system, we might trigger a recovery strategy here

            # 6. Prune the trajectory graph to avoid cycles/dead-ends
            self.pruner.prune_unproductive_branches(self.uncertainty_threshold)
            
            if done:
                break

        return {
            "trajectory": self.logger.to_dict(),
            "status": "completed" if done else "max_steps_reached",
            "final_observation": observation
        }
