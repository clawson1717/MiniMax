"""Core agent implementation for pruned adaptive web agent.

This module integrates:
- CATTS: Uncertainty estimation with vote distribution statistics
- WebClipper: Trajectory graph for cycle detection and pruning
- CM2: Checklist-based evaluation with fine-grained rewards
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import time
import random

from src.uncertainty import UncertaintyEstimator, VoteDistribution
from src.trajectory_graph import TrajectoryGraph
from src.checklist import Checklist, ChecklistEvaluator, CheckStatus
from src.pruning import PruningManager, PruningContext, CompositePruningStrategy
from src.recovery import RecoveryManager, RecoveryAction, RecoveryStrategy


@dataclass
class AgentConfig:
    """Configuration for AdaptiveWebAgent."""
    max_steps: int = 50
    max_retries: int = 3
    uncertainty_threshold: float = 0.7
    stuck_threshold: float = 0.3  # Checklist progress below this = stuck
    trajectory_pruning_threshold: float = 0.5
    use_recovery: bool = True
    use_checklist: bool = True
    save_trajectory: bool = True
    debug: bool = False
    # Uncertainty config
    min_samples: int = 3
    max_samples: int = 20
    # Recovery config
    recovery_max_retries: int = 3


@dataclass
class Step:
    """Represents a single step in the agent's trajectory."""
    index: int
    action: str
    observation: str = ""
    thought: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    uncertainty_stats: Dict[str, float] = field(default_factory=dict)
    is_pruned: bool = False
    compute_budget: int = 5
    vote_distribution: Optional[VoteDistribution] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "index": self.index,
            "action": self.action,
            "observation": self.observation,
            "thought": self.thought,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "uncertainty": self.uncertainty,
            "uncertainty_stats": self.uncertainty_stats,
            "is_pruned": self.is_pruned,
            "compute_budget": self.compute_budget,
        }


class AdaptiveWebAgent:
    """
    Adaptive web agent with uncertainty quantification, trajectory pruning,
    checklist evaluation, and recovery mechanisms.
    
    Combines concepts from CATTS, WebClipper, and CM2:
    - CATTS: Uncertainty-driven compute allocation
    - WebClipper: Graph-based trajectory management with cycle detection
    - CM2: Checklist-based fine-grained evaluation
    
    Execution loop:
        while not done:
            1. Estimate uncertainty → determine compute budget
            2. Generate candidate actions (scaled by budget)
            3. Execute best action
            4. Update trajectory graph
            5. Detect and prune cycles/redundancy
            6. Score step against checklist
            7. If stuck (low checklist progress + high uncertainty): trigger recovery
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the adaptive web agent.
        
        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or AgentConfig()
        self.trajectory: List[Step] = []
        self.current_step: int = 0
        self.is_running: bool = False
        self.task_description: str = ""
        
        # Initialize subsystems
        self._uncertainty_estimator: Optional[UncertaintyEstimator] = None
        self._trajectory_graph: Optional[TrajectoryGraph] = None
        self._checklist_evaluator: Optional[ChecklistEvaluator] = None
        self._current_checklist: Optional[Checklist] = None
        self._pruning_manager: Optional[PruningManager] = None
        self._recovery_manager: Optional[RecoveryManager] = None
        
        # State tracking
        self._last_progress: float = 0.0
        self._steps_without_progress: int = 0
        self._retry_count: int = 0
        
        # Action execution callback (set by user)
        self._execute_action_callback: Optional[Callable] = None
    
    def initialize(self) -> None:
        """Initialize all agent subsystems."""
        # Initialize uncertainty estimator (CATTS)
        self._uncertainty_estimator = UncertaintyEstimator(
            method="ensemble",
            num_samples=self.config.max_samples,
            random_seed=42
        )
        
        # Initialize trajectory graph (WebClipper)
        self._trajectory_graph = TrajectoryGraph()
        
        # Initialize checklist evaluator (CM2)
        self._checklist_evaluator = ChecklistEvaluator()
        
        # Initialize pruning manager
        self._pruning_manager = PruningManager()
        self._pruning_manager.setup_default()
        
        # Initialize recovery manager
        self._recovery_manager = RecoveryManager(
            max_retries=self.config.recovery_max_retries
        )
        
        if self.config.debug:
            print(f"[Agent] Initialized with config: max_steps={self.config.max_steps}")
    
    def set_action_executor(self, callback: Callable[[str], Dict[str, Any]]) -> None:
        """Set the callback function for executing actions.
        
        Args:
            callback: Function that takes action string and returns result dict
                     with keys: observation, success, metadata
        """
        self._execute_action_callback = callback
    
    def set_task_type(self, task_type: str) -> None:
        """Set the task type for checklist creation.
        
        Args:
            task_type: Type of task ('navigation', 'search', 'form', 
                      'information_extraction', or custom)
        """
        # Auto-initialize if not already done
        if self._checklist_evaluator is None:
            self.initialize()
        
        self._current_checklist = self._checklist_evaluator.create_checklist(task_type)
        if self.config.debug:
            print(f"[Agent] Task type set to: {task_type}")
    
    def run_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a web task.
        
        This is the main entry point for running tasks.
        
        Args:
            task_description: The task description to accomplish.
            
        Returns:
            Dictionary containing task results and metadata.
        """
        self.initialize()
        self.task_description = task_description
        self.is_running = True
        self.current_step = 0
        
        result = {
            "task": task_description,
            "success": False,
            "steps_taken": 0,
            "final_answer": None,
            "trajectory": [],
            "checklist_score": 0.0,
            "recovery_attempts": 0,
            "pruned_states": 0,
        }
        
        # Add initial state to trajectory graph
        self._trajectory_graph.add_state(
            observation=task_description,
            action="start",
            metadata={"task": task_description}
        )
        
        if self.config.debug:
            print(f"[Agent] Starting task: {task_description}")
        
        # Main execution loop
        while self.current_step < self.config.max_steps:
            # Check if done
            if self._current_checklist and self._current_checklist.is_complete():
                result["success"] = True
                break
            
            # Execute a step
            step_result = self.execute_step()
            
            if step_result is None:
                # Task completed or no more steps
                break
            
            # Add to trajectory
            self.trajectory.append(step_result)
            result["steps_taken"] = self.current_step
            
            # Check for stuck condition
            if self.check_stuck():
                if self.config.debug:
                    print(f"[Agent] Stuck detected at step {self.current_step}")
                
                # Trigger recovery
                recovered = self.recover()
                if not recovered:
                    if self.config.debug:
                        print("[Agent] Recovery failed, attempting to continue")
                    # Try to continue anyway
        
        # Finalize
        self.is_running = False
        result["trajectory"] = [s.to_dict() for s in self.trajectory]
        result["final_checklist_score"] = (
            self._current_checklist.get_score() if self._current_checklist else 0.0
        )
        
        if self.config.debug:
            print(f"[Agent] Task complete. Steps: {result['steps_taken']}, "
                  f"Success: {result['success']}, "
                  f"Score: {result.get('final_checklist_score', 0):.2f}")
        
        return result
    
    def execute_step(self) -> Optional[Step]:
        """Execute a single step in the agent loop.
        
        Returns:
            The executed Step, or None if task is complete.
        """
        # Step 1: Estimate uncertainty → determine compute budget
        uncertainty, stats, budget = self._estimate_uncertainty_and_budget()
        
        # Step 2: Generate candidate actions (scaled by budget)
        candidates = self._generate_candidates(budget)
        
        # Step 3: Execute best action
        action = self._select_best_action(candidates, uncertainty)
        step_result = self._execute_action(action)
        
        # Update step with uncertainty info
        step_result.uncertainty = uncertainty
        step_result.uncertainty_stats = stats
        step_result.compute_budget = budget
        
        # Step 4: Update trajectory graph
        self._update_trajectory_graph(step_result)
        
        # Step 5: Detect and prune cycles/redundancy
        pruned_count = self._prune_trajectory()
        
        # Step 6: Score step against checklist
        checklist_result = self._evaluate_checklist(step_result)
        
        # Track progress
        if checklist_result:
            progress = checklist_result.get("progress", 0.0)
            if progress > self._last_progress:
                self._last_progress = progress
                self._steps_without_progress = 0
            else:
                self._steps_without_progress += 1
        
        self.current_step += 1
        
        return step_result
    
    def _estimate_uncertainty_and_budget(self) -> tuple:
        """Estimate uncertainty and determine compute budget.
        
        Returns:
            Tuple of (uncertainty, stats, compute_budget)
        """
        # Get current observation
        observation = ""
        if self.trajectory:
            observation = self.trajectory[-1].observation
        if not observation:
            observation = self.task_description
        
        # Generate vote distribution
        votes = self._uncertainty_estimator.generate_votes(
            observation=observation,
            n_samples=self.config.min_samples
        )
        
        # Get uncertainty statistics
        stats = self._uncertainty_estimator.get_uncertainty_stats(votes)
        
        # Determine compute budget based on uncertainty
        budget = self._uncertainty_estimator.get_compute_budget(
            stats,
            min_samples=self.config.min_samples,
            max_samples=self.config.max_samples
        )
        
        return stats["uncertainty"], stats, budget
    
    def _generate_candidates(self, budget: int) -> List[str]:
        """Generate candidate actions based on compute budget.
        
        Args:
            budget: Number of candidate actions to generate
            
        Returns:
            List of candidate action strings
        """
        # In a real implementation, this would use the LLM to generate
        # diverse candidate actions. For simulation, we use predefined sets.
        
        observation = ""
        if self.trajectory:
            observation = self.trajectory[-1].observation
        
        # Generate candidates based on context
        if "search" in observation.lower():
            candidates = ["type_query", "press_enter", "click_result", 
                        "scroll_down", "go_back", "wait"]
        elif "form" in observation.lower():
            candidates = ["fill_field", "submit_form", "clear_field",
                         "validate_input", "scroll_down", "go_back"]
        elif "button" in observation.lower() or "click" in observation.lower():
            candidates = ["click_button", "fill_form", "scroll_down",
                         "wait", "go_back", "take_screenshot"]
        else:
            candidates = ["navigate", "click", "type", "scroll",
                         "wait", "go_back", "take_screenshot"]
        
        # Return budget-limited candidates (with some randomness)
        random.shuffle(candidates)
        return candidates[:max(3, min(budget, len(candidates)))]
    
    def _select_best_action(self, candidates: List[str], uncertainty: float) -> str:
        """Select the best action from candidates.
        
        Args:
            candidates: List of candidate action strings
            uncertainty: Current uncertainty score
            
        Returns:
            Selected action string
        """
        if not candidates:
            return "wait"
        
        # In high uncertainty, use voting to select
        if uncertainty > self.config.uncertainty_threshold:
            observation = ""
            if self.trajectory:
                observation = self.trajectory[-1].observation
            
            votes = self._uncertainty_estimator.generate_votes(
                observation=observation,
                n_samples=self.config.max_samples,
                candidate_actions=candidates
            )
            
            # Select most voted action
            best_action, _ = votes.get_most_common()
            if best_action:
                return best_action
        
        # Otherwise, select first (greedy)
        return candidates[0]
    
    def _execute_action(self, action: str) -> Step:
        """Execute an action and return the step result.
        
        Args:
            action: Action string to execute
            
        Returns:
            Step with execution results
        """
        step = Step(
            index=self.current_step,
            action=action,
            thought=f"Executing {action}"
        )
        
        # Use callback if provided
        if self._execute_action_callback:
            try:
                result = self._execute_action_callback(action)
                step.observation = result.get("observation", "")
                step.metadata = result.get("metadata", {})
                step.metadata["success"] = result.get("success", True)
            except Exception as e:
                step.observation = f"Error: {str(e)}"
                step.metadata = {"success": False, "error": str(e)}
        else:
            # Simulation mode
            step.observation = self._simulate_observation(action)
            step.metadata = {"success": True, "simulated": True}
        
        return step
    
    def _simulate_observation(self, action: str) -> str:
        """Simulate an observation for testing.
        
        Args:
            action: The action that was executed
            
        Returns:
            Simulated observation string
        """
        observations = {
            "navigate": "Navigated to target page. Found search bar and navigation links.",
            "click": "Clicked element. Page updated with new content.",
            "type": "Typed text into field. Input accepted.",
            "scroll": "Scrolled down. More content loaded.",
            "wait": "Waited for page to load. Page is stable.",
            "go_back": "Navigated back. Previous page loaded.",
            "submit_form": "Form submitted. Success message displayed.",
            "take_screenshot": "Screenshot captured.",
        }
        return observations.get(action, f"Action '{action}' executed.")
    
    def _update_trajectory_graph(self, step: Step) -> None:
        """Update trajectory graph with new step.
        
        Args:
            step: The executed step
        """
        if not self._trajectory_graph:
            return
        
        # Get previous state
        prev_state_id = None
        if self.current_step > 0 and self._trajectory_graph.nodes:
            # Find the most recent state
            prev_state_id = max(self._trajectory_graph.nodes.keys())
        
        # Add new state
        new_state_id = self._trajectory_graph.add_state(
            observation=step.observation,
            action=step.action,
            metadata={
                "step_index": step.index,
                "uncertainty": step.uncertainty,
                "success": step.metadata.get("success", True),
            },
            timestamp=step.timestamp
        )
        
        # Add edge from previous state
        if prev_state_id is not None:
            try:
                self._trajectory_graph.add_edge(
                    from_state=prev_state_id,
                    to_state=new_state_id,
                    action=step.action,
                    success=step.metadata.get("success", True),
                    weight=1.0 - (step.uncertainty or 0.5),
                )
            except ValueError:
                pass  # Graph not ready yet
    
    def _prune_trajectory(self) -> int:
        """Detect and prune cycles/redundancy in trajectory.
        
        Returns:
            Number of states pruned
        """
        if not self._pruning_manager or not self._trajectory_graph:
            return 0
        
        # Create pruning context
        context = PruningContext(
            task_description=self.task_description,
            current_step=self.current_step,
            max_steps=self.config.max_steps,
            uncertainty_scores={
                s.index: s.uncertainty or 0.5 
                for s in self.trajectory
            },
            success_history={
                s.index: s.metadata.get("success", True)
                for s in self.trajectory
            }
        )
        
        # Get prunable branches
        prunable = self._trajectory_graph.get_prunable_branches()
        
        pruned_count = 0
        for state_id in prunable:
            if self._pruning_manager.prune_if_needed(
                state_id, 
                self._trajectory_graph, 
                context
            ):
                pruned_count += 1
        
        return pruned_count
    
    def _evaluate_checklist(self, step: Step) -> Optional[Dict[str, Any]]:
        """Evaluate step against checklist.
        
        Args:
            step: The executed step
            
        Returns:
            Evaluation results or None if no checklist
        """
        if not self._checklist_evaluator or not self._current_checklist:
            return None
        
        # Build step data for evaluation
        step_data = {
            "action": step.action,
            "observation": step.observation,
            "step_index": step.index,
            "timestamp": step.timestamp,
            "metadata": step.metadata,
            "url": step.metadata.get("url", ""),
        }
        
        result = self._checklist_evaluator.evaluate_step(
            step_data, 
            self._current_checklist
        )
        
        # Update step metadata with checklist info
        step.metadata["checklist_score"] = result.get("score", 0.0)
        step.metadata["checklist_progress"] = result.get("progress", 0.0)
        
        return result
    
    def check_stuck(self) -> bool:
        """Detect if the agent is stuck.
        
        Stuck condition: low checklist progress + high uncertainty
        
        Returns:
            True if agent is stuck and needs recovery
        """
        if not self._current_checklist:
            return False
        
        progress = self._current_checklist.get_progress_percentage()
        uncertainty = 0.5
        
        if self.trajectory:
            uncertainty = self.trajectory[-1].uncertainty or 0.5
        
        # Check stuck conditions
        is_stuck = (
            progress < self.config.stuck_threshold and
            uncertainty > self.config.uncertainty_threshold and
            self._steps_without_progress >= 3
        )
        
        if self.config.debug and is_stuck:
            print(f"[Agent] Stuck detected: progress={progress:.1f}%, "
                  f"uncertainty={uncertainty:.2f}, "
                  f"steps_without_progress={self._steps_without_progress}")
        
        return is_stuck
    
    def recover(self) -> bool:
        """Execute recovery strategy when stuck.
        
        Returns:
            True if recovery was successful
        """
        if not self._recovery_manager:
            return False
        
        if self.config.debug:
            print(f"[Agent] Executing recovery strategy...")
        
        # Create context for recovery decision
        context = {
            "current_step": self.current_step,
            "retry_count": self._retry_count,
            "uncertainty": (
                self.trajectory[-1].uncertainty 
                if self.trajectory else 0.5
            ),
            "trajectory": [s.to_dict() for s in self.trajectory],
            "checklist_progress": (
                self._current_checklist.get_progress_percentage()
                if self._current_checklist else 0.0
            )
        }
        
        # Assess failure and get recovery action
        error = Exception("Stuck condition detected")
        recovery_action = self._recovery_manager.assess_failure(error, context)
        
        if self.config.debug:
            print(f"[Agent] Recovery action: {recovery_action.strategy.value}")
        
        # Execute recovery
        updated_context = self._recovery_manager.execute_recovery(
            recovery_action, 
            context
        )
        
        # Handle specific recovery strategies
        if recovery_action.strategy == RecoveryStrategy.BACKTRACK:
            # Go back to previous state
            if len(self.trajectory) > 1:
                self.trajectory = self.trajectory[:-1]
                self.current_step = len(self.trajectory)
                self._steps_without_progress = 0
                
        elif recovery_action.strategy == RecoveryStrategy.RETRY:
            # Increment retry count
            self._retry_count += 1
            
        elif recovery_action.strategy == RecoveryStrategy.RESET:
            # Reset trajectory but keep checklist
            self.trajectory = []
            self.current_step = 0
            self._steps_without_progress = 0
            self._retry_count = 0
            
        elif recovery_action.strategy == RecoveryStrategy.HUMAN_IN_LOOP:
            # In real implementation, this would pause for human input
            if self.config.debug:
                print("[Agent] Human-in-the-loop requested")
            return False
        
        # Record the recovery attempt
        self._recovery_manager.record_attempt(recovery_action, success=True)
        
        return True
    
    def get_trajectory(self) -> List[Step]:
        """Get the current trajectory."""
        return self.trajectory.copy()
    
    def get_trajectory_graph(self) -> Optional[TrajectoryGraph]:
        """Get the trajectory graph."""
        return self._trajectory_graph
    
    def get_checklist(self) -> Optional[Checklist]:
        """Get the current checklist."""
        return self._current_checklist
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "current_step": self.current_step,
            "trajectory_length": len(self.trajectory),
            "max_steps": self.config.max_steps,
            "checklist_score": (
                self._current_checklist.get_score() 
                if self._current_checklist else 0.0
            ),
            "checklist_progress": (
                self._current_checklist.get_progress_percentage()
                if self._current_checklist else 0.0
            ),
            "graph_stats": (
                self._trajectory_graph.get_graph_stats()
                if self._trajectory_graph else {}
            ),
            "pruning_stats": (
                self._pruning_manager.get_stats()
                if self._pruning_manager else {}
            ),
            "recovery_stats": {
                "attempts": len(
                    self._recovery_manager.attempt_history
                ) if self._recovery_manager else 0,
            } if self._recovery_manager else {}
        }
    
    def prune_trajectory(self, threshold: Optional[float] = None) -> int:
        """Prune low-uncertainty trajectory nodes.
        
        Args:
            threshold: Uncertainty threshold for pruning. 
                      Uses config default if not provided.
            
        Returns:
            Number of nodes pruned.
        """
        threshold = threshold or self.config.trajectory_pruning_threshold
        pruned_count = 0
        
        for step in self.trajectory:
            if step.uncertainty is not None and step.uncertainty < threshold:
                step.is_pruned = True
                pruned_count += 1
                
        return pruned_count
    
    def step(self, action: str) -> Step:
        """Execute a single step (backwards compatible alias).
        
        Args:
            action: The action to execute.
            
        Returns:
            The executed Step.
        """
        # Auto-initialize if needed
        if self._uncertainty_estimator is None:
            self.initialize()
        
        # Use the same logic as execute_step but simplified
        observation = ""
        if self.trajectory:
            observation = self.trajectory[-1].observation
        
        votes = self._uncertainty_estimator.generate_votes(
            observation=observation or self.task_description,
            n_samples=self.config.min_samples
        )
        stats = self._uncertainty_estimator.get_uncertainty_stats(votes)
        
        step = Step(
            index=self.current_step,
            action=action,
            observation=self._simulate_observation(action),
            uncertainty=stats["uncertainty"],
            uncertainty_stats=stats,
            compute_budget=self._uncertainty_estimator.get_compute_budget(
                stats,
                min_samples=self.config.min_samples,
                max_samples=self.config.max_samples
            ),
            metadata={"success": True, "simulated": True}
        )
        
        self.trajectory.append(step)
        self.current_step += 1
        return step
    
    def reset(self) -> None:
        """Reset agent state."""
        self.trajectory = []
        self.current_step = 0
        self.is_running = False
        self._last_progress = 0.0
        self._steps_without_progress = 0
        self._retry_count = 0
        
        # Reset subsystems
        if self._trajectory_graph:
            self._trajectory_graph.reset()
        if self._current_checklist:
            self._current_checklist.reset()


# Backwards compatibility aliases
IntegratedAgent = AdaptiveWebAgent
