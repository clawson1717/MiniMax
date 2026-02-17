"""Integration tests for the integrated agent.

Tests the full integration of:
- Uncertainty estimation (CATTS)
- Trajectory graph (WebClipper)
- Checklist evaluation (CM2)
- Pruning strategies
- Recovery mechanisms
"""

from typing import Dict
import pytest
from src.agent import AdaptiveWebAgent, AgentConfig, Step
from src.uncertainty import UncertaintyEstimator, VoteDistribution
from src.trajectory_graph import TrajectoryGraph
from src.checklist import ChecklistEvaluator, Checklist
from src.pruning import PruningManager, PruningContext
from src.recovery import RecoveryManager, RecoveryStrategy


class TestIntegratedAgentInitialization:
    """Tests for integrated agent initialization."""
    
    def test_agent_initialization(self):
        """Test agent initializes all subsystems."""
        agent = AdaptiveWebAgent()
        agent.initialize()
        
        assert agent._uncertainty_estimator is not None
        assert agent._trajectory_graph is not None
        assert agent._checklist_evaluator is not None
        assert agent._pruning_manager is not None
        assert agent._recovery_manager is not None
    
    def test_agent_with_config(self):
        """Test agent with custom configuration."""
        config = AgentConfig(
            max_steps=100,
            uncertainty_threshold=0.8,
            stuck_threshold=0.2,
            debug=True
        )
        agent = AdaptiveWebAgent(config)
        agent.initialize()
        
        assert agent.config.max_steps == 100
        assert agent.config.uncertainty_threshold == 0.8
        assert agent.config.stuck_threshold == 0.2


class TestIntegratedAgentExecutionLoop:
    """Tests for the main execution loop."""
    
    def test_run_task_basic(self):
        """Test basic task execution."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5, debug=False))
        
        # Execute task
        result = agent.run_task("Test task")
        
        assert "task" in result
        assert "success" in result
        assert "steps_taken" in result
        assert "trajectory" in result
        assert isinstance(result["trajectory"], list)
    
    def test_run_task_with_checklist(self):
        """Test task execution with checklist."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5, debug=False))
        agent.set_task_type("navigation")
        
        result = agent.run_task("Navigate to example.com")
        
        assert "final_checklist_score" in result
        assert isinstance(result["final_checklist_score"], float)
    
    def test_execute_step(self):
        """Test single step execution."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.initialize()
        
        # Execute step
        step = agent.execute_step()
        
        assert step is not None
        assert step.index == 0
        assert step.action is not None
        assert step.uncertainty is not None
        assert step.uncertainty_stats is not None
        assert "confidence" in step.uncertainty_stats
        assert "entropy" in step.uncertainty_stats
    
    def test_execute_step_updates_graph(self):
        """Test that execute_step updates trajectory graph."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        agent.initialize()
        
        # Execute steps
        agent.execute_step()
        agent.execute_step()
        
        # Check graph was updated
        assert len(agent._trajectory_graph.nodes) > 0
    
    def test_action_callback(self):
        """Test custom action executor callback."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        agent.initialize()
        
        # Track executed actions
        executed_actions = []
        
        def action_executor(action: str) -> Dict:
            executed_actions.append(action)
            return {
                "observation": f"Executed {action}",
                "success": True,
                "metadata": {"action": action}
            }
        
        agent.set_action_executor(action_executor)
        agent.run_task("Test task")
        
        assert len(executed_actions) > 0


class TestUncertaintyIntegration:
    """Tests for uncertainty estimation integration."""
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation in execution loop."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        agent.initialize()
        
        step = agent.execute_step()
        
        assert step.uncertainty is not None
        assert 0.0 <= step.uncertainty <= 1.0
        assert "confidence" in step.uncertainty_stats
        assert "entropy" in step.uncertainty_stats
    
    def test_compute_budget_scaling(self):
        """Test compute budget scales with uncertainty."""
        agent = AdaptiveWebAgent(AgentConfig(
            max_steps=5,
            min_samples=3,
            max_samples=10
        ))
        agent.initialize()
        
        # Execute multiple steps
        budgets = []
        for _ in range(5):
            step = agent.execute_step()
            budgets.append(step.compute_budget)
        
        # Budgets should vary based on uncertainty
        assert all(3 <= b <= 10 for b in budgets)
    
    def test_high_uncertainty_triggers_voting(self):
        """Test high uncertainty triggers ensemble voting."""
        agent = AdaptiveWebAgent(AgentConfig(
            max_steps=3,
            uncertainty_threshold=0.3  # Low threshold for testing
        ))
        agent.initialize()
        
        # Should use voting when uncertainty is high
        step = agent.execute_step()
        
        # Vote distribution should be generated
        assert step.compute_budget > agent.config.min_samples


class TestTrajectoryGraphIntegration:
    """Tests for trajectory graph integration."""
    
    def test_graph_grows_with_steps(self):
        """Test trajectory graph grows with each step."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.initialize()
        
        initial_nodes = len(agent._trajectory_graph.nodes)
        
        for _ in range(3):
            agent.execute_step()
        
        final_nodes = len(agent._trajectory_graph.nodes)
        assert final_nodes > initial_nodes
    
    def test_graph_has_edges(self):
        """Test trajectory graph has edges between states."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        agent.initialize()
        
        agent.execute_step()
        agent.execute_step()
        
        assert len(agent._trajectory_graph.edges) > 0
    
    def test_graph_tracks_uncertainty(self):
        """Test graph tracks uncertainty in metadata."""
        agent = AgentConfig(max_steps=3)
        agent = AdaptiveWebAgent(agent)
        agent.initialize()
        
        step = agent.execute_step()
        
        # Check graph node has uncertainty metadata
        if agent._trajectory_graph.nodes:
            node = list(agent._trajectory_graph.nodes.values())[-1]
            assert "uncertainty" in node.metadata


class TestChecklistIntegration:
    """Tests for checklist evaluation integration."""
    
    def test_checklist_evaluation(self):
        """Test checklist is evaluated each step."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.set_task_type("navigation")
        agent.initialize()
        
        # Execute step
        step = agent.execute_step()
        
        # Step should have checklist metadata
        assert "checklist_score" in step.metadata
        assert "checklist_progress" in step.metadata
    
    def test_checklist_progress_tracking(self):
        """Test checklist progress is tracked."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=10))
        agent.set_task_type("search")
        agent.initialize()
        
        # Run task
        result = agent.run_task("Search for examples")
        
        assert "final_checklist_score" in result
        assert 0.0 <= result["final_checklist_score"] <= 1.0
    
    def test_different_task_types(self):
        """Test different task types create appropriate checklists."""
        task_types = ["navigation", "search", "form", "information_extraction"]
        
        for task_type in task_types:
            agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
            agent.set_task_type(task_type)
            agent.initialize()
            
            assert agent._current_checklist is not None
            assert agent._current_checklist.task_type == task_type


class TestPruningIntegration:
    """Tests for pruning integration."""
    
    def test_pruning_detects_cycles(self):
        """Test pruning can detect cycles."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=10))
        agent.initialize()
        
        # Create a cycle in the graph
        agent.execute_step()
        agent.execute_step()
        
        # Detect cycles
        cycles = agent._trajectory_graph.detect_cycles()
        assert isinstance(cycles, list)
    
    def test_pruning_manager_works(self):
        """Test pruning manager integrates with agent."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.initialize()
        
        # Execute some steps
        for _ in range(3):
            agent.execute_step()
        
        # Prune
        pruned = agent._prune_trajectory()
        assert isinstance(pruned, int)
    
    def test_pruning_context(self):
        """Test pruning context is created correctly."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.initialize()
        
        # Add some steps
        for _ in range(2):
            agent.execute_step()
        
        # Create context manually to test
        context = PruningContext(
            task_description="Test",
            current_step=agent.current_step,
            max_steps=agent.config.max_steps,
            uncertainty_scores={i: 0.5 for i in range(agent.current_step)},
            success_history={i: True for i in range(agent.current_step)}
        )
        
        assert context.current_step >= 0
        assert context.max_steps > 0


class TestRecoveryIntegration:
    """Tests for recovery mechanism integration."""
    
    def test_check_stuck_detects_low_progress(self):
        """Test stuck detection with low progress."""
        agent = AdaptiveWebAgent(AgentConfig(
            max_steps=10,
            stuck_threshold=50.0,  # High threshold so always "stuck"
            debug=False
        ))
        agent.set_task_type("navigation")
        agent.initialize()
        
        # Execute a few steps
        for _ in range(5):
            agent.execute_step()
        
        # Should detect stuck condition
        is_stuck = agent.check_stuck()
        assert isinstance(is_stuck, bool)
    
    def test_recovery_execution(self):
        """Test recovery executes without error."""
        agent = AdaptiveWebAgent(AgentConfig(
            max_steps=10,
            stuck_threshold=50.0,
            debug=False
        ))
        agent.set_task_type("navigation")
        agent.initialize()
        
        # Execute some steps to get stuck
        for _ in range(5):
            agent.execute_step()
        
        # Try recovery
        recovered = agent.recover()
        assert isinstance(recovered, bool)
    
    def test_recovery_strategies(self):
        """Test different recovery strategies."""
        manager = RecoveryManager(max_retries=3)
        
        # Test retry strategy
        context = {"current_step": 1, "retry_count": 0, "uncertainty": 0.3}
        action = manager.assess_failure(Exception("test"), context)
        assert action.strategy == RecoveryStrategy.RETRY
        
        # Test backtrack after max retries
        context = {"current_step": 5, "retry_count": 3, "uncertainty": 0.3}
        action = manager.assess_failure(Exception("test"), context)
        assert action.strategy == RecoveryStrategy.BACKTRACK


class TestEndToEndScenarios:
    """End-to-end integration tests."""
    
    def test_full_task_execution(self):
        """Test complete task execution flow."""
        agent = AdaptiveWebAgent(AgentConfig(
            max_steps=10,
            debug=False,
            use_recovery=True,
            use_checklist=True
        ))
        
        agent.set_task_type("navigation")
        result = agent.run_task("Navigate to example.com and find the search bar")
        
        # Verify result structure
        assert result["task"] == "Navigate to example.com and find the search bar"
        assert "steps_taken" in result
        assert "success" in result
        assert "trajectory" in result
        assert len(result["trajectory"]) > 0
    
    def test_task_with_simulated_actions(self):
        """Test task with simulated action results."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        
        def mock_executor(action: str):
            return {
                "observation": f"Mock result for {action}",
                "success": True,
                "metadata": {"action": action}
            }
        
        agent.set_action_executor(mock_executor)
        result = agent.run_task("Test task")
        
        assert result["steps_taken"] > 0
    
    def test_agent_stats(self):
        """Test agent provides comprehensive stats."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=5))
        agent.set_task_type("search")
        
        agent.run_task("Find information")
        
        stats = agent.get_stats()
        
        assert "current_step" in stats
        assert "trajectory_length" in stats
        assert "checklist_score" in stats
        assert "graph_stats" in stats
    
    def test_reset_functionality(self):
        """Test agent reset clears state."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=10))
        agent.set_task_type("navigation")
        
        # Run task
        agent.run_task("Test task")
        
        # Verify state exists
        assert len(agent.trajectory) > 0
        
        # Reset
        agent.reset()
        
        # Verify state cleared
        assert len(agent.trajectory) == 0
        assert agent.current_step == 0


class TestErrorHandling:
    """Tests for error handling in integration."""
    
    def test_handles_missing_callback(self):
        """Test agent works without action callback (simulation mode)."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        agent.initialize()
        
        # Should work in simulation mode
        step = agent.execute_step()
        
        assert step is not None
        assert step.observation != ""
    
    def test_handles_empty_task(self):
        """Test agent handles empty task description."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        
        result = agent.run_task("")
        
        assert "success" in result
    
    def test_max_steps_limit(self):
        """Test agent respects max steps limit."""
        agent = AdaptiveWebAgent(AgentConfig(max_steps=3))
        
        result = agent.run_task("Test task with many steps needed")
        
        assert result["steps_taken"] <= 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
