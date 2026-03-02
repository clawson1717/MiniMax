import unittest
from unittest.mock import MagicMock, patch
from src.agent import NavigationAgent

class TestNavigationAgent(unittest.TestCase):
    def setUp(self):
        # Mock environment
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = "initial_state"
        self.mock_env.step.return_value = ("next_state", 0, False, {})
        
        # Mock model
        self.mock_model = MagicMock()
        # Returns 3 identical votes to keep uncertainty low initially
        self.mock_model.return_value = ["action1", "action1", "action1"]

    def test_run_basic_loop(self):
        agent = NavigationAgent(
            env=self.mock_env,
            model=self.mock_model,
            base_compute=3,
            uncertainty_threshold=0.5
        )
        
        result = agent.run(goal="find the treasure", max_steps=2)
        
        # Verify env interactions
        self.mock_env.reset.assert_called_once()
        self.assertGreaterEqual(self.mock_env.step.call_count, 1)
        
        # Verify model interactions
        self.assertGreaterEqual(self.mock_model.call_count, 2) # 2 steps * (preliminary + final)
        
        # Verify result structure
        self.assertIn("trajectory", result)
        self.assertIn("status", result)
        self.assertEqual(len(result["trajectory"]["nodes"]), 3) # Initial + 2 steps
        self.assertEqual(len(result["trajectory"]["edges"]), 2)

    def test_scaling_logic_integration(self):
        # Model returns divergent votes to trigger scaling
        self.mock_model.side_effect = [
            ["a", "b", "c"], # Preliminary (high uncertainty)
            ["a", "a", "a", "a", "a", "a"], # Final (scaled budget)
            ["x", "x", "x"], # Step 2 Preliminary
            ["x", "x", "x"]  # Step 2 Final
        ]
        
        agent = NavigationAgent(
            env=self.mock_env,
            model=self.mock_model,
            base_compute=3,
            uncertainty_threshold=0.2
        )
        
        agent.run(goal="test scaling", max_steps=2)
        
        # First call was for 3 votes. 
        # Uncertainty for ["a", "b", "c"] is 1.0.
        # Excess ratio = (1.0 - 0.2) / (1.0 - 0.2) = 1.0
        # Multiplier = 1.0 + 1.0 = 2.0
        # Scaled compute = 3 * 2.0 = 6
        
        # The second call to model (final_votes for step 0) should have budget 6
        self.mock_model.assert_any_call(unittest.mock.ANY, 6)

    def test_monitoring_alerts(self):
        # High uncertainty and long trajectory to trigger fatigue
        self.mock_model.return_value = ["a", "b", "c"] # Always uncertain
        
        agent = NavigationAgent(
            env=self.mock_env,
            model=self.mock_model,
            base_compute=3,
            uncertainty_threshold=0.5
        )
        
        # Run for many steps to increase fatigue score
        with patch('builtins.print') as mock_print:
            agent.run(goal="fatigue test", max_steps=5)
            # Check if any alerts were printed
            # Fatigue calculation includes length, repetition, and uncertainty.
            # With constant high uncertainty and increasing length, it should eventually alert.
            self.assertTrue(mock_print.called)

if __name__ == "__main__":
    unittest.main()
