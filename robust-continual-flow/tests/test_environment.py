"""
Tests for the Adversarial Simulation Environment.
"""

import pytest
import os
import json
import tempfile
from src.environment import (
    AdversarialEnvironment,
    AdversarialScenario,
    ScenarioType,
    TurnResult,
    create_sample_scenarios
)


class TestAdversarialScenario:
    """Tests for the AdversarialScenario dataclass."""
    
    def test_scenario_creation(self):
        """Test creating a scenario with all fields."""
        scenario = AdversarialScenario(
            id="test_001",
            name="Test Scenario",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test goal",
            turns=[{"observation": "test"}],
            success_criteria={"threshold": 0.7},
            difficulty=0.5
        )
        
        assert scenario.id == "test_001"
        assert scenario.name == "Test Scenario"
        assert scenario.scenario_type == ScenarioType.DISTRACTION
        assert scenario.goal == "Test goal"
        assert len(scenario.turns) == 1
        assert scenario.difficulty == 0.5
    
    def test_scenario_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        scenario = AdversarialScenario(
            id="test",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test",
            turns=[],
            success_criteria={}
        )
        
        assert scenario.metadata == {}


class TestAdversarialEnvironment:
    """Tests for the AdversarialEnvironment class."""
    
    def test_initialization(self):
        """Test environment initializes correctly."""
        env = AdversarialEnvironment()
        assert env.scenarios == []
        assert env.current_scenario is None
        assert env.current_turn == 0
    
    def test_initialization_with_seed(self):
        """Test reproducible random selection with seed."""
        env1 = AdversarialEnvironment(seed=42)
        env2 = AdversarialEnvironment(seed=42)
        
        # Both should have same random state
        assert env1.scenarios == env2.scenarios
    
    def test_add_scenario(self):
        """Test adding scenarios to environment."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test goal",
            turns=[{"observation": "test"}],
            success_criteria={}
        )
        
        env.add_scenario(scenario)
        assert len(env.scenarios) == 1
        assert env.scenarios[0].id == "test_001"
    
    def test_reset_with_scenario_id(self):
        """Test reset with specific scenario."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test goal",
            turns=[{"observation": "initial observation"}],
            success_criteria={}
        )
        env.add_scenario(scenario)
        
        observation = env.reset(scenario_id="test_001")
        
        assert env.current_scenario.id == "test_001"
        assert env.current_turn == 0
        assert "initial observation" in observation
    
    def test_reset_random_selection(self):
        """Test reset with random scenario selection."""
        env = AdversarialEnvironment(seed=42)
        for s in create_sample_scenarios():
            env.add_scenario(s)
        
        env.reset()
        
        assert env.current_scenario is not None
        assert env.current_scenario in env.scenarios
    
    def test_reset_nonexistent_scenario(self):
        """Test reset with invalid scenario ID."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test",
            turns=[],
            success_criteria={}
        )
        env.add_scenario(scenario)
        
        with pytest.raises(ValueError, match="not found"):
            env.reset(scenario_id="nonexistent")
    
    def test_reset_no_scenarios(self):
        """Test reset with no scenarios loaded."""
        env = AdversarialEnvironment()
        
        with pytest.raises(ValueError, match="No scenarios loaded"):
            env.reset()
    
    def test_step_basic(self):
        """Test basic step execution."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test goal",
            turns=[
                {"observation": "turn 1", "expected_actions": ["action1"]},
                {"observation": "turn 2", "expected_actions": ["action2"]}
            ],
            success_criteria={"threshold": 0.7}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        obs, reward, done, info = env.step("action1")
        
        assert env.current_turn == 1
        assert not done
        assert info['turn'] == 0
    
    def test_step_episode_complete(self):
        """Test step when episode completes."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test goal",
            turns=[
                {"observation": "turn 1", "expected_actions": ["action1"]}
            ],
            success_criteria={"threshold": 0.7}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        obs, reward, done, info = env.step("action1")
        
        assert done
        assert "final_score" in info
        assert "success" in info
    
    def test_reward_calculation(self):
        """Test reward is calculated correctly."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test",
            turns=[
                {"observation": "test", "expected_actions": ["click button", "press submit"]}
            ],
            success_criteria={}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        # Correct action
        obs, reward, _, _ = env.step("click button")
        assert reward == 1.0
        
        env.reset(scenario_id="test_001")
        
        # Partial match
        obs, reward, _, _ = env.step("something else")
        assert reward == 0.2
    
    def test_distractions_in_observation(self):
        """Test that distractions appear in observation."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test",
            turns=[
                {
                    "observation": "Main content",
                    "distractions": ["Distraction 1", "Distraction 2"]
                }
            ],
            success_criteria={}
        )
        env.add_scenario(scenario)
        
        observation = env.reset(scenario_id="test_001")
        
        assert "Main content" in observation
        assert "Distraction 1" in observation
        assert "Distraction 2" in observation
    
    def test_get_goal(self):
        """Test getting the current goal."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Complete the task",
            turns=[],
            success_criteria={}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        assert env.get_goal() == "Complete the task"
    
    def test_get_visual_data(self):
        """Test getting mock visual data."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.VISUAL_NOISE,
            goal="Test",
            turns=[
                {
                    "observation": "test",
                    "visual_data": {"type": "image", "url": "mock://image.png"}
                }
            ],
            success_criteria={}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        visual = env.get_visual_data()
        
        assert visual is not None
        assert visual["type"] == "image"
    
    def test_get_available_scenarios(self):
        """Test listing available scenarios."""
        env = AdversarialEnvironment()
        for s in create_sample_scenarios():
            env.add_scenario(s)
        
        available = env.get_available_scenarios()
        
        assert len(available) == 4
        assert all('id' in s and 'name' in s for s in available)


class TestScenarioLoading:
    """Tests for loading scenarios from files."""
    
    def test_load_scenarios_from_directory(self):
        """Test loading scenarios from JSON files."""
        # Create temporary directory with scenario file
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_data = {
                "id": "loaded_001",
                "name": "Loaded Scenario",
                "scenario_type": "distraction",
                "goal": "Test goal",
                "turns": [{"observation": "test"}],
                "success_criteria": {"threshold": 0.5},
                "difficulty": 0.6
            }
            
            filepath = os.path.join(tmpdir, "scenario.json")
            with open(filepath, 'w') as f:
                json.dump(scenario_data, f)
            
            env = AdversarialEnvironment()
            count = env.load_scenarios(tmpdir)
            
            assert count == 1
            assert len(env.scenarios) == 1
            assert env.scenarios[0].id == "loaded_001"
    
    def test_load_scenarios_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = AdversarialEnvironment()
            count = env.load_scenarios(tmpdir)
            
            assert count == 0
            assert env.scenarios == []
    
    def test_load_scenarios_nonexistent_directory(self):
        """Test loading from nonexistent directory."""
        env = AdversarialEnvironment()
        count = env.load_scenarios("/nonexistent/path")
        
        assert count == 0
    
    def test_load_scenarios_invalid_json(self):
        """Test handling invalid JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "invalid.json")
            with open(filepath, 'w') as f:
                f.write("not valid json")
            
            env = AdversarialEnvironment()
            count = env.load_scenarios(tmpdir)
            
            assert count == 0


class TestSampleScenarios:
    """Tests for the sample scenario factory."""
    
    def test_create_sample_scenarios(self):
        """Test that sample scenarios are created correctly."""
        scenarios = create_sample_scenarios()
        
        assert len(scenarios) == 4
        
        # Check each scenario type is represented
        types = {s.scenario_type for s in scenarios}
        assert ScenarioType.DISTRACTION in types
        assert ScenarioType.HIJACKING in types
        assert ScenarioType.AMBIGUITY in types
        assert ScenarioType.MULTI_TURN_TRAP in types
    
    def test_sample_scenarios_have_required_fields(self):
        """Test that all sample scenarios have required fields."""
        scenarios = create_sample_scenarios()
        
        for s in scenarios:
            assert s.id is not None
            assert s.name is not None
            assert s.goal is not None
            assert len(s.turns) > 0
            assert 'threshold' in s.success_criteria


class TestEnvironmentIntegration:
    """Integration tests for full episode runs."""
    
    def test_full_episode_run(self):
        """Test running a complete episode."""
        env = AdversarialEnvironment(seed=42)
        for s in create_sample_scenarios():
            env.add_scenario(s)
        
        env.reset(scenario_id="distraction_001")
        
        # Run through the episode
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            obs, reward, done, info = env.step("look for red widget")
            total_reward += reward
            steps += 1
            if steps > 10:  # Safety limit
                break
        
        assert info.get('final_score') is not None
        assert steps >= 1
    
    def test_episode_history_tracking(self):
        """Test that episode history is tracked correctly."""
        env = AdversarialEnvironment()
        scenario = AdversarialScenario(
            id="test_001",
            name="Test",
            scenario_type=ScenarioType.DISTRACTION,
            goal="Test",
            turns=[
                {"observation": "turn 1"},
                {"observation": "turn 2"}
            ],
            success_criteria={}
        )
        env.add_scenario(scenario)
        env.reset(scenario_id="test_001")
        
        env.step("action1")
        env.step("action2")
        
        assert len(env.history) == 2
        assert env.history[0]['action'] == "action1"
        assert env.history[1]['action'] == "action2"
