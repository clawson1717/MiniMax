import pytest
import torch
import numpy as np
from src.model_wrapper import DraftWrapper
from src.trainer import BDRLTrainer
from src.reward import VerifiableReward
from src.sensor import UncertaintySensor
from src.curriculum import CurriculumManager, TrainingStage
from src.engine import PhysicsEngine

def test_bdrl_integration_pipeline():
    """
    Integration test for the full beam-draft-rl pipeline.
    Scenario:
    1. Define a beam problem (inputs).
    2. Pass through DraftWrapper to augment prompt.
    3. Simulate model output with draft tokens.
    4. Extract draft and verify with VerifiableReward and PhysicsEngine.
    5. Check uncertainty with UncertaintySensor.
    6. Update curriculum stage via CurriculumManager.
    7. Simulate a trainer-like update step (mocking the heavy lifting).
    """
    
    # 1. Setup components
    wrapper = DraftWrapper(model_name="test-model")
    reward_fn = VerifiableReward()
    sensor = UncertaintySensor()
    curriculum = CurriculumManager(mastery_threshold=0.8, window_size=5)
    engine = PhysicsEngine()
    
    # 2. Problem definition: Simple beam with 10N at L/2
    beam_input = "Simple beam, L=10m, Force=10N at 5m. Calculate reactions at A and B."
    ground_truth = "RA=5N, RB=5N"
    
    # 3. Prompt wrapping
    wrapped_prompt = wrapper.wrap_prompt(beam_input)
    assert "<|draft_start|>" in wrapped_prompt
    assert "<|draft_end|>" in wrapped_prompt
    
    # 4. Simulate model output (Chain of Thought + Draft + Final Answer)
    # The model "thinks" and provides a draft before the final answer.
    simulated_output = (
        "Thinking: The force is symmetric.\n"
        "<|draft_start|>\n"
        "RA + RB = 10\n"
        "Sum of moments at A: 10 * 5 - RB * 10 = 0 => RB = 5\n"
        "RA = 5\n"
        "<|draft_end|>\n"
        "Final Answer: RA=5N, RB=5N"
    )
    
    # 5. Pipeline processing
    # Extract draft
    draft_content = wrapper.extract_draft(simulated_output)
    assert "RA = 5" in draft_content
    
    # Verify reward
    reward = reward_fn.calculate_reward(simulated_output, ground_truth)
    assert reward == 1.0
    
    # Physics engine check (Internal consistency)
    # Let's say we check the equations in the draft
    equations = ["5N + 5N - 10N = 0"] # Simplified for test
    is_physically_sound = reward_fn.verify_equations(equations)
    assert is_physically_sound is True
    
    # 6. Sensor check (Uncertainty/Noise)
    # Simulate high confidence (low entropy)
    probs = [0.95, 0.05]
    entropy = sensor.calculate_entropy(probs)
    assert entropy < 0.5
    
    # 7. Curriculum update
    # Simulate a series of successes to trigger stage transition
    # Set min_samples_per_stage low for testing
    curriculum.min_samples_per_stage = 5
    for _ in range(10):
        curriculum.update_mastery(1.0)
    
    initial_stage = TrainingStage.FULL_COT
    assert curriculum.get_current_stage() != initial_stage # Should have progressed
    
    # 8. Trainer integration (Mocked)
    # In a real scenario, BDRLTrainer would use these rewards to update weights.
    # Here we just verify the trainer can be instantiated with default logic.
    # Note: We don't call setup_model() to avoid downloading a real LLM in CI.
    trainer = BDRLTrainer(model_name="hf-internal-testing/tiny-random-gpt2")
    assert trainer.model_name == "hf-internal-testing/tiny-random-gpt2"
    assert trainer.ppo_epochs == 4

if __name__ == "__main__":
    # Allow running this test script directly
    test_bdrl_integration_pipeline()
    print("Integration test passed!")
