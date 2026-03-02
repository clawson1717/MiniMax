import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import ReasoningAgent
from src.trainer import MasteryTrainer
from src.pedagogy import BloomTier

def main():
    print("Initializing demo evaluation...")
    # Initialize with a tiny model if possible, but here we use the one defined in Agent
    # For demo purposes, we can mock the model to avoid loading weights if we want it fast,
    # but the prompt says "Run training cycles".
    # However, loading Qwen might be slow. Let's see if I can use a mock agent for the demo
    # to show the logic without waiting for GPU/Model loading.
    
    # Let's create a MockAgent that inherits from ReasoningAgent but doesn't load the model
    class MockAgent(ReasoningAgent):
        def __init__(self):
            print("Using MockAgent for demonstration.")
            self.device = "cpu"
            
        def generate_reasoning(self, prompt, num_samples=1, temperature=0.7):
            return ["Sample reasoning for: " + prompt] * num_samples
            
        def refine_reasoning(self, prompt, previous_reasoning, temperature=0.5):
            return previous_reasoning + " Refined with more detail and logic."
            
        def get_logits(self, prompt):
            # Return random logits for vocab size 1000
            return torch.randn(1, 1000)

    agent = MockAgent()
    trainer = MasteryTrainer(agent=agent)
    
    dataset = [
        {"prompt": "What is 2+2?", "tier": BloomTier.KNOWLEDGE},
        {"prompt": "Explain gravity.", "tier": BloomTier.COMPREHENSION},
        {"prompt": "Design a bridge.", "tier": BloomTier.SYNTHESIS}
    ]
    
    print("\n--- Running Training Cycles ---")
    for i, example in enumerate(dataset):
        metrics = trainer.train_step(example)
        print(f"Step {i+1}:")
        print(f"  Prompt: {metrics['prompt']}")
        print(f"  Success Score: {metrics['success_score']:.2f}")
        print(f"  Semantic Info Gain: {metrics['info_gain_bits']:.4f} bits")
        
    print("\n--- Evaluating World Model Capacity ---")
    capacity = trainer.evaluate_world_model(dataset, env_states=50)
    print(f"Aggregate World Model Capacity: {capacity:.4f} bits")

if __name__ == "__main__":
    main()
