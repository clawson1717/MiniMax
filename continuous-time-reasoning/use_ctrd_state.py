import torch
from src.state import ReasoningState

def simulate_trajectory(steps=5):
    trajectory = []
    hidden_dim = 16
    
    print(f"Simulating a trajectory of {steps} states...")
    
    for i in range(steps):
        # Create a mock hidden state tensor
        hidden_state = torch.randn(1, hidden_dim)
        
        # Create a ReasoningState instance
        # Note: Step 3 will use torchdiffeq, so timestamp 't' is crucial
        state = ReasoningState(
            timestamp=float(i),
            hidden_state=hidden_state,
            confidence=0.1 * i,
            uncertainty=1.0 / (i + 1)
        )
        trajectory.append(state)
        print(f"Step {i}: Created state at t={state.timestamp}")
        
    return trajectory

def test_serialization(trajectory):
    print("\nTesting serialization to list of dicts...")
    serialized = []
    for i, state in enumerate(trajectory):
        try:
            # Simple conversion to dict for now
            d = {
                "timestamp": state.timestamp,
                "hidden_state": state.hidden_state,
                "confidence": state.confidence,
                "uncertainty": state.uncertainty
            }
            serialized.append(d)
            print(f"Step {i} serialized successfully.")
        except Exception as e:
            print(f"Step {i} serialization failed: {e}")
    
    return serialized

if __name__ == "__main__":
    try:
        traj = simulate_trajectory()
        dicts = test_serialization(traj)
        print(f"\nSuccessfully serialized {len(dicts)} states.")
        # Print the first one to see structure
        if dicts:
            first = dicts[0]
            print(f"Sample keys: {list(first.keys())}")
            print(f"Hidden state type in dict: {type(first['hidden_state'])}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
