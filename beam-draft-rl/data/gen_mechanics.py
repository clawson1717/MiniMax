import json
import random
import os

def solve_simply_supported_beam(L, P, a):
    """
    Solve for reactions in a simply supported beam with a point load.
    L: Length of the beam
    P: Magnitude of the point load
    a: Distance from the left support (R0) to the load
    
    Returns: (R0, RL)
    """
    # Sum of moments about RL = 0 => R0 * L - P * (L - a) = 0
    R0 = P * (L - a) / L
    # Sum of vertical forces = 0 => R0 + RL - P = 0
    RL = P - R0
    return round(R0, 4), round(RL, 4)

def generate_dataset(num_samples=50):
    dataset = []
    for _ in range(num_samples):
        L = round(random.uniform(1.0, 10.0), 2)
        P = round(random.uniform(10.0, 100.0), 2)
        a = round(random.uniform(0.0, L), 2)
        
        R0, RL = solve_simply_supported_beam(L, P, a)
        
        example = {
            "input": {
                "L": L,
                "P": P,
                "a": a
            },
            "ground_truth": {
                "R0": R0,
                "RL": RL
            }
        }
        dataset.append(example)
    return dataset

if __name__ == "__main__":
    num_samples = 50
    data = generate_dataset(num_samples)
    
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "beam_dataset.json")
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Generated {num_samples} examples in {output_path}")
