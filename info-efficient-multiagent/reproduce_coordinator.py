
import numpy as np
from src.environment import MultiAgentEnvironment, AgentRole
from src.agent import create_agent
from src.coordinator import create_coordinator

def test_functioning_coordinator():
    print("Testing Diffusion Coordinator with actual agents...")
    
    # Setup environment with actual agents
    env = MultiAgentEnvironment(name="test-env")
    for i in range(3):
        agent = create_agent(agent_id=f"worker_{i}")
        # Vary capacity to see weighting
        capacity = 0.3 + (i * 0.3)  # 0.3, 0.6, 0.9
        env.register_agent(f"worker_{i}", agent, role=AgentRole.WORKER, capacity=capacity)
        print(f"Registered worker_{i} with capacity {capacity:.2f}")

    # Create coordinator
    coordinator = create_coordinator(environment=env, num_steps=10)
    
    # Run diffusion
    task_id = "test_task"
    print(f"\nRunning diffusion for task: {task_id}")
    state = coordinator.run_diffusion(task_id, "Solve this complex problem")
    
    # Check results
    response = coordinator.vector_to_response(state)
    
    print(f"\nDiffusion Results:")
    print(f"  Status: {response['status']}")
    print(f"  Steps: {response['steps_completed']}")
    print(f"  Final Entropy: {response['final_entropy']:.4f}")
    print(f"  Total Contributors: {response['total_contributors']}")
    
    print("\nAgent Influences:")
    for aid, influence in response['agent_influences'].items():
        print(f"  {aid}: {influence:.4f}")

    if response['total_contributors'] == 3:
        print("\n✅ Success: All agents contributed.")
    else:
        print(f"\n❌ Failure: Expected 3 contributors, got {response['total_contributors']}.")

if __name__ == "__main__":
    test_functioning_coordinator()
