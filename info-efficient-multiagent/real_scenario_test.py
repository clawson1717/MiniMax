
from src.agent import create_agent_pool
from src.environment import MultiAgentEnvironment
from src.routing import MessageRouter, RouteMode
from src.coordinator import DiffusionCoordinator
import numpy as np

def run_real_scenario():
    print("--- REAL SCENARIO TEST ---")
    
    # 1. Setup Agents
    print("\n[1] Creating Agent Pool...")
    # Create 5 agents with different specializations
    agents = create_agent_pool(num_agents=5)
    
    # Manually set some interesting capacities for testing
    for i, agent in enumerate(agents):
        # We simulate that some agents have been 'trained' or 'measured'
        # The project lacks a simple way to 'set' capacity other than internal attributes
        agent._capacity = 0.2 + (i * 0.15) 
        print(f"Agent: {agent.agent_id}, Specializations: {agent.specializations}, Simulated Capacity: {agent._capacity:.2f}")

    # 2. Setup Environment
    print("\n[2] Registering Agents in Environment...")
    env = MultiAgentEnvironment(name="production-env")
    for agent in agents:
        env.register_agent(agent.agent_id, agent, capacity=agent._capacity)

    # 3. Message Routing
    print("\n[3] Routing a Complex Query...")
    router = MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
    
    complex_query = "Calculate the derivative of x^2 + 5x and explain the second derivative's meaning in physics."
    
    # We want the top 2 'expert' agents
    routed_messages = router.route_to_high_capacity(
        sender_id="coordinator_sim",
        content=complex_query,
        top_k=2
    )
    
    print(f"Routed to {len(routed_messages)} agents:")
    recipient_ids = [m.receiver_id for m in routed_messages]
    for rid in recipient_ids:
        print(f" - {rid} (Capacity: {env.agent_states[rid].capacity:.2f})")

    # 4. Agent Execution (Friction Point!)
    print("\n[4] Collecting Responses...")
    # ISSUE: The router doesn't automatically trigger the agents. 
    # The user has to manually find which agents received messages and call them.
    responses = {}
    for msg in routed_messages:
        agent = env.agents[msg.receiver_id]
        # Generate response
        resp_list = agent.generate(msg.content)
        responses[agent.agent_id] = resp_list[0].content
        print(f"\nResponse from {agent.agent_id}:")
        print(f"---")
        print(responses[agent.agent_id])
        print(f"---")

    # 5. Coordination (The 'Diffusion' promise)
    print("\n[5] Attempting to Coordinate via Diffusion...")
    coordinator = DiffusionCoordinator(env)
    
    # ISSUE: Starting diffusion... but wait, how do I pass the AGENT RESPONSES to the coordinator?
    # Looking at src/coordinator.py, run_diffusion only takes a task_id and prompt.
    # It doesn't take the actual text responses gathered in step 4.
    state = coordinator.run_diffusion("complex_task_1", complex_query)
    
    print(f"Diffusion completed in {state.current_step} steps.")
    
    # Convert to response
    final_output = coordinator.vector_to_response(state)
    print("\nFinal Coordinator Output (Vector Analysis):")
    print(final_output)
    
    print("\n[CRITICAL FINDING] The coordinator produced a vector analysis but NO TEXT.")
    print("There is no 'decoding' from the diffusion latent space back to natural language.")

if __name__ == "__main__":
    run_real_scenario()
