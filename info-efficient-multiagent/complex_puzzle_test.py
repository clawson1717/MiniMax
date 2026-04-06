
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agent import ReasoningAgent, AgentConfig, SpecializedAgent
from src.environment import MultiAgentEnvironment, AgentRole
from src.routing import MessageRouter, RouteMode
from src.coordinator import DiffusionCoordinator, CoordinatorConfig
from src.capacity import InformationCapacityEstimator

def run_complex_puzzle_test():
    print("🧩 TESTING COMPLEX MULTI-DOMAIN LOGIC PUZZLE")
    print("=" * 60)

    # 1. Define the Puzzle
    puzzle = """
    SCENARIO: A 10kg experimental fusion battery (antique status) is swinging on a 5m frictionless 
    carbon-nanotube tether. At the lowest point of its swing (v=10m/s), a smart-contract-bot 
    triggers a laser that severs the tether because the battery's market value dropped below 
    $1M (a 'liquidation event'). 
    
    COMPLEX QUERY: 
    1. (Physics) What is the horizontal distance the battery travels before hitting a platform 2m below?
    2. (Law) The smart contract has a 'Common Good' clause stating liquidation is void if it 
       results in 'unreasonable destruction of heritage'. Is the bot liable for the impact damage?
    3. (Economics) Calculate if the liquidation was 'Pareto Optimal' given the battery's 
       re-sale value vs the cost of platform repairs ($50k).
    """

    # 2. Setup Specialized Agents
    # We'll give them specific "knowledge" (simulated via mock responses)
    
    physics_agent = SpecializedAgent(
        config=AgentConfig(agent_id="Physics_Expert", specializations=["physics", "kinematics"]),
        domain_knowledge={
            "physics": [
                "Using d = v * sqrt(2h/g). With v=10m/s and h=2m, d = 10 * sqrt(4/9.8) ≈ 6.39m.",
                "The conservation of momentum is interrupted by the external force of the laser."
            ]
        }
    )
    # Give Physics expert high capacity for physics
    physics_agent._capacity = 0.9

    law_agent = SpecializedAgent(
        config=AgentConfig(agent_id="Legal_Expert", specializations=["law", "contracts"]),
        domain_knowledge={
            "law": [
                "The 'Common Good' clause likely overrides the automated liquidation trigger if heritage destruction is imminent.",
                "Liability depends on whether 'unreasonable' is defined by value or by physical state."
            ]
        }
    )
    law_agent._capacity = 0.85

    econ_agent = SpecializedAgent(
        config=AgentConfig(agent_id="Econ_Expert", specializations=["economics", "markets"]),
        domain_knowledge={
            "economics": [
                "Liquidation is only Pareto Optimal if no party is worse off. The platform owner loses $50k, making it non-optimal.",
                "Market volatility doesn't justify physical destruction in a rational actor model."
            ]
        }
    )
    econ_agent._capacity = 0.82

    general_agent = ReasoningAgent(
        config=AgentConfig(agent_id="General_Bot", specializations=["general"])
    )
    general_agent._capacity = 0.3

    agents = [physics_agent, law_agent, econ_agent, general_agent]

    # 3. Setup Environment
    env = MultiAgentEnvironment(name="puzzle-env")
    for agent in agents:
        env.register_agent(agent.agent_id, agent, capacity=agent.get_current_capacity() or getattr(agent, '_capacity', 0.1))

    # 4. CAPACITY-WEIGHTED ROUTING
    print("\n[STEP 1] Capacity-Weighted Routing")
    router = MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
    routed_messages = router.route_to_high_capacity(
        sender_id="user",
        content=puzzle,
        top_k=3
    )
    
    print(f"Selected Experts for the puzzle:")
    for msg in routed_messages:
        print(f" -> {msg.receiver_id} (Capacity: {env.agent_states[msg.receiver_id].capacity:.2f})")

    # 5. EXECUTION & CONFLICTING INPUTS
    print("\n[STEP 2] Collecting Expert Inputs")
    expert_responses = {}
    for msg in routed_messages:
        agent = env.agents[msg.receiver_id]
        resp = agent.generate(msg.content)[0]
        expert_responses[agent.agent_id] = resp.content
        print(f"\n--- Output from {agent.agent_id} ---")
        print(resp.content)

    # 6. COORDINATION (THE CORE TEST)
    print("\n[STEP 3] Diffusion-Based Coordination (OMAD)")
    coordinator = DiffusionCoordinator(env, config=CoordinatorConfig(num_steps=15))
    
    # We hit the friction point here: run_diffusion doesn't actually take the 'expert_responses'!
    # It just takes the prompt and runs its internal simulation.
    state = coordinator.run_diffusion("puzzle_task", puzzle)
    
    print(f"\nCoordination metadata after {state.current_step} steps:")
    print(f" - Final Entropy: {state.entropy:.4f}")
    
    # Let's see the 'influence' map
    analysis = coordinator.vector_to_response(state)
    print("\nAgent Influence in Final Decision:")
    for aid, influence in analysis['agent_influences'].items():
        print(f" - {aid}: {influence:.4f}")

    # 7. CRITICAL GAP ANALYSIS
    print("\n[CRITICAL EVALUATION]")
    print("1. Routing Efficiency: " + ("PASS" if "General_Bot" not in [m.receiver_id for m in routed_messages] else "FAIL"))
    print("2. Information Integration: FAIL - The coordinator does NOT use the agent's text responses.")
    print("3. Response Legibility: FAIL - The final 'output' is a vector summary, not a synthesized answer.")
    print("4. Conflict Resolution: UNKNOWN - Since text isn't processed, we can't tell if it resolved the Law vs Econ conflict.")

if __name__ == "__main__":
    run_complex_puzzle_test()
