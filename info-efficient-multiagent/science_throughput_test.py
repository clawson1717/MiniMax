
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from src.agent import create_agent_pool, AgentConfig, SpecializedAgent
from src.environment import MultiAgentEnvironment
from src.routing import MessageRouter, RouteMode
from src.coordinator import DiffusionCoordinator
import numpy as np

def run_science_comparison_test():
    print("--- 🔬 HIGH-THROUGHPUT SCIENCE THEORY COMPARISON TEST ---")
    
    # 1. Setup Agents with specific expertise
    print("\n[1] Creating Specialized Agent Pool...")
    
    # We'll create 3 experts for the 3 theories
    # Relative Theory Expert
    relativity_expert = SpecializedAgent(
        config=AgentConfig(
            agent_id="relativity_expert", 
            specializations=["relativity", "physics"],
            model_name="expert-1"
            ),
        domain_knowledge={
            "relativity": ["Space and time are linked in a four-dimensional continuum known as spacetime.",
                          "Gravity is the curvature of spacetime caused by mass and energy.",
                          "Time dilation occurs as an object approaches the speed of light."]
        }
    )
    relativity_expert._capacity = 0.92
    
    # Quantum Mechanics Expert
    quantum_expert = SpecializedAgent(
        config=AgentConfig(
            agent_id="quantum_expert", 
            specializations=["quantum mechanics", "physics"],
            model_name="expert-2"
        ),
        domain_knowledge={
            "quantum mechanics": ["Particles exist in multiple states simultaneously until observed.",
                                 "Energy is quantized into discrete packets called quanta.",
                                 "Heisenberg's uncertainty principle limits precision of paired measurements."]
        }
    )
    quantum_expert._capacity = 0.88
    
    # String Theory Expert
    string_expert = SpecializedAgent(
        config=AgentConfig(
            agent_id="string_expert", 
            specializations=["string theory", "physics"],
            model_name="expert-3"
        ),
        domain_knowledge={
            "string theory": ["Fundamental particles are one-dimensional oscillating strings.",
                             "The theory requires 10 or 11 dimensions to be mathematically consistent.",
                             "It aims to reconcile general relativity and quantum mechanics."]
        }
    )
    string_expert._capacity = 0.85
    
    # Generalist (Low capacity)
    generalist = SpecializedAgent(
        config=AgentConfig(
            agent_id="generalist", 
            specializations=["general"],
            model_name="general-1"
        )
    )
    generalist._capacity = 0.30
    
    agents = [relativity_expert, quantum_expert, string_expert, generalist]

    # 2. Setup Environment
    print("\n[2] Registering Agents in Environment...")
    env = MultiAgentEnvironment(name="science-env")
    for agent in agents:
        env.register_agent(agent.agent_id, agent, capacity=agent._capacity)

    # 3. Capacity-Weighted Routing
    print("\n[3] Routing complex prompt...")
    router = MessageRouter(env, default_mode=RouteMode.CAPACITY_WEIGHTED)
    
    theory_prompt = "Compare General Relativity, Quantum Mechanics, and String Theory. How do they treat the concept of a 'particle' and 'gravity' differently?"
    
    # The router should pick the top 3 experts and ignore the generalist
    routed_messages = router.route_to_high_capacity(
        sender_id="coordinator",
        content=theory_prompt,
        top_k=3
    )
    
    print(f"Routed to {len(routed_messages)} agents:")
    recipient_ids = [m.receiver_id for m in routed_messages]
    for rid in recipient_ids:
        print(f" - {rid} (Capacity: {env.agent_states[rid].capacity:.2f})")
    
    if "generalist" in recipient_ids:
        print("❌ FAIL: Generalist was selected despite being low capacity.")
    else:
        print("✅ SUCCESS: Only high-capacity experts selected.")

    # 4. Agent Execution (Internal Friction Observation)
    print("\n[4] Collecting Responses (Manual step required in current API)...")
    responses = {}
    for msg in routed_messages:
        agent = env.agents[msg.receiver_id]
        resp_list = agent.generate(msg.content)
        responses[agent.agent_id] = resp_list[0].content
        print(f"\nResponse from {agent.agent_id}:")
        print(f"---")
        print(responses[agent.agent_id][:200] + "...") # Truncated
        print(f"---")

    # 5. OMAD Diffusion Coordination
    print("\n[5] Coordinating via OMAD Diffusion...")
    coordinator = DiffusionCoordinator(env)
    
    # We'll observe if the diffusion process converges
    state = coordinator.run_diffusion("science_task", theory_prompt)
    
    print(f"Diffusion completed in {state.current_step} steps.")
    print(f"Final Entropy: {state.entropy:.4f}")
    
    # Check agent influences
    summary = coordinator.vector_to_response(state)
    print("\nAgent Influences on Final State:")
    for aid, influence in summary['agent_influences'].items():
        print(f" - {aid}: {influence:.4f}")
    
    # Validate influence ordering
    influences = summary['agent_influences']
    sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)
    
    top_voter = sorted_influences[0][0]
    print(f"\nTop Influencer: {top_voter}")
    
    if top_voter == "relativity_expert":
         print("✅ Influence correctly matches highest capacity.")
    else:
         print(f"ℹ️ {top_voter} was the top influencer (Capacities: Rel=0.92, Quant=0.88, String=0.85)")

if __name__ == "__main__":
    run_science_comparison_test()
