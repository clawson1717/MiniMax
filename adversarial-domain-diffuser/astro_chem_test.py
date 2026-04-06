import logging
import argparse
import sys
from src.integrated_loop import IntegratedAdversarialLoop
from src.visualization import print_terminal_chart

def main():
    parser = argparse.ArgumentParser(description="Astro-Chem Domain Shift Scenario Test")
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterations")
    args = parser.parse_args()

    # Use a more detailed logging format to observe internal transitions
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Domain Shift Scenario: Chemical Engineering -> Astrophysics Logic
    # We want to see if the system can resolve the high-entropy shift from 
    # fluid dynamics/reaction kinetics to planetary formation and stellar nucleosynthesis.
    
    agent_configs = [
        {"id": "chem_eng_optimizer", "domain": "chemical_engineering", "morphology": {"expertise": "process_kinetics"}},
        {"id": "stellar_physicist", "domain": "astrophysics", "morphology": {"expertise": "plasma_physics"}},
        {"id": "cosmochemist", "domain": "cosmochemistry", "morphology": {"expertise": "isotopic_analysis"}},
    ]
    
    # Initial query is purely Chemical Engineering focused (Industrial scale)
    initial_query = """
    Optimize the Haber-Bosch process for ammonia synthesis at 450°C and 200 atm. 
    Focus on the catalyst surface reaction kinetics of nitrogen dissociation 
    on iron-based clusters and the effect of potassium promoters on the 
    Langmuir-Hinshelwood rate equations.
    """
    
    # The expert reference is a radical shift: 
    # From industrial synthesis to the prebiotic chemistry in proto-planetary disks.
    # This introduces high-entropy adversarial inputs regarding non-equilibrium 
    # plasma and UV-driven radical chemistry in vacuum.
    expert_astro_reference = """
    In the context of the solar nebula, ammonia synthesis is dominated by 
    ice-mantle surface reactions on silicate grains at 10-30 Kelvin. 
    The high-entropy challenge is the flux of cosmic rays inducing 
    non-thermal desorption. Stellar UV radiation from the T-Tauri phase 
    dissociates N2 into N radicals, bypassing the high-pressure Haber-Bosch 
    kinetic barriers. The 'gap' is the transition from collision-dominated 
    industrial thermodynamics to radiation-dominated interstellar chemistry. 
    The logic must shift from pressure-vessel equilibrium to stochastic 
    particle-field interactions in a vacuum.
    """

    print(f"\n--- Starting ASTRO-CHEM Domain Shift Scenario Test ---")
    print(f"Initial Focus: Industrial Chemical Engineering")
    print(f"Target Focus: Astrophysics/Cosmochemistry (Adversarial Shift)")
    print(f"Iterations: {args.iterations}\n")

    # We use more iterations to observe the diffusion convergence
    loop = IntegratedAdversarialLoop(
        agent_configs=agent_configs, 
        expert_reference=expert_astro_reference, 
        max_iterations=args.iterations
    )
    
    final_state = loop.run_iteration(initial_query)
    history = final_state["history"]
    
    print("\n--- DETAILED ITERATION LOG ---")
    for entry in history:
        print(f"\n[ITERATION {entry['iteration'] + 1}]")
        print(f"Gap Score: {entry['gap_score']:.6f}")
        print(f"Adversarial Query (sample): {entry['query'][:200]}...")
        # Check if the query is actually incorporating the 'adversarial' astrophysical terms
        if "nebula" in entry['query'].lower() or "uv" in entry['query'].lower() or "t-tauri" in entry['query'].lower():
            print("  -> Domain injection detected in query.")
        else:
            print("  -> Domain injection NOT apparent in query yet.")
    
    print("\n--- VISUALIZATION ---")
    try:
        print_terminal_chart(history)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print(f"\nFinal Gap Score: {final_state['final_gap_score']:.6f}")
    
    # Analytical checks for reporting
    initial_gap = history[0]['gap_score']
    final_gap = final_state['final_gap_score']
    improvement = (initial_gap - final_gap) / initial_gap if initial_gap > 0 else 0
    
    print(f"Relative Improvement: {improvement:.2%}")
    
    if final_gap < initial_gap:
        print("Convergence Check: PASSED (System adapted to the shift)")
    else:
        print("Convergence Check: FAILED (System stagnated or diverged)")

if __name__ == "__main__":
    main()
