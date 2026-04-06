import logging
import argparse
import sys
from src.integrated_loop import IntegratedAdversarialLoop
from src.visualization import print_terminal_chart

def main():
    parser = argparse.ArgumentParser(description="Complex Domain Shift Scenario Test")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Domain Shift Scenario: Shifting from Legal Reasoning to Medical Reasoning
    # Initial reasoning context is legal, but we provide medical expert reference.
    
    agent_configs = [
        {"id": "legal_expert", "domain": "legal", "morphology": {"expertise": "law"}},
        {"id": "medical_researcher", "domain": "medical", "morphology": {"expertise": "medicine"}},
        {"id": "bioethicist", "domain": "ethics", "morphology": {"expertise": "philosophy"}},
    ]
    
    # Complex scenario: Informed consent for an experimental gene therapy.
    # Requires legal (regulatory), medical (efficacy/risk), and ethical (autonomy) reasoning.
    
    initial_query = """
    Evaluate the legal framework for informed consent in the context of phase I 
    gene therapy trials for pediatric patients with ultra-rare genetic disorders.
    Focus on the liability of the principal investigator.
    """
    
    # The expert reference shifts the focus heavily towards the MEDICAL/BIOLOGICAL nuances
    # which might be "high-entropy" or "adversarial" to a purely legal-focused initial probe.
    expert_medical_reference = """
    The medical priority for gene therapy in pediatric populations must balance 
    the vector-induced immunogenicity against the potential for permanent 
    phenotypic correction. Informed consent is not merely a legal shield (liability), 
    but a continuous dialogue regarding the biological uncertainty of 
    off-target insertions and the long-term metabolic consequences 
    of genomic modification. The 'gap' exists where the legal framework 
    ignores the stochastic nature of viral vector integration.
    """

    print(f"\n--- Starting Domain Shift Scenario Test ---")
    print(f"Initial Domain Focus: Legal Reasoning")
    print(f"Expert Reference Focus: Medical/Biological Nuance (Domain Shift)")
    print(f"Iterations: {args.iterations}\n")

    loop = IntegratedAdversarialLoop(
        agent_configs=agent_configs, 
        expert_reference=expert_medical_reference, 
        max_iterations=args.iterations
    )
    
    final_state = loop.run_iteration(initial_query)
    history = final_state["history"]
    
    print("\n--- Results ---")
    for entry in history:
        print(f"Iter {entry['iteration']+1}: Gap Score {entry['gap_score']:.4f}")
        # Print a small snippet of how the query evolves to see adversarial injection
        print(f"  Next Query (Adversarial): {entry['query'][:100]}...")
    
    print_terminal_chart(history)
    
    print(f"\nFinal Gap Score: {final_state['final_gap_score']:.4f}")
    if final_state['final_gap_score'] < history[0]['gap_score']:
        print("Convergence Check: PASSED (Gap decreased)")
    else:
        print("Convergence Check: FAILED or NEUTRAL (Gap did not decrease)")

if __name__ == "__main__":
    main()
