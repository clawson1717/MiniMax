# PACE-CommonGround: Meta-Learning Multi-Agent System

## Concept
PACE-CommonGround (PACE-CG) is a meta-learning framework that orchestrates multi-agent systems via a **Personalized Adaptive Curriculum Engine (PACE)**. It dynamically assigns tasks by evaluating agents through **Distributed Partial Information Puzzles (DPIP)**, ensuring that agents only collaborate on tasks where they share a verified "common ground" of knowledge and capability. This process is monitored and self-regulated by a **Judge Reliability Harness (JRH)**, which ensures that the agents' outputs remain stable and consistent under minor textual or prompt perturbations.

## Novel Contribution
The key technical innovation is the **Closed-Loop Reliability-Curriculum (CLRC)** mechanism. Unlike standard multi-agent managers that use static roles or simple LLM-based routing, PACE-CG:
1. **Verifies Hidden State**: Uses DPIP to ensure agents actually possess the latent information/skills required for a task before assignment.
2. **Contextual Bandit Task Allocation**: Applies MAB (Multi-Armed Bandit) strategies to the Skill Graph to learn optimal agent-task pairings over time.
3. **Internal Calibration**: Uses the JRH as a high-fidelity reward signal that specifically penalizes "brittle" or hallucination-prone agent behaviors by measuring stability under minor input shifts.

## Roadmap
1. **Core Infrastructure**: Implement the Agent Protocol and the DPIP environment.
2. **Curriculum Engine**: Develop the PACE contextual bandit logic and Skill Graph ontology.
3. **Reliability Layer**: Build the JRH for N-way perturbation testing and agent calibration.
4. **Integration**: Link JRH feedback to PACE rewards for dynamic task re-assignment.
5. **Evaluation**: Benchmark the system against baseline reactive multi-agent architectures (e.g., AutoGen, LangGraph).

## Technical Implementation Details
- **PACE Engine**: Employs Thompson Sampling for balancing exploitation of known successful agent/task pairings with exploration of new potential matches.
- **DPIP Protocol**: Agents must solve asymmetric information puzzles where the solution is only reachable if they successfully exchange their uniquely held hidden states.
- **Judge Reliability Harness**: Measures the "Brittle Index" (BI) by calculating the Jensen-Shannon Divergence between agent outputs for the same task with minor prompt variations.
