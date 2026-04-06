# Project Usage Log - [2026-03-04]

## Adversarial Domain Diffuser (ADD) Testing

### Context
Tested the diffusion-based noise-injection defense and adversarial loop with a **complex domain shift scenario**: shifting from **Chemical Engineering** (industrial scale) to **Astrophysics/Cosmochemistry** (stochastic vacuum kinetics).

### Test Scenario: "Astro-Chem Haber-Bosch Synthesis"
- **Initial Query (Chem-Eng focus):** Industrial optimize Haber-Bosch at 450°C and 200 atm.
- **Expert Reference (Astro focus):** Prebiotic chemistry in proto-planetary disks, 10-30 Kelvin, silicate grains, and UV-driven radical chemistry.
- **Goal:** Observe how the system handles the high-entropy jump from high-pressure equilibrium to radiation-dominated vacuum chemistry.

### Findings & Observations

#### 1. Adversarial Injection Effectiveness
- **Observation:** The `AdversarialGenerator` correctly identified gaps in industrial chemical kinetics and injected astrophysical concepts like "stochastic" and "vacuum" into the subsequent adversarial prompts.
- **Result:** The system successfully forced the chemical engineering model to confront non-equilibrium states it wasn't initially designed for.

#### 2. Denoising & Convergence (OMAD)
- **Observation:** Under the current mock implementation, the "denoising" steps in the `DiffusionPolicy` are purely stochastic, leading to an **oscillatory gap score** (e.g., jumping between 0.5518 and 0.5630).
- **Limitation:** The convergence check passed because the gap decreased from the initial state (0.7321), but it hit a "limit cycle" rather than a steady-state convergence. This suggests the denoising steps aren't yet capturing higher-order domain integration.
- **High-Entropy Inputs:** The system "accepted" the high-entropy astrophysical inputs without crashing, but the reasoning agents (as currently mocked) didn't fully synthesize the two disparate domains into a unified "Astro-Chem" model.

#### 3. UX and Reporting
- **Observation:** The detailed log in `astro_chem_test.py` highlighted that while the system was "detecting" domain shifts, the adversarial queries often defaulted to a generic "Wait, the previous analysis missed 'keyword'..." template.
- **UX Friction:** The coordination step in `OMADOrchestrator` is a "black box" in the current CLI output. It's hard to see *how* the diffusion policies are being updated or if the "noise" Is being successfully rejected.

### Bugs & Issues
- **Bug (Adversarial Generator):** The mock keyword list in `src/adversarial_gen.py` was hardcoded to only include medical terms, which broke the "Astro-Chem" test scenario initially. I had to manually expand the vocabulary to support the domain shift.
- **Inconsistency:** The `IntegratedAdversarialLoop.evaluate_performance` logic was also hardcoded for medical keywords. This makes the system currently "domain-locked" in its mock state.
- **Issue:** The convergence check in the demo scripts is too simplistic; it only compares the first and last gap scores, which can be misleading if the system is oscillating.

### Feature Ideas
- **Dynamic Vocabulary Discovery:** The `AdversarialGenerator` should automatically extract keywords from the `expert_reference` instead of relying on a hardcoded list.
- **Entropy-Based Stopping Criterion:** Stop the iterations once the variance in the `OMAD` trajectory falls below a certain threshold, indicating the system has "settled" into the new domain.
- **Cross-Domain Mapping:** A tool to visualize the "semantic distance" between the Agent's current domain (e.g., Chemical) and the Target domain (e.g., Astro) to better understand the diffusion path.

### Conclusion
The architecture successfully supports the "Astro-Chem" high-entropy shift, but the current implementation is heavily biased towards the medical domain in its default mock state. Significant improvements are needed in the "integrative" logic of the reasoning agents to move beyond keyword matching towards true domain-synthesis.
