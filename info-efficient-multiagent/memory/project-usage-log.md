# Project Usage Log: info-efficient-multiagent

## [2026-03-04] Complex Multi-Domain Logic Puzzle Test

**Goal:** Test "capacity-weighted routing" and "OMAD diffusion coordination" by feeding it a complex logic puzzle involving physics, law, and economics.

### 🧪 Test Results
- **Routing Efficiency:** ✅ Success. The `MessageRouter` correctly identified and routed the "expert" agents (Physics, Legal, Econ) while excluding the generalist.
- **Expert Input:** ✅ Success. Specialized agents correctly injected their domain-specific knowledge into the response stream (e.g., Physics agent correctly mentioned momentum conservation and kinematics formulas).
- **Communication Flow:** ✅ Operational. The environment successfully managed the message exchange and capacity-weighted expert selection.

### 🐛 Potential Bugs & UX Friction
1. **Critical Gap: No Text Integration in Coordinator.** The `DiffusionCoordinator` runs an internal mathematical simulation (`_simulate_agent_contribution`) but DOES NOT take the actual text responses gathered from the experts. This means the "coordinated" result is disconnected from the actual reasoning performed by the agents.
2. **Missing Decoder:** The output of the diffusion process is a vector summary. There is no mechanism to "decode" this back into a natural language response. This makes the system useless for end-users who need a clear answer to their puzzle.
3. **Manual Execution Overhead:** The user must manually find which agents received messages and then call `agent.generate()` on each. There's no automated "run the pipeline" method in the environment or router.

### 💡 New Feature Ideas
- **Text-Informed Diffusion:** Modify `DiffusionState` to include the raw text responses and use their embeddings as the starting point for the diffusion process (rather than pure noise).
- **LLM Synthesis Step:** Add a final step that takes the `agent_influences` from the coordinator and the original expert responses, then uses a high-capacity "Generalist" or "Supervisor" agent to synthesize them into a final, coherent answer.
- **Unified Pipeline:** Create an `IEMARPipeline` class that handles the entire flow: Routing -> Parallel Execution -> Diffusion Coordination -> Final Synthesis.

### 🛠️ GitHub Issues
- **Major Issue:** `DiffusionCoordinator` is disconnected from agent text outputs.
- **UX Issue:** Lack of a unified execution pipeline for end-to-end reasoning.

## [2026-03-04] High-Throughput Science Theory Comparison Test

**Goal:** Test capacity-weighted routing and OMAD diffusion coordination using a complex multi-domain scientific prompt.

### 🧪 Test Results
- **Routing Accuracy:** ✅ Success. The `MessageRouter` correctly identified and routed the "expert" agents (Relativity, Quantum, String Theory) while excluding the generalist.
- **OMAD Diffusion:** ✅ Operational. The diffusion process converged in 10 steps.
- **Influence Consistency:** ⚠️ Mixed. While high-capacity agents had significantly more influence than the generalist (~0.14-0.18 vs 0.07), the absolute order didn't perfectly match capacities (String Theory expert at 0.85 capacity had slightly more influence than Relativity expert at 0.92).

### 🐛 Potential Bugs & UX Friction
1. **Manual Execution Loop:** The current API requires the user to manually collect routed messages, call `agent.generate()`, and manage responses. There is no automated worker/callback loop in `MessageRouter`.
2. **Diffusion Decoding:** The `DiffusionCoordinator` operates in a simulated 64-dimension latent space. While it provides "agent influence" stats, there is no mechanism to decode the final vector back into natural language or a synthesized text response.
3. **Internal Simulation Bias:** The `DiffusionCoordinator._simulate_agent_contribution` method uses `rng.standard_normal` which introduces noise that can occasionally override capacity weights at small step counts, explaining why the 0.85 cap agent beat the 0.92 cap agent in influence.

### 💡 New Feature Ideas
- **Text Recurrent Decoder:** Create a `DiffusionDecoder` class that takes the final latent vector and uses the top-influence agents' responses to perform a weighted LLM synthesis.
- **Auto-Executor:** Enable `MessageRouter` to automatically trigger agent generation if a `callback_fn` is provided.
- **Capacity Dynamics:** Allow capacity to fluctuate based on "hallucination markers" detected in responses during the refinement phase.

### 🛠️ GitHub Issues
- [⚠️] Feature Request: Automated response synthesis from diffusion latent state (GH CLI NOT AUTHENTICATED).
- [⚠️] UX Improvement: Integrated routing and generation pipeline (GH CLI NOT AUTHENTICATED).
