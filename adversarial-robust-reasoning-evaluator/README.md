# AD-REASON (Adversarial Robust Reasoning Evaluator)

An evaluation and stress-testing framework for LLM-based reasoning agents. It generates "Adversarial Thought-Cycles" (misleading intermediate steps, corrupted state-summaries) based on **TraderBench** logic and measures an agent's ability to "denoise" its own internal reasoning using **DenoiseFlow** and **LOGIGEN** structures.

## 🚀 Concept
AD-REASON is designed to measure the **resilience of the reasoning process itself**. It subjects an agent to "Reasoning Noise"—misleading hints or corrupted state summaries—and objectively evaluates if the agent can sense this noise and correct its path using internal denoising frameworks.

## 🧠 Key Techniques
- **Adversarial Thought-Cycles:** Using **TraderBench** logic to generate complex reasoning scenarios with intentionally injected logical traps.
- **Sensing-Regulating-Correcting (SRC):** Evaluating the target agent's ability to sense and mitigate semantic ambiguity using the **DenoiseFlow** framework.
- **LOGIGEN Ground-Truth:** Using deterministic triple-agent orchestration to verify the absolute state-truth of every reasoning node.

## 🗺️ Roadmap
- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Adversarial Thought-Cycle Model
- [ ] **Step 3:** Scene Designer (The Adversary)
- [ ] **Step 4:** Reasoning Apprentice (Agent-Under-Test)
- [ ] **Step 5:** Denoising Senser Stage
- [ ] **Step 6:** Regulator Logic
- [ ] **Step 7:** LOGIGEN Ground-Truth Checker
- [ ] **Step 8:** AD-REASON Evaluation Orchestrator
- [ ] **Step 9:** Robustness Metrics & Curves
- [ ] **Step 10:** Adversarial Scenario Export
- [ ] **Step 11:** Real-time Evaluation Dashboard
- [ ] **Step 12:** Documentation & Final PR

## 🛠️ Requirements
- `pydantic`, `asyncio`, `numpy`
- Python 3.10+

## 📄 License
MIT
