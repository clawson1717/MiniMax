# CAD-TRACE (Causal Adversarial Drift-Tracker & Corrective Evaluator)

A reasoning monitor that tracks "Causal Drift" in multi-agent collaboration. It maps every reasoning node to a **Dynamic Interaction Graph (DIG)** and applies an adversarial **TraderBench**-inspired evaluator to sense exactly when a reasoning branch drifts from the original groundwork. It then triggers **DenoiseFlow** corrections for just the drifting branches.

## 🚀 Concept
CAD-TRACE is designed to solve the "Hallucination Spiral" in complex multi-agent chains. By measuring "Causal distance" at every reasoning node using **DIG to Heal**, we can pinpoint exactly when intent was lost. We then use an adversarial senser (inspired by **TraderBench**) to verify the drift and a **DenoiseFlow** regulator to "heal" just that specific branch of the logic.

## 🧠 Key Techniques
- **Dynamic Interaction Graphs (DIG):** Causal path tracing for multi-agent reasoning nodes to identify collaboration explanation.
- **Adversarial Robustness Evaluation:** Using **TraderBench** logic to generate and inject reasoning noise for stress-testing.
- **Sensing-Regulating-Correcting (SRC):** Uncertainty-aware denoising to pinpoint and correct "Broken" branches in the causal graph.

## 🗺️ Roadmap
- [x] **Step 1:** Project Scaffold
- [x] **Step 2:** Causal Interaction Payload
- [x] **Step 3:** Drift-Sensing Adversary
- [x] **Step 4:** Live Causal DIG Tracker
- [x] **Step 5:** Semantic Drift Calculator
- [x] **Step 6:** Uncertainty Flow Senser
- [x] **Step 7:** Regulator of Truth-Resilience
- [x] **Step 8:** Corrective Logic Healer
- [x] **Step 9:** CAD-TRACE Monitor Agent
- [x] **Step 10:** Adversarial Drift Benchmark
- [ ] **Step 11:** Plotly Drift Visualizer
- [ ] **Step 12:** Documentation & Final PR

## 🛠️ Requirements
- `networkx`, `pydantic`, `asyncio`, `plotly`
- Python 3.10+

## 📄 License
MIT
