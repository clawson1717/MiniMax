# DRIFT-SENSE (Dynamic Real-time Interaction Frequency Tracker & Sensory Evaluator)

A multi-agent monitoring system that uses **Dynamic Interaction Graphs (DIG)** to monitor and explain agent collaboration in real-time. It applies **DenoiseFlow** sensing to every node in the graph to identify and correct "Semantic Drift"—the gradual loss of original intent across multiple agent turns.

## 🚀 Concept
DRIFT-SENSE provides an explainable, graph-based defense against "hallucination spirals" in complex multi-agent reasoning chains. By measuring the "semantic distance" from the original user query at every interaction node, it can pinpoint exactly where intent was lost and proactively prune or "heal" drifting reasoning branches.

## 🧠 Key Techniques
- **Dynamic Interaction Graph (DIG):** Using path-based causal tracing to monitor interaction frequency and intent flow.
- **Sensing-Regulating-Correcting (SRC):** Applying DenoiseFlow's uncertainty-aware denoising to multi-agent communication streams.
- **Trajectory Verification:** Logic-grounded verification inspired by the Trajectory Verification Cascade (TVC) for absolute state-truth checking.

## 🗺️ Roadmap
- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Interaction Payload Schema
- [ ] **Step 3:** Real-Time Interaction Tracker
- [ ] **Step 4:** Semantic Drift Senser
- [ ] **Step 5:** SRC Regulator Engine
- [ ] **Step 6:** Targeted Path Correction
- [ ] **Step 7:** Multi-Agent Cascade Verifier
- [ ] **Step 8:** DRIFT-SENSE Monitoring Agent
- [ ] **Step 9:** Interaction Frequency Benchmark
- [ ] **Step 10:** Real-time Plotly Dashboard
- [ ] **Step 11:** CLI Monitoring Tool
- [ ] **Step 12:** Documentation & Final PR

## 🛠️ Requirements
- `networkx`, `pydantic`, `asyncio`, `plotly`
- Python 3.10+

## 📄 License
MIT
