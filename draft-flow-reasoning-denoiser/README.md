# DRAFT-FLOW (Uncertainty-Aware Concise Reasoning Denoiser)

A high-efficiency reasoning engine that uses **Draft-Thinking** concise tokens for initial exploration and applies **DenoiseFlow**'s SRC framework to identifies and "denoise" reasoning errors incompressed semantic space. It uses **Dynamic Interaction Graphs (DIG)** to trace precisely which draft-token introduced semantic ambiguity.

## 🚀 Concept
DRAFT-FLOW explores reasoning paths using ultra-cheap "Draft" tokens (concise symbols). If the **Senser** detects uncertainty in a draft, the **Regulator** uses a **DIG** to trace the causal origin of that noise back to a single token. The **Healer** then re-drafts ONLY that segment, yielding a high-accuracy result at a fraction of the token cost of standard Chain-of-Thought (CoT).

## 🧠 Key Techniques
- **Draft-Thinking Integration:** Using concise, curriculum-grounded reasoning structures for efficient exploration.
- **Uncertainty-Aware Denoising:** Applying the Sensing-Regulating-Correcting (SRC) framework to identify and resolve reasoning errors.
- **Causal Path Tracking:** Using Dynamic Interaction Graphs (DIG) to pinpoint the root cause of semantic drift in compressed token space.

## 🗺️ Roadmap
- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Concise Draft Tokenizer
- [ ] **Step 3:** Draft-Generation Agent
- [ ] **Step 4:** Uncertainty Senser Module
- [ ] **Step 5:** Draft Interaction Graph
- [ ] **Step 6:** SRC Flow Regulator
- [ ] **Step 7:** Targeted Draft Correction
- [ ] **Step 8:** DRAFT-FLOW Orchestrator
- [ ] **Step 9:** Logic-Grounded Synthesis
- [ ] **Step 10:** Efficiency vs. Accuracy Benchmark
- [ ] **Step 11:** Real-time Draft-DIG Dashboard
- [ ] **Step 12:** Documentation & Final PR

## 🛠️ Requirements
- `torch`, `networkx`, `pydantic`, `asyncio`
- Python 3.10+

## 📄 License
MIT
