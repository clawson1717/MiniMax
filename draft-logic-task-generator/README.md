# DRAFT-LOGIC (Efficient Curriculum-Grounded Task Generator)

An ultra-efficient task generation system that uses **Draft-Thinking** to synthesize complex, state-verified agentic scenarios. It applies **LOGIGEN**'s triple-agent orchestration to build "Adversarial Training Sets" (inspired by **TraderBench**) for testing agent reasoning stability while minimizing token costs.

## 🚀 Concept
DRAFT-LOGIC is designed to solve the "expensive dataset" problem. Generating tens of thousands of causally valid agentic tasks typically burns millions of tokens in Chain-of-Thought (CoT). By using **Draft-Thinking** as the communication medium between generative agents, we can synthesize ultra-complex, state-verified tasks for a fraction of the cost.

## 🧠 Key Techniques
- **Draft-Thinking Integration:** Using curriculum-grounded concise reasoning to reduce token usage between Architect and Designer agents.
- **LOGIGEN Generation:** Triple-agent orchestration (Architect, Set Designer, Explorer) for deterministic, verifiable task synthesis.
- **Adversarial Buffering:** Injecting **TraderBench**-inspired adversarial noise into valid tasks to create robust training data.

## 🗺️ Roadmap
- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Concise Draft Schema
- [ ] **Step 3:** Architect Agent (Draft-Enabled)
- [ ] **Step 4:** Set Designer (World Builder)
- [ ] **Step 5:** Explorer Agent (Verification)
- [ ] **Step 6:** Adversarial Noise Injector
- [ ] **Step 7:** Curriculum Manager
- [ ] **Step 8:** Multi-Agent Task Pipeline
- [ ] **Step 9:** DRAFT-LOGIC Task Exporter
- [ ] **Step 10:** Task Diversity Benchmark
- [ ] **Step 11:** Generation Dashboard
- [ ] **Step 12:** Documentation & Final PR

## 🛠️ Requirements
- `pydantic`, `asyncio`, `jsonlines`
- Python 3.10+

## 📄 License
MIT
