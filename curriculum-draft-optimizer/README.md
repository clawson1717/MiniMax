# CURRICULUM-DRAFT (Long-CoT Learning Optimizer)

A training and optimization framework designed to teach agents efficient reasoning using strictly concise "Draft-style" tokens. It uses **LOGIGEN** for verifiable task generation and organizes training into 6 tiers of **Bloom's Taxonomy** (inspired by **Kepler Skills Distiller**) to ensure mastery across all levels of reasoning complexity.

## 🚀 Concept
CURRICULUM-DRAFT is about training agents to "get the point across" as efficiently as possible. By organizing **Draft-Thinking** learning into a Bloom-level curriculum, we can systematically teach a small model (apprentice) to handle complex, logic-verified tasks using a fraction of the token budget of standard Chain-of-Thought (CoT).

## 🧠 Key Techniques
- **Draft-Thinking Optimization:** Learning to internalize concise "drafting" structures to reduce Long-CoT token costs.
- **LOGIGEN Task Generation:** Synthesis of causally valid training scenarios to provide a ground-truth for drafting logic.
- **Bloom's Taxonomy Mastery:** Organizing training into 6 hierarchical tiers (from Remember to Create) to bridge the gap from simple retrieval to complex synthesis.

## 🗺️ Roadmap
- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Draft-Token Specification
- [ ] **Step 3:** LOGIGEN Curriculum Generator
- [ ] **Step 4:** Apprentice Draft Model
- [ ] **Step 5:** Master Draft Oracle
- [ ] **Step 6:** Remember/Understand: Initial Mastery
- [ ] **Step 7:** Apply/Analyze: Intermediate Logic
- [ ] **Step 8:** Evaluate/Create: Advanced Synthesis
- [ ] **Step 9:** Drafting Efficiency Benchmark
- [ ] **Step 10:** Training & Loss Statistics
- [ ] **Step 11:** Real-time Training Dashboard
- [ ] **Step 12:** Final Documentation & PR

## 🛠️ Requirements
- `torch`, `transformers`, `asyncio`, `pydantic`
- Python 3.10+

## 📄 License
MIT
