# Beam-Draft RL (BDRL)

A parameter-efficient reinforcement learning (PEFT-RL) framework that trains compact LLMs to internalize complex physics-based reasoning through a "Draft-Thinking" curriculum.

## 🚀 Concept
By using verifiable rewards (from physics engines or symbolic solvers), the model learns to generate concise, mathematically sound reasoning "drafts" that avoid procedural templates and achieve deep internalization of governing equations.

## 🧬 Papers Combined
1. **BeamPERL** (Buehler et al.) — PEFT-RL with verifiable rewards for structured mechanics reasoning.
2. **Draft-Thinking** (Cao et al.) — Curriculum learning for concise "draft-style" reasoning structures.
3. **DenoiseFlow** (Yan et al.) — Sensing-Regulating-Correcting framework for uncertainty-aware denoising in agentic workflows.

## ✨ Novel Contribution
Combines the precision of verifiable physics rewards (BeamPERL) with the efficiency of compressed reasoning tokens (Draft-Thinking), while using DenoiseFlow's uncertainty-aware sensing to identify where a model's "draft" deviates from physical reality, enabling targeted correction during RL training.

## 📅 Roadmap (12 steps)

1. **Project Scaffold** [DONE]
   - Setup directory structure, requirements, and package initialization.
2. **Physics Environment & Reward Engine** [DONE]
   - Implement `PhysicsEngine` and `VerifiableReward` for mechanics/equilibrium.
3. **Draft-Thinking Tokenizer & Wrapper** [DONE]
   - Logic to truncate/compress CoT into "draft-style" with special tokens.
4. **Uncertainty Sensor** [DONE]
   - DenoiseFlow-style sensing using entropy and semantic consistency.
5. **PEFT-RL Trainer (LoRA/PPO)** [DONE]
   - Core RL training loop with LoRA and PPO.
6. **Step-wise Drafting Curriculum** [DONE]
   - Curriculum stages from Full CoT to Pure Draft based on mastery.
7. **Regulator & Corrector** [DONE]
   - Sensing-Regulating-Correcting loop for fine-grained RL feedback.
8. **Dataset Preparation (Beam Mechanics)** [DONE]
   - Generate problem/solution pairs (forces, moments, stress).
9. **Benchmark Suite** [DONE]
   - Accuracy vs. Efficiency metrics and comparison to baseline RL.
10. **CLI Interface** [DONE]
    - Commands: `train`, `evaluate`, `visualize-drafts`, `solve`.
11. **Integration Tests** [DONE]
    - End-to-end verification (Problem → Draft → Verifiable Success).
12. **Documentation & Demo** [DONE]
    - Final README and "Drafting vs Traditional CoT" token savings demo.

## 🛠 Usage

To use Beam-Draft RL (BDRL):

### 1. Run the Demo
Compare "Drafting" vs "Traditional CoT" token savings and performance.
```bash
python3 src/demo.py
```

### 2. Run the Benchmark
Evaluate accuracy and efficiency metrics programmatically.
```bash
python3 -m src.benchmark
```

### 3. Solve a Problem (CLI)
Use the CLI to solve a mechanics problem using a trained draft-thinking model.
```bash
python3 -m src.cli solve "cantilever beam 5m with 10kN at free end"
```

---
Part of the **ClawWork** series. 🦞
