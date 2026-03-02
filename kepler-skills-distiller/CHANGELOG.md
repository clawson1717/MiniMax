# Changelog

All notable changes to Kepler-Skills-Distiller are documented in this file.

## [1.0.0] - 2026-02-26

### Added

#### Core Components
- **Physical Property Inferencer**: Infers dimensional consistency, symmetries, and conservation laws from raw data
- **Kepler Reasoning Agent**: Physics-guided discovery agent with LLM integration
- **Expert Skill Generator**: Converts discovery trajectories into SkillsBench-format skills
- **Skill Schema**: Structured JSON schema for procedural knowledge

#### Training & Distillation
- **Pedagogical Data Synthesizer**: Creates training data organized by Bloom's Taxonomy tiers
- **Apprentice Model Scaffold**: Configurable small model setup (Qwen-1.5B, Llama-3-1B, etc.)
- **Mastery Distillation Loop**: Progressive curriculum training with 85% mastery threshold
- **Trainer Utilities**: Training helpers and checkpoint management

#### Benchmarking
- **Equation Discovery Benchmark**: Feynman Symbolic Regression Benchmark with 13 equations
- **Performance Metrics**: MAE, RMSE, exact match, improvement ratio
- **Category Filtering**: Filter by physics/math, complexity level

#### CLI
- **discover**: Run equation discovery pipeline
- **distill**: Run knowledge distillation with Bloom's curriculum
- **benchmark**: Run Feynman benchmark suite
- **visualize**: ASCII visualizations for capacity, curriculum, skills
- **list-skills**: List generated expert skills

#### Testing
- 103 comprehensive tests across all modules
- Mock model support for testing without LLM inference

### Papers Implemented
- **KeplerAgent** (Yang et al., 2026): Physics-guided reasoning loop
- **SkillsBench** (Li et al., 2026): Expert skill structure
- **Pedagogically-Inspired Data Synthesis** (He et al., 2026): Bloom's curriculum

### Project Completion
All 11 planned steps completed:
1. ✅ Project Scaffold
2. ✅ Physical Property Inferencer
3. ✅ Skill Schema Definition
4. ✅ Kepler Reasoning Agent
5. ✅ Expert Skill Generator
6. ✅ Pedagogical Data Synthesizer
7. ✅ Apprentice Model Scaffold
8. ✅ Mastery Distillation Loop
9. ✅ Equation Discovery Benchmark
10. ✅ CLI & Visualization
11. ✅ Final Documentation & README