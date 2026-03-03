# Adversarial Denoising Draft-Agent (ADDA)

## Concept
A robust, efficient reasoning system for non-stationary environments (like finance) that combines concise "draft-style" reasoning with uncertainty-aware denoising to combat accumulated semantic ambiguity and environmental noise (market manipulation).

## Source Papers

1. **DenoiseFlow** (Yan et al., March 2026)
   - Technique: Noisy MDP formulation for long-horizon workflows; Sensing-Regulating-Correcting denoising stages.
   - Use: Manage "accumulated semantic ambiguity" by routing between fast draft-reasoning and parallel exploration based on uncertainty.

2. **TraderBench** (Yuan et al., March 2026)
   - Technique: Adversarial market-manipulation transforms in stochastic simulations; realized performance metrics (Sharpe ratio, drawdown).
   - Use: The test environment and "noise" source; performance-grounded feedback loop.

3. **Draft-Thinking** (Cao et al., March 2026)
   - Technique: Concise "draft-style" reasoning structures; progressive curriculum learning.
   - Use: Core reasoning strategy to maintain efficiency and avoid "overthinking" until the environment becomes too noisy.

## Innovation: The Denoising Draft
The **ADDA** project introduces a **Denoising Draft** paradigm:
1. **Efficiency by Design**: The agent primarily uses concise "Draft-Thinking" to generate plans/trades, avoiding expensive long-CoT.
2. **Environmentally-Triggered Denoising**: Using DenoiseFlow's "Regulating" stage, the agent senses when environmental "market noise" (from TraderBench transforms) creates too much semantic uncertainty in its draft.
3. **Adaptive Branching**: Only when uncertainty spikes does the agent branch into expensive parallel simulations to "denoise" its reasoning trajectory.
4. **Realized-Performance Feedback**: Instead of using an LLM-judge, the "Correcting" stage uses realized P&L and risk metrics (Sharpe) to heal failing reasoning paths.

## 12-Step Implementation Plan

### Step 1: Project Scaffold [ ]
- Create `ClawWork/adversarial-denoising-draft-agent/` directory
- `src/`, `tests/`, `data/` structure
- `requirements.txt` with torch, numpy, pandas, gym/gymnasium
- Initial `src/__init__.py`

### Step 2: Adversarial Market Environment [ ]
- `src/env.py` with `AdversarialMarketEnv` (Noisy MDP)
- Implement stochastic price generation and "Adversarial Transforms" (Market Manipulation: spoofing, wash trading)
- Metrics: realized P&L, Sharpe ratio, Drawdown

### Step 3: Draft-Reasoning Core [ ]
- `src/reasoner.py` with `DraftReasoner` class
- Produces concise, step-based "Draft Plans" for trading decisions
- Avoids long-CoT/overthinking by default

### Step 4: Semantic Uncertainty Senser [ ]
- `src/sensing.py` with `UncertaintySenser`
- Estimates per-step semantic uncertainty for draft reasoning
- Uses entropy/variance across multiple draft heads (lite)

### Step 5: Denoising Regulator [ ]
- `src/regulating.py` with `DenoiseRegulator`
- Threshold-based routing: Draft vs. Parallel Exploration
- Adapts to environmental noise intensity

### Step 6: Parallel Exploration Engine [ ]
- `src/exploration.py` with `ParallelExplorer`
- Executes multiple alternative reasoning paths for high-uncertainty nodes
- Aggregates "denoised" consensus for the next step

### Step 7: Realized Performance Verifier [ ]
- `src/verifier.py` with `PerformanceVerifier`
- Uses environment feedback (P&L, Sharpe) as the primary verification signal
- Maps realized metrics to reasoning step validity

### Step 8: Correcting & Healing Module [ ]
- `src/correcting.py` with `TrajectoryHealer`
- Influence-based root-cause localization (which step caused the drawdown?)
- Targeted recovery/backtracking to last stable state

### Step 9: ADDA Agent Integration [ ]
- `src/agent.py` with `ADDAAgent` class
- Combines sensing, regulating, draft-reasoning, and correcting
- Unified interface for task execution

### Step 10: Curriculum Learning Layer [ ]
- `src/curriculum.py` with `ProgressiveTrainer`
- Logic to internalize efficient "draft" patterns under increasing adversarial noise
- "Draft-Thinking" style curriculum

### Step 11: Trading Benchmarks [ ]
- `src/benchmark.py` with `ADDACompetenceSuite`
- Compare ADDA against pure Draft and pure Long-CoT agents
- Evaluate: Accuracy, Token Efficiency, Resilience to Manipulation

### Step 12: Documentation & CLI [ ]
- Comprehensive README with architecture and Mermaid diagrams
- CLI for running simulations and benchmarks
- Tutorial on "Denoising Drafts"

---

## Getting Started
To begin work on Step 1, run the `github-project-selector` skill during the next heartbeat cycle.

## License
MIT
