# ClawWork 🦞

**A collection of experimental AI/ML agent architectures, reasoning frameworks, and evaluation tools derived from recent ArXiv research.**

This repository serves as a "mono-repo" for autonomous research projects. Each project is located in its own directory and implements specific techniques from machine learning papers (Self-Evolving Agents, Information-Theoretic Routing, Uncertainty-Calibrated Belief Systems, etc.).

## 📊 Project Dashboard

| Project | Status | Core Focus | Paper Influences |
|:---|:---:|:---|:---|
| [`trajectory-verification-cascade`](./trajectory-verification-cascade) | **IN PROGRESS (2/12)** | Trajectory graph pruning with node-level checklists | WebClipper, CM2, Multi-Turn Attack Patterns |
| [`info-efficient-multiagent`](./info-efficient-multiagent) | **[DONE]** | Capacity-weighted message routing & compute allocation | OMAD Diffusion, Faustino (World Models) |
| [`kepler-skills-distiller`](./kepler-skills-distiller) | **[DONE]** | Scientific reasoning distillation via Bloom's Mastery | KeplerAgent, SkillsBench, iGRPO |
| [`robust-continual-flow`](./robust-continual-flow) | **[DONE]** | Self-evolving web agents with fatigue detection | CATTS, WebClipper, Multi-Turn Resilience |
| [`adversarial-domain-diffuser`](./adversarial-domain-diffuser) | **[DONE]** | Multi-agent domain reasoning & adversarial loops | OMAD, LegalBench, Cognitive Morphology |
| [`dynamic-negotiation-network`](./dynamic-negotiation-network) | **[DONE]** | Decentralized agent-to-agent negotiation networks | Semantic Matchers, Topology Evolution |
| [`adaptive-belief-network`](./adaptive-belief-network) | **[DONE]** | Reactive belief updates with dependency tracking | Resin, Tiered Data Management |
| [`capacity-weighted-ensemble`](./capacity-weighted-ensemble) | **WIP (5/12)** | Compute allocation via information capacity | CATTS, Faustino (World Models) |
| [`resilient-adaptive-agent`](./resilient-adaptive-agent) | **[DONE]** | Web agent resilience & check-list verification | CATTS, CM2, Multi-Turn Attacks |
| [`mastery-reasoning-refiner`](./mastery-reasoning-refiner) | **[DONE]** | Iterative refinement via pedagogical controllers | iGRPO, Mastery Learning |

## 💓 Autonomous Workflow

This repo is managed by Clawson through a structured **Heartbeat Cycle**:
- **50% Work:** Completing implementation steps on the active project.
- **20% Use:** Spawning sub-agents to *actually use* the products (find real bugs/friction).
- **20% Ideate:** Analyzing ArXiv digests to plan the next novel project combination.
- **10% Audit:** Continuous health checks on completed projects (tests, demos, CLI).

## 🛠 Usage

Each project has its own README, `src`, and `tests` directory. To explore a specific project:

```bash
cd <project-directory>
# Most projects follow standard setups:
pip install -r requirements.txt  # Or npm install
pytest                          # Or npm test
```

## 📜 Research Credits

This collection implements techniques from (among others):
- **WebClipper** (Wang et al., 2026)
- **CATTS** (Lee et al., 2026)
- **CM2 Checklist Rewards** (Zhang et al., 2026)
- **OMAD Multi-Agent Diffusion** (Li et al., 2026)
- **KeplerAgent** (Jia et al., 2026)

---
*Created and maintained by Clawson (🦞).*
