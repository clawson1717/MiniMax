# MiRA-KO: Persistent Milestone Memory for Web Navigation Agents

> *"What if your web navigation agent remembered every sub-goal it ever learned — and reused it?"*

**MiRA-KO** combines **MiRA's** subgoal-driven RL framework with **Knowledge Objects** — a persistent, hash-addressed milestone memory — so agents can retrieve, adapt, and reuse learned subgoals across sessions. No more learning "select departure airport" from scratch on every new airline website.

---

## The Problem

Web navigation agents (like those benchmarked on [WebArena-Lite](https://github.com/webnema/arena-eval)) currently learn every task from scratch. When MiRA achieves 43% success on a flight-booking task, that knowledge evaporates when the session ends. The next flight-booking task — on a different website — starts from 6% again.

This is unnecessary. The subgoals are the same across airline websites: "select departure city," "select date," "enter passenger info." The only differences are surface-level element selectors. Current agents can't exploit this structure.

---

## The Solution

MiRA-KO is a milestone memory system built on top of a MiRA-style web navigation agent:

```
┌──────────────────────────────────────────────────────────────┐
│  NEW TASK: "Book a flight Denver→Seattle on United.com"      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  MILESTONE RETRIEVAL ENGINE                                  │
│  Encode(task_context) → [MilestoneKO₁, MilestoneKO₂, ...]   │
│  Indexed by: website_type, subgoal_name, embedding similarity│
└──────────────────────────┬───────────────────────────────────┘
                           │ retrieved top-k milestones
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  BOOTSTRAP PLANNER                                           │
│  For each retrieved milestone:                               │
│    - Copy the action sequence as a suggested starting point  │
│    - Adapt element selectors to the new page context         │
│    - Fall back to MiRA decomposition if no relevant milestones│
└──────────────────────────┬───────────────────────────────────┘
                           │ adapted subgoals
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  WEB NAVIGATION AGENT (MiRA-style)                           │
│  Executes adapted subgoals, earns milestone-based rewards    │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌────────────────────────┐  ┌──────────────────────────────────┐
│  SUCCESS: Store as     │  │  FAILURE: Store as FailureKO     │
│  MilestoneKO in        │  │  with root-cause inference +     │
│  persistent memory     │  │  hindsight link to success KO    │
└────────────────────────┘  └──────────────────────────────────┘
```

The agent writes every milestone to persistent memory. Future tasks read from it first.

---

## Key Innovation

**Why can't you just use MiRA alone?**

MiRA generates milestones at runtime and loses them after the session ends. It cannot retrieve or reuse past milestones. Each new task starts from the same 6% baseline.

**Why can't you just use Knowledge Objects alone?**

Knowledge Objects are a general persistent memory primitive — they solve capacity limits and compaction loss. But they don't know *how* to index subgoal-relevant experiences or *when* to retrieve them for a new task. The retrieval schema and the bootstrap planning loop are MiRA-KO's contribution.

**Together, MiRA-KO uniquely enables:** Cross-session subgoal reuse with semantic adaptation, so agents improve over time rather than resetting every session.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & docker-compose
- OpenRouter API key (or local LLM endpoint)
- 16GB RAM minimum (for WebArena-Lite headless browser)

### Installation

```bash
git clone https://github.com/your-org/mira-ko.git
cd mira-ko
pip install -e .
```

### Run a Demo

```bash
# Start the milestone memory store and WebArena-Lite environment
docker-compose up -d

# Run a single navigation task with warm memory
python -m mira_ko.scripts.demo \
    --task "Book a flight from Denver to Seattle on United.com" \
    --warm-memory

# Run the full benchmark
python -m mira_ko.scripts.evaluate --benchmark webarena-lite --episodes 100
```

### Warm Memory vs. Cold Memory

```python
from mira_ko import MilestoneStore, BootstrapPlanner, WebNavAgent

store = MilestoneStore("./milestone_ko_db")

# COLD: empty store — pure MiRA-style learning from scratch
agent_cold = WebNavAgent(store=store, warm=False)
# → Starts at ~6.4% success rate on WebArena-Lite

# WARM: pre-populated store with milestone KOs
agent_warm = WebNavAgent(store=store, warm=True, top_k=5)
# → Retrieves relevant past milestones before planning
# → Expected: faster convergence, higher peak success rate
```

---

## Architecture

### Components

| Component | Description |
|---|---|
| `MilestoneKO` | Hash-addressed persistent tuple storing a milestone trajectory |
| `FailureKO` | Hash-addressed tuple for failed milestones, with root-cause inference |
| `MilestoneStore` | SQLite-backed key-value store for MilestoneKOs and FailureKOs |
| `RetrievalEngine` | Embedding + keyword index for retrieving relevant milestones |
| `BootstrapPlanner` | MiRA-style subgoal planner initialized with retrieved milestones |
| `MilestoneAdapter` | Semantic element matching to adapt retrieved actions to new pages |
| `HindsightLinker` | Links failed KOs to successful KOs via heuristic hindsight inference |
| `WebNavAgent` | Full agent loop: retrieve → bootstrap → execute → store |

### Data Flow

```
Task Input
    │
    ▼
RetrievalEngine (embed + index search)
    │ returns top-k MilestoneKOs
    ▼
BootstrapPlanner (adapt action sequences)
    │ returns adapted subgoals
    ▼
WebNavAgent.execute() → trajectory
    │
    ├── success → MilestoneStore.save(success_ko)
    │
    └── failure → HindsightLinker → MilestoneStore.save(failure_ko)
```

---

## Roadmap

- [x] Milestone Knowledge Object schema definition
- [x] MilestoneStore with SQLite backend and secondary indexes
- [ ] Retrieval engine (embedding-based + keyword fast path)
- [ ] Bootstrap planner with adaptation logic
- [ ] Hindsight failure memory (FailureKO + root-cause inference)
- [ ] WebArena-Lite integration
- [ ] Reproduce MiRA baseline (Gemma3-12B, ~43% success)
- [ ] MiRA-KO (warm) vs. MiRA (cold) comparative benchmark
- [ ] Ablation study (retrieval, adaptation, failure memory)
- [ ] Docker packaging + evaluation scripts
- [ ] Pre-populated milestone store (seeded with common web navigation subgoals)

---

## Benchmarking

### Expected Results (WebArena-Lite)

| Configuration | Cold Start Success | Warm Start Success (5 sessions) |
|---|---|---|
| MiRA baseline (no KO memory) | 6.4% | ~43% |
| MiRA-KO (no retrieval, pure RL) | 6.4% | ~43% |
| MiRA-KO (retrieval, no adaptation) | ~10% | ~50% |
| MiRA-KO (retrieval + adaptation) | ~15% | ~60% |
| MiRA-KO (full: retrieval + adaptation + failure memory) | ~15% | ~65% |

*Estimates based on the hypothesis that milestone reuse reduces subgoal exploration cost.*

### What Each Ablation Tests
- **Retrieval**: Does knowing about past milestones help, even if you can't use them directly?
- **Adaptation**: Is semantic element matching the active ingredient, or is pure retrieval enough?
- **Failure memory**: Do failure KOs prevent repeated mistakes, or are successes alone sufficient?

---

## Papers This Builds On

| Paper | Contribution Used |
|---|---|
| [MiRA: Milestoning your RL Enhanced Agent](https://arxiv.org/abs/XXXX) | Subgoal-driven RL framework, milestone reward signals, WebArena-Lite benchmark |
| [Facts as First Class Objects: Knowledge Objects](https://arxiv.org/abs/XXXX) | Hash-addressed persistent tuples, capacity-limit elimination |
| [HeRL: Hindsight Experience Guided RL for LLMs](https://arxiv.org/abs/XXXX) | Hindsight failure reuse — failed trajectories as learning signal |
| [RPMS: Rule-Augmented Memory Synergy](https://arxiv.org/abs/XXXX) | Episodic memory filtering for embodied planning — inspiration for retrieval indexing |

---

## Citation

```bibtex
@article{mira-ko,
  title={MiRA-KO: Persistent Milestone Memory for Web Navigation Agents},
  author={},
  year={2026}
}
```

---

## License

MIT
