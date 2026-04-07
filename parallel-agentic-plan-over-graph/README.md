# Parallel Agentic Plan-over-Graph (PAPoG)

**A graph-based framework for decomposing complex goals into parallel, dependency-aware task execution with dynamic failure recovery.**

---

## Overview

PAPoG combines three recent research ideas — Plan-over-Graph scheduling, ARIES-style state observation, and Agentic Reasoning with tool use — into a single execution substrate. Given a high-level goal, PAPoG:

1. **Decomposes** it into a Directed Acyclic Graph (DAG) of interdependent tasks.
2. **Schedules** ready tasks in priority order using a pluggable policy controller.
3. **Executes** independent tasks in parallel via a thread-pool engine.
4. **Observes** every state transition and identifies critical-path nodes.
5. **Re-plans** dynamically when critical nodes fail, injecting replacement sub-graphs without disrupting in-flight work.

The result is a framework where complex, multi-step objectives are completed faster through parallelism and more reliably through automatic failure recovery.

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                  PAPoG Pipeline             │
                         └─────────────────────────────────────────────┘

  ┌──────┐    ┌───────────┐    ┌────────────┐    ┌─────────────────┐
  │ Goal │───►│ Architect │───►│ TaskGraph  │───►│PolicyController │
  └──────┘    │           │    │  (DAG)     │    │ (priority queue)│
              │ decompose │    │            │    │                 │
              │ into nodes│    │ TaskNodes  │    │ schedule ready  │
              └───────────┘    │ + edges    │    │ nodes           │
                               └────────────┘    └────────┬────────┘
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │   ExecutionEngine     │
                                              │  (ThreadPoolExecutor) │
                                              │                       │
                                              │  ┌───────┐ ┌───────┐ │
                                              │  │Worker │ │Worker │ │
                                              │  │  #1   │ │  #2   │ │
                                              │  └───────┘ └───────┘ │
                                              │  ┌───────┐ ┌───────┐ │
                                              │  │Worker │ │Worker │ │
                                              │  │  #3   │ │  #4   │ │
                                              │  └───────┘ └───────┘ │
                                              └───────────┬───────────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼                               ▼
                                ┌──────────────────┐            ┌──────────────────┐
                                │  GraphObserver   │            │ DynamicRePlanner  │
                                │                  │◄──────────►│                  │
                                │ • transitions    │  critical  │ • failure detect  │
                                │ • critical path  │  failure   │ • sub-graph gen   │
                                │ • event pub/sub  │  callback  │ • dep rewiring    │
                                └──────────────────┘            └──────────────────┘
```

**Data flow:** A goal string enters the Architect, which produces a TaskGraph (DAG of TaskNodes). The PolicyController maintains a priority queue of ready nodes and hands them to the ExecutionEngine, which dispatches work across a thread pool of ReasoningWorkers. The GraphObserver records every state transition and identifies critical-path nodes. When a critical node fails, the DynamicRePlanner generates a replacement sub-graph via the Architect, injects it into the live graph, and rewires dependencies so execution continues.

## Components

| Module | File | Role |
|---|---|---|
| **Models** | `src/models.py` | `TaskNode`, `TaskGraph`, `TaskStatus` — the core DAG data structures backed by networkx |
| **Architect** | `src/architect.py` | Decomposes goal strings into `TaskGraph` DAGs via pluggable `DecompositionStrategy` |
| **Policy Controller** | `src/policy.py` | Thread-safe priority scheduling, agent assignment, status transitions, stuck detection |
| **Worker** | `src/worker.py` | `ReasoningWorker` — executes individual tasks using pluggable `ExecutionStrategy` + tools |
| **Execution Engine** | `src/engine.py` | Parallel execution via `ThreadPoolExecutor`, dependency-aware dispatch, failure cascading |
| **Observer** | `src/observer.py` | ARIES-inspired state tracking, critical-path computation, event pub/sub system |
| **Re-Planner** | `src/replan.py` | Failure-triggered sub-graph generation, dependency rewiring, retry budgeting |
| **Tools** | `src/tools/` | `Tool` protocol + built-in tools: `SearchTool`, `PythonREPLTool`, `MemoryTool`, `ToolRegistry` |
| **Benchmark** | `src/benchmark.py` | End-to-end pipeline runner, metrics collection, timeline recording |
| **CLI** | `src/cli.py` | Command-line interface: `run`, `visualize`, `scenario`, `list-scenarios` |
| **Visualizer** | `src/visualizer.py` | DOT/PNG/SVG graph export with status-based coloring, execution timeline rendering |

## Installation

```bash
git clone https://github.com/clawson1717/ClawWork.git
cd ClawWork/parallel-agentic-plan-over-graph
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, networkx, pydantic, graphviz (Python package). For PNG/SVG rendering, the system `dot` binary is also needed (`apt install graphviz`).

## Quick Start

Run a goal through the full pipeline:

```bash
python -m src.cli run --goal "Build a distributed caching system"
```

This decomposes the goal into tasks, executes them in parallel, and prints a benchmark summary with timing, status, and timeline information.

## Usage Examples

### Run a custom goal

```bash
python -m src.cli run --goal "Design and implement a REST API" --workers 4
```

### Visualize a task graph

```bash
# Export as DOT (always works, no extra deps)
python -m src.cli visualize --goal "Build a chatbot" --output graph.dot

# Export as PNG (requires graphviz system package)
python -m src.cli visualize --goal "Build a chatbot" --output graph.png --format png

# Export as SVG
python -m src.cli visualize --goal "Build a chatbot" --output graph.svg --format svg
```

### Run a benchmark scenario

```bash
# Deep research scenario — parallel fan-out with synthesis
python -m src.cli scenario --name deep_research

# Failure recovery scenario — exercises dynamic re-planning
python -m src.cli scenario --name failure_recovery
```

### List available scenarios

```bash
python -m src.cli list-scenarios
```

### Direct benchmark CLI

```bash
# Run via the benchmark module directly
python -m src.benchmark --scenario deep_research --workers 8

# Custom goal with re-planning disabled
python -m src.benchmark --goal "Analyze market trends" --no-replan

# List scenarios
python -m src.benchmark --list
```

## Research Foundations

PAPoG draws from three recent papers:

- **Plan-over-Graph** — *Plan-over-Graph: Towards Parallelable LLM Agent Schedule* (Li et al., Feb 2025). Introduces DAG-based task decomposition for maximum parallel execution of LLM agent workflows.

- **ARIES** — *ARIES: Autonomous Reasoning on Interactive Thought Graphs* (Zhang et al., Feb 2025). Provides the state observation and critical-path monitoring model used by `GraphObserver`.

- **Agentic Reasoning** — *Agentic Reasoning: A Streamlined Framework* (Wang et al., Feb 2025). Informs the tool-using worker agent design and the reasoning-execution loop in `ReasoningWorker`.

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_engine.py -v

# Run with coverage (if pytest-cov is installed)
python -m pytest tests/ -v --cov=src
```

## Project Structure

```
parallel-agentic-plan-over-graph/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── scenarios/
│       ├── __init__.py
│       ├── deep_research.py        # Multi-step parallel research scenario
│       └── failure_recovery.py     # Failure + re-plan exercise scenario
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── models.py                   # TaskNode, TaskGraph, TaskStatus
│   ├── architect.py                # Goal → TaskGraph decomposition
│   ├── policy.py                   # Priority scheduling & orchestration
│   ├── worker.py                   # ReasoningWorker task executor
│   ├── engine.py                   # Parallel ThreadPoolExecutor engine
│   ├── observer.py                 # State tracking & event pub/sub
│   ├── replan.py                   # Dynamic failure recovery
│   ├── benchmark.py                # End-to-end pipeline runner
│   ├── benchmark_cli.py            # Benchmark CLI entry point
│   ├── cli.py                      # Main CLI (run/visualize/scenario)
│   ├── visualizer.py               # DOT/PNG/SVG graph export
│   └── tools/
│       ├── __init__.py             # Tool protocol & ToolABC base
│       ├── search_tool.py          # Simulated web search
│       ├── python_repl_tool.py     # Simulated Python execution
│       ├── memory_tool.py          # Cross-node key-value store
│       └── registry.py             # ToolRegistry & ToolRegistryProvider
└── tests/
    ├── __init__.py
    ├── test_architect.py
    ├── test_benchmark.py
    ├── test_cli.py
    ├── test_engine.py
    ├── test_models.py
    ├── test_observer.py
    ├── test_policy.py
    ├── test_replan.py
    ├── test_scaffold.py
    ├── test_tools.py
    ├── test_visualizer.py
    └── test_worker.py
```

## Status

All 12 implementation steps are complete.

- [x] **Step 1** — Project Scaffold (`models.py` — `TaskNode`, `TaskGraph`, `TaskStatus`)
- [x] **Step 2** — Task Node & Graph Model (networkx-backed DAG with topological sort)
- [x] **Step 3** — Architect Agent (`PlanArchitect` — goal decomposition via `DecompositionStrategy`)
- [x] **Step 4** — Policy Controller (priority scheduling, batch assignment, stuck detection)
- [x] **Step 5** — Worker Agent (`ReasoningWorker` — tool-using executor with `ExecutionStrategy`)
- [x] **Step 6** — Parallel Execution Engine (`ThreadPoolExecutor`, concurrent dispatch)
- [x] **Step 7** — State Observer (ARIES-inspired transitions, critical path, event pub/sub)
- [x] **Step 8** — Dynamic Re-Plan (failure recovery, sub-graph injection, dependency rewiring)
- [x] **Step 9** — Tool Integration (`Tool` protocol, `SearchTool`, `PythonREPLTool`, `MemoryTool`, `ToolRegistry`)
- [x] **Step 10** — Benchmark (`BenchmarkRunner`, deep research scenario, failure scenario, CLI)
- [x] **Step 11** — CLI & Visualization (`cli.py`, `visualizer.py`, DOT/PNG/SVG export)
- [x] **Step 12** — Documentation & README Update

## License

Part of the [ClawWork](https://github.com/clawson1717/ClawWork) project collection.
