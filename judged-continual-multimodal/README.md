# Self-Evaluating Continual Multimodal Learner (SECML)

A multimodal agent that continually learns skills from visual experiences, validated by a **reasoning LLM-judge** before storage. Skills persist as hash-addressed Knowledge Objects across sessions.

## Overview

Built from three ArXiv papers:
- **XSkill** (Jiang et al.) — Dual-stream continual learning from visual trajectories without parameter updates
- **ExamTelligence** (Liu et al.) — Reasoning LLM-judges for feedback on non-verifiable tasks
- **Knowledge Objects** (Chana et al.) — Hash-addressed persistent memory tuples

## Architecture

```
Visual Observation
       ↓
TrajectoryBuffer (experience storage)
       ↓
SkillExtractor → SkillObject
       ↓
ReasoningJudge (CoT evaluation) → score + reasoning trace
       ↓ (iterate until judge satisfied)
SkillLibrary (hash-addressed, persistent)
       ↓
TriggerDetector → applicable skills ranked
       ↓
MultimodalAgent.act()
```

## Installation

```bash
pip install -r requirements.txt
```

## Components

| Component | File | Description |
|-----------|------|-------------|
| Trajectory | `src/trajectory.py` | Visual trajectory buffer |
| Skill Extractor | `src/skill_extractor.py` | Experience → skill conversion |
| Skill Object | `src/skill_object.py` | Hash-addressed skill storage |
| Judge | `src/judge.py` | Reasoning LLM-judge for skill evaluation |
| Pipeline | `src/pipeline.py` | Extract → Judge → Refine → Store |
| Trigger | `src/trigger.py` | Detect when skills are applicable |
| Agent | `src/agent.py` | Full agent integration |
| Evolution | `src/evolution.py` | Skill re-judging and composition |
| Eval | `src/eval.py` | Evaluation suite |

## Roadmap

See `memory/project-judged-continual-multimodal-plan.md` for full 12-step implementation plan.

**Status:** Step 0 complete — scaffold created.

## Testing

```bash
python -m pytest tests/ -v
```
