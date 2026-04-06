# Self-Healing RAG

> A retrieval system that diagnoses WHY your RAG fails and actively repairs its knowledge base.

## Problem Solved

Standard RAG systems fail silently — they retrieve wrong or insufficient context and generate plausible-sounding but incorrect answers with no diagnostic signal. You never know whether retrieval failed due to a query mismatch, a reasoning error, or a genuine gap in the knowledge base. Debugging is guesswork.

## Core Innovation

Self-Healing RAG combines **WriteBack-RAG** (trainable knowledge base with write-back enrichment) with **FASHA-style failure diagnosis** to create a closed-loop system that: (1) detects retrieval failures via cycle-consistent consistency checking, (2) classifies failure modes (context loss, reasoning error, knowledge gap, semantic drift), and (3) actively writes corrected evidence back into the trainable KB — so future queries don't fail the same way. R-C2's label-free cycle-consistency reward and MiRA's milestone-driven fine-tuning guide the KB updates automatically.

## Architecture Summary

```
Query → [Failure Detector] → [Failure Mode Classifier]
                               ↓
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
   Context Loss         Reasoning Error      Knowledge Gap
          ↓                    ↓                    ↓
   Query Rewrite        Self-Reflect         Write-Back Enrich
   + Re-Retrieve        + Context Expand      + KB Fine-Tune
          └────────────────────┼────────────────────┘
                               ↓
                    [Trainable Knowledge Base]
                               ↓
                         Final Answer
```

**Components:**
1. **Cycle-Consistency Failure Detector** — Label-free scorer flags inconsistent query→context→answer triplets
2. **Failure Mode Classifier** — Categorizes failures to route to correct recovery strategy
3. **Recovery Strategy Selector (FASHA-style)** — Maps failure modes to concrete recovery actions
4. **Trainable Knowledge Base** — Differentiable two-tower retriever fine-tuned via consistency rewards (WriteBack-RAG)
5. **Write-Back Enrichment Pipeline** — Extracts validated evidence and updates KB params + embeddings

## Quick Start

```bash
# Install
pip install self-healing-rag

# Initialize with your corpus
python -m self_healing_rag index --corpus ./docs

# Query with automatic self-healing
python -m self_healing_rag query "What is the capital of France?"
# → Detects failure, diagnoses, heals KB, returns answer

# Run diagnostic on existing RAG pipeline
python -m self_healing_rag diagnose --query "..." --context "..." --answer "..."
```

## Roadmap

- [ ] **Step 1** — Cycle-Consistency Failure Detector
- [ ] **Step 2** — Failure Mode Classifier
- [ ] **Step 3** — Evidence Distillation Module
- [ ] **Step 4** — Trainable Knowledge Base Backend
- [ ] **Step 5** — Write-Back Enrichment Pipeline
- [ ] **Step 6** — Recovery Strategy Selector (FASHA-style)
- [ ] **Step 7** — Closed-Loop Evaluation Harness
- [ ] **Step 8** — Cycle-Consistent Reward Signal
- [ ] **Step 9** — Milestone-Driven Fine-Tuning (MiRA-style)
- [ ] **Step 10** — API Server & SDK
- [ ] **Step 11** — Dashboard & Observability
- [ ] **Step 12** — Production Hardening & Benchmarking
