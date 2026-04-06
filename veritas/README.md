# VERITAS — Verifiable Ethical Reasoning with Integrated Test-time Activation and Skill Synthesis

**An agent reasoning architecture where every intermediate step is step-level verifiable against explicit constitutional constraints, with test-time compute adaptively routed to the steps that need it most.**

## Status

**VERITAS Step 1: Constitutional Framework and Skill Contracts - DONE ✅**

The following components are fully implemented and tested:

### ✅ Implemented Features

1. **Constitutional Principles (8 principles)**
   - `AVOID_HARM` - Prevent harmful content
   - `AVOID_HALLUCINATION` - Prevent false or fabricated information
   - `CITE_SOURCES` - Ensure proper attribution
   - `PRESERVE_PRIVACY` - Protect private information
   - `BE_CONCISE` - Encourage concise communication
   - `BE_ACCURATE` - Ensure factual accuracy
   - `BE_FAIR` - Promote fairness and balance
   - `BE_TRANSPARENT` - Encourage transparency about limitations

2. **Constitutional Contract System**
   - Schema validation with comprehensive parameter checks
   - Support for custom thresholds and compute budgets
   - Advanced scrutiny-based compute routing (alpha/beta weights)
   - Escalation models (continue, fail, escalate)

3. **Constitutional Reviewer**
   - PRM (Process Reward Model) integration with configurable models
   - Advanced critique logic with severity assessment
   - Scrutiny score calculation for compute routing
   - Compute strategy determination (efficient, standard, enhanced, maximum)

4. **Compute Router**
   - Dynamic compute allocation based on scrutiny
   - Multi-level strategies with appropriate timeouts
   - Human-in-the-loop escalation capability

5. **Supporting Modules**
   - `Logger` - Comprehensive logging system
   - `PRM` - Process Reward Model with training and evaluation
   - Full test suite with 20+ comprehensive tests

### Test Coverage

- **20+ unit tests** covering all constitutional principles
- Edge case testing (empty outputs, very short/long outputs)
- Contract validation testing
- Compute routing decision testing
- PRM integration testing
- Logger functionality testing

## Architecture

```
Task Input
    │
    ▼
┌─────────────────────────────────────────┐
│  Task Decomposer (skill chain planner) │
│  Pre-flags high-constitutional-risk     │
│  steps and pre-allocates compute budgets │
└──────────────┬──────────────────────────┘
               │ skill chain + constitutional contracts
               ▼
┌─────────────────────────────────────────┐
│  Step Executor (one step at a time)     │
│  LLM call → PRM scorer → output         │
└──────────────┬──────────────────────────┘
               │ step_output, PRM_score, constitutional_flags
               ▼
┌─────────────────────────────────────────┐
│  Constitutional Reviewer                 │
│  Checks output against step's contract  │
│  Flags violations → triggers regen       │
│  Flags low PRM → triggers verify         │
└──────┬──────────────────┬────────────────┘
       │ PASS             │ FAIL
       ▼                  ▼
┌──────────────┐   ┌─────────────────────────┐
│  Next Step    │   │  Compute Router         │
│              │   │  scrutiny = α·(1-PRM) + │
│              │   │         β·violation_score│
│              │   │  Routes to appropriate   │
│              │   │  expansion strategy:      │
│              │   │  • regeneration           │
│              │   │  • CAI critique-regen    │
│              │   │  • cross-model ensemble  │
│              │   │  • human escalation      │
│              └───────────┬──────────────┘
                       │ re-verified output
                       ▼
            ┌─────────────────────────┐
            │  Re-verify (PRM + Const) │
            │  Pass → next step       │
            │  Fail → escalate        │
            └─────────────────────────┘
```

## Quick Start

```python
from veritas import (
    VeritasAgent, 
    ConstitutionalContract, 
    ConstitutionalPrinciples,
    create_constitutional_reviewer,
    evaluate_step
)
from veritas.src.veritas.prm import PRMConfig
from veritas.src.veritas.logger import Logger

# Create a constitutional contract for a skill
contract = ConstitutionalContract(
    skill_id="fact_check_claim",
    step_name="verify_statistic",
    principles=[
        ConstitutionalPrinciples.AVOID_HALLUCINATION,
        ConstitutionalPrinciples.CITE_SOURCES,
        ConstitutionalPrinciples.PRESERVE_PRIVACY
    ],
    prm_threshold=0.75,
    constitutional_threshold=0.8,
    max_regenerations=2,
    compute_budget=5.0,  # 5 second compute budget
    scrutiny_alpha=0.6,  # Weight for PRM in scrutiny score
    scrutiny_beta=0.4,   # Weight for violations in scrutiny score
    base_compute_budget=1.0
)

# Initialize reviewer with custom PRM configuration
prm_config = PRMConfig(
    model_name="bert-base-uncased",
    learning_rate=2e-5,
    use_gpu=True
)

logger = Logger(min_level="DEBUG")
reviewer = create_constitutional_reviewer(
    critique_model="bert-base-uncased",
    critique_model_size="base",
    use_prm=True,
    logger=logger,
    prm_config=prm_config
)

# Evaluate a step output
output = "According to Smith et al. (2023), the success rate is 95%."
result = evaluate_step(output, contract, context={"task": "Verify success rate claim"})

is_pass, violations, prm_score, constitutional_score, scrutiny_score, strategy, budget = result

print(f"Pass: {is_pass}")
print(f"Violations: {len(violations)}")
print(f"PRM Score: {prm_score:.2f}")
print(f"Constitutional Score: {constitutional_score:.2f}")
print(f"Scrutiny Score: {scrutiny_score:.2f}")
print(f"Compute Strategy: {strategy}")
print(f"Compute Budget: {budget:.2f}s")
```

## Constitutional Contract Schema

```python
contract = ConstitutionalContract(
    skill_id="web_search_and_summarize",
    step_name="fetch_climate_data",
    principles=[
        ConstitutionalPrinciples.AVOID_HALLUCINATION,
        ConstitutionalPrinciples.CITE_SOURCES,
        ConstitutionalPrinciples.PRESERVE_PRIVACY
    ],
    prm_threshold=0.7,           # Minimum PRM confidence
    constitutional_threshold=0.8, # Minimum constitutional score
    max_regenerations=3,         # Max regeneration attempts
    compute_budget=10.0,         # Max compute time in seconds
    escalation_model="escalate", # "continue", "fail", or "escalate"
    scrutiny_alpha=0.6,          # PRM weight in scrutiny (0-1)
    scrutiny_beta=0.4,           # Violation severity weight in scrutiny (0-1)
    base_compute_budget=1.0      # Base compute budget for this step
)
```

## Roadmap

### ✅ Completed - Step 1: Constitutional Framework
- [x] Constitutional principles definition
- [x] Skill contract schema and validation
- [x] PRM integration
- [x] Constitutional critique logic
- [x] Compute routing based on scrutiny
- [x] Comprehensive test suite (20+ tests)
- [x] Logger and PRM modules

### Planned - Step 2: Step Verification
- [ ] Task Decomposer with constitutional risk assessment
- [ ] Skill chain execution with real-time monitoring
- [ ] Process Reward Model training pipeline
- [ ] Cross-model ensemble verification

### Planned - Step 3: Compute Routing Engine
- [ ] Dynamic compute allocation algorithms
- [ ] Multi-strategy execution (regeneration, ensemble, escalation)
- [ ] Human-in-the-loop workflow integration
- [ ] Compute budget management and tracking

### Planned - Step 4: Constitutional Skill Synthesis
- [ ] Skill gap analysis against constitutional principles
- [ ] On-the-fly skill generation with constitutional constraints
- [ ] Skill validation and integration
- [ ] Adaptive skill repertoire expansion

### Planned - Step 5: Escalation Manager
- [ ] Multi-level escalation hierarchy
- [ ] Cross-model consensus mechanisms
- [ ] Human expert escalation workflows
- [ ] Escalation analytics and reporting

### Planned - Step 6: Analytics Dashboard
- [ ] Constitutional audit logging
- [ ] PRM score analytics and visualization
- [ ] Compute usage tracking
- [ ] Violation pattern analysis

## Relationship to Other Projects

VERITAS integrates with and extends several other agent projects in this workspace:

- **ASC (Adaptive Skill Composer)** — Provides skill chains; VERITAS adds constitutional contracts to each skill
- **BDR (Belief-Driven Retriever)** — Stores reasoning traces; VERITAS uses BDR-style memory for audit trails
- **ACTR (Adaptive Calibration-Triggered Reasoning)** — PRM scoring complements ACTR's calibration mechanisms
- **FASHA (Failure-Aware Self-Healing Agent)** — Escalation system extends FASHA's failure recovery with step-level granularity
- **MCR-FV (Multi-hop Counterfactual Reasoning)** — Enables constitutional "what-if" scenario testing

## Papers

- Lightman et al., "Let's Verify Step by Step" (2023)
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (2022)
- Wang et al., "Math-Shepherd: A Labeling Procedure for Process Supervision" (2024)
- Yao et al., "Self-Rewarding Language Models" (2024)
- DeepMind, "Test-Time Compute Scaling" (2024)

## License

Apache 2.0 - See LICENSE file for details.

## Contributing

Contributions welcome! Please read the contribution guidelines first.