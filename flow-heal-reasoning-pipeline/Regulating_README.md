# ReasoningRegulator Module

## Overview

The `ReasoningRegulator` is the "R" in the SRC (Sensing-Regulating-Correcting) framework. It analyzes reasoning steps to identify root causes of semantic drift and uncertainty, then generates targeted recommendations for the healing agent.

This module is a critical component of the FLOW-HEAL pipeline, working between the NoiseSenser (sensing) and the HealAgent (correcting) to provide intelligent regulation of the reasoning process.

## Key Features

### 1. Semantic Drift Detection
- Uses embedding distance and textual similarity metrics
- Identifies when the reasoning has deviated from the expected path
- Quantifies drift confidence and provides indicators

### 2. Uncertainty Quantification
- Analyzes uncertainty scores from language models
- Detects hedging language and coherence issues
- Provides normalized uncertainty metrics

### 3. Root Cause Analysis
- Leverages the Dynamic Interaction Graph (DIG) to trace causal relationships
- Identifies which steps are likely causing problems
- Ranks root causes by composite score (quality, uncertainty, importance)

### 4. Targeted Healing Recommendations
- Generates specific actions for the healing agent
- Provides confidence scores and priority levels
- Supports multiple recommendation types:
  - `re_execute_step`: Complete re-execution of low-quality steps
  - `clarify_step`: Request clarification for high-uncertainty steps
  - `review_step`: Careful review of critical nodes
  - `targeted_healing`: Focused healing for moderate issues
  - `validate_current_step`: Validate before proceeding
  - `general_review`: Broad review when no specific cause

### 5. Quality Metrics Calculation
- **session_quality**: Overall quality score (0-1, higher is better)
- **drift_prevalence**: Proportion of recent steps with drift (0-1)
- **uncertainty_level**: Average uncertainty score (0-1)
- **root_cause_severity**: Severity based on root cause quality (0-1)
- **healing_urgency**: Combined urgency score (0-1, higher = more healing needed)

### 6. Visualization Capabilities
- **Regulation Reports**: Text, markdown, or JSON format summaries
- **DIG Visualization**: Graphviz DOT representation with:
  - Node color based on quality (green=good, red=poor)
  - Root causes highlighted in red/diamond shape
  - Current step highlighted in blue/box shape
  - Edge colors based on causal confidence

## Architecture

```
FLOW-HEAL Pipeline
├── Sensing Module (NoiseSenser) → detects drift/uncertainty
├── Regulating Module (ReasoningRegulator) → identifies root causes
└── Correcting Module (HealAgent) → performs healing
```

## Integration

The ReasoningRegulator seamlessly integrates with other FLOW-HEAL components:

### With Payload Module (`payload.py`)
- Uses `ReasoningStep` and `ReasoningSession` data structures
- Extends payload with analysis results and node metadata

### With Sensing Module (`sensing.py`)
- Uses `NoiseSenser` for semantic analysis
- Integrates embedding-based similarity metrics
- Leverages uncertainty quantification methods

### With Graph Module (`graph.py`)
- Uses `ReasoningDIG` for causal tracking
- Integrates root cause analysis via DIG traversal
- Updates node metadata with quality scores

## Usage

### Basic Usage

```python
from flow_heal.src.regulating import create_reasoning_regulator
from flow_heal.src.payload import ReasoningStep, ReasoningSession
from flow_heal.src.graph import ReasoningDIG

# Create regulator instance
regulator = create_reasoning_regulator()

# Initialize a session
regulator.initialize_session(
    session_id="my_session_001",
    conversation_id="my_conversation_001"
)

# Create a reasoning step
step = ReasoningStep(
    step_id="step_001",
    agent_id="my_agent",
    agent_role="thinker",
    intent_hash="initial_thoughts",
    content="I will think about this problem carefully.",
    uncertainty_score=0.2,
    coherence_score=0.9,
    logic_score=0.8
)

# Analyze and regulate
result = regulator.analyze_and_regulate(step)

# Access regulation results
print(f"Session quality: {result.quality_metrics['session_quality']:.2f}")
print(f"Healing urgency: {result.quality_metrics['healing_urgency']:.2f}")

# Get recommendations
for rec in result.healing_recommendations:
    print(f"Recommendation: {rec['action']} with confidence {rec['confidence']:.2f}")
```

### Advanced Usage with Custom Configuration

```python
from flow_heal.src.regulating import ReasoningRegulator, RegulatorConfig

# Create custom configuration
config = RegulatorConfig(
    drift_threshold=0.7,        # More sensitive to drift
    uncertainty_threshold=0.6,  # Less tolerant of uncertainty
    quality_threshold=0.6       # Higher quality expectations
)

# Create regulator with custom config
regulator = ReasoningRegulator(config=config)

# Or use factory function
regulator = create_reasoning_regulator(config=config)
```

### Integration with the Full Pipeline

```python
# 1. Sensing phase
noise_senser = create_noise_senser()
analysis = noise_senser.analyze_step(current_step, previous_steps, dig)

# 2. Regulation phase (identify root causes)
regulator = create_reasoning_regulator()
result = regulator.analyze_and_regulate(current_step, model_probs)

# 3. Healing phase (act on recommendations)
if result.healing_recommendations:
    for rec in result.healing_recommendations[:3]:  # Top 3 recommendations
        if rec['priority'] == 'critical':
            heal_agent.heal_step(rec['target'])
        elif rec['priority'] == 'high':
            schedule_healing(rec['target'])
```

## API Reference

### Class: ReasoningRegulator

**Attributes:**
- `config`: Regulator configuration parameters
- `noise_senser`: Semantic noise senser instance
- `reasoning_dig`: Dynamic Interaction Graph for the session
- `session_state`: Current reasoning session state
- `previous_steps`: History of processed steps (capped at 20)
- `current_step`: Currently analyzed step

**Methods:**
- `initialize_session(session_id, conversation_id, reasoning_dig=None)`: Initialize a new regulation session
- `analyze_and_regulate(current_step, model_probs=None)`: Main entry point for analysis
- `update_history(step)`: Update step history after processing
- `get_last_regulation()`: Retrieve last regulation result
- `visualize_regulation(regulation_result=None, output_format='text')`: Generate human-readable report
- `visualize_dig_with_analysis(regulation_result=None, highlight_root_causes=True)`: Generate DOT visualization

### Data Classes

#### RegulatorConfig
```python
RegulatorConfig(
    drift_threshold=0.6,
    uncertainty_threshold=0.7,
    quality_threshold=0.5,
    healing_trigger_threshold=0.75,
    root_cause_limit=5,
    use_causal_analysis=True,
    use_uncertainty_amplification=True,
    min_causal_strength=0.3,
    similarity_threshold=0.65
)
```

#### RegulationResult
```python
RegulationResult(
    session_id="",
    current_step_id="",
    issues_detected=[],
    root_causes=[],
    healing_recommendations=[],
    quality_metrics={},
    confidence=0.0,
    timestamp=datetime.now(),
    next_actions=[]
)
```

### Factory Functions

- `create_reasoning_regulator(config=None, noise_senser=None)`: Create a regulator instance
- `create_noise_senser(model_name=None, **kwargs)`: Create a noise senser (from sensing module)

## Performance Considerations

- **History Window**: Configurable limit (default 20 steps) to manage memory
- **Real-time Analysis**: Designed for near real-time regulation in streaming pipelines
- **Batch Processing**: Can analyze multiple steps in batch for efficiency
- **Caching**: Embeddings are cached to avoid recomputation

## Error Handling

The module includes robust error handling:
- Graceful degradation when embedding models are unavailable
- Fallback to TF-IDF vectorization when needed
- Proper handling of missing or malformed data
- Configurable thresholds to avoid false positives

## Testing

The module includes comprehensive unit tests covering:
- Basic initialization and configuration
- Session management
- Drift detection and root cause analysis
- Recommendation generation
- Quality metrics calculation
- Visualization outputs
- Integration with payload and graph modules

Run tests with:
```bash
pytest tests/test_regulating.py -v
```

## Examples

See the `__main__` section of `src/regulating.py` for a complete integration example
demonstrating the full regulation workflow with sample data.

## Dependencies

The ReasoningRegulator relies on the following FLOW-HEAL components:
- `payload.py` (ReasoningStep, ReasoningSession)
- `graph.py` (ReasoningDIG, ReasoningNode, CausalLink)
- `sensing.py` (NoiseSenser)

External dependencies (from requirements.txt):
- networkx>=3.4.0
- pydantic>=2.5.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- sentence-transformers>=2.2.0 (optional)
- nltk>=3.8.0 (optional)

## License

FLOW-HEAL is open source software licensed under the MIT License. See LICENSE for details.

## Support

For questions or support, please refer to the project's GitHub repository or contact the maintainers.

---

**Last Updated:** 2026-04-04
**Module Status:** ✅ COMPLETE