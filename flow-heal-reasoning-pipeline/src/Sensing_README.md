# Semantic Noise Senser (Step 4 of FLOW-HEAL)

The Semantic Noise Senser (`NoiseSenser` class) is a critical component of the FLOW-HEAL self-correcting reasoning pipeline. It detects semantic drift and quantifies uncertainty in multi-agent reasoning steps, providing essential input to the SRC Regulator for targeted healing interventions.

## Core Functionality

### 1. Semantic Drift Detection
Uses embedding distance metrics (cosine similarity, Jaccard index) to measure semantic similarity between consecutive reasoning steps. Significant drops in similarity indicate potential "broken" reasoning paths.

### 2. Uncertainty Quantification
Combines multiple indicators to produce an overall uncertainty score:
- Built-in uncertainty/coherence/logic scores from reasoning steps
- Linguistic markers (hedging words, first-person uncertainty phrases)
- Self-correction indicators (question marks, "but I" corrections)
- Model probability entropy (if available)

### 3. DIG Integration
Analyzes steps within the context of the Dynamic Interaction Graph (DIG), identifying:
- Critical nodes with high betweenness centrality
- Highly connected nodes that influence many others
- Root cause analysis for targeted healing

## Usage

### Basic Usage

```python
from src.sensing import NoiseSenser
from src.payload import ReasoningStep, AgentRole

# Create a senser instance
senser = NoiseSenser(
    similarity_threshold=0.65,
    uncertainty_threshold=0.7,
    history_window=3
)

# Create reasoning steps
step1 = ReasoningStep(
    step_id="step_1",
    agent_id="planner",
    agent_role=AgentRole.PLANNER,
    intent_hash="plan_initial",
    content="I will create an initial project plan."
)

step2 = ReasoningStep(
    step_id="step_2",
    agent_id="researcher",
    agent_role=AgentRole.RESEARCHER,
    intent_hash="research_requirements",
    content="I will research the system requirements."
)

# Detect drift between steps
drift_result = senser.detect_semantic_drift(
    current_step=step2,
    previous_step=step1
)

print(f"Drift detected: {drift_result['drift_detected']}")
print(f"Drift confidence: {drift_result['confidence']:.3f}")
```

### Full Pipeline Integration

```python
from src.payload import ReasoningSession
from src.graph import ReasoningDIG
from src.sensing import NoiseSenser

# Create reasoning session with multiple steps
session = ReasoningSession(session_id="demo", conversation_id="demo")

# Add steps to session (omitted for brevity)

# Build Dynamic Interaction Graph from session
dig = ReasoningDIG(session_id=session.session_id, conversation_id=session.conversation_id)
# ... add nodes and links to DIG ...

# Create senser and analyze all steps
senser = NoiseSenser()
analyses = senser.batch_analyze(session.steps.values(), dig)

# Update steps with analysis results
for step in session.steps.values():
    analysis = next(a for a in analyses if a['step_id'] == step.step_id)
    senser.update_step_with_analysis(step, analysis, dig)
```

### Factory Function

Use `create_noise_senser()` for easier instantiation with optional custom embedding models:

```python
from src.sensing import create_noise_senser

# Create with default model
senser = create_noise_senser()

# Create with custom model name
senser = create_noise_senser(model_name='paraphrase-multilingual-MiniLM-L12-v2')
```

## Integration with FLOW-HEAL Pipeline

The NoiseSenser is designed to work seamlessly with other FLOW-HEAL components:

1. **Payload Module** (`payload.py`): Analyzes `ReasoningStep` objects
2. **Graph Module** (`graph.py`): Integrates with `ReasoningDIG` for causal context
3. **Regulating Module** (Step 5): Provides input for root cause identification
4. **Healing Module** (Step 6): Targets problematic nodes for correction

## Key Methods

### `detect_semantic_drift(current_step, previous_step, next_step, context_steps)`
Detects semantic drift between reasoning steps using multiple similarity metrics.

### `quantify_uncertainty(step, model_probs, include_context)`
Quantifies uncertainty for a reasoning step using linguistic and contextual indicators.

### `analyze_step(current_step, previous_steps, dig, model_probs)`
Comprehensive analysis of a single step within its context.

### `batch_analyze(steps, dig, model_probs_list)`
Analyzes multiple steps in sequence, maintaining proper context.

### `update_step_with_analysis(step, analysis, dig)`
Updates a reasoning step with analysis results, modifying quality metrics and metadata.

## Configuration Parameters

- `similarity_threshold`: Minimum semantic similarity to avoid drift flag (default: 0.65)
- `uncertainty_threshold`: Threshold for high uncertainty flag (default: 0.7)
- `use_tfidf_fallback`: Use TF-IDF if embedding model unavailable (default: True)
- `history_window`: Number of previous steps to consider for context (default: 3)

## Output Format

The senser produces structured analysis dictionaries that can be easily consumed by other pipeline components:

```python
{
    'step_id': 'step_001',
    'agent_id': 'planner_agent',
    'agent_role': 'planner',
    'analysis_timestamp': '2026-04-04T17:22:00.000000',
    'drift_analysis': {
        'drift_detected': True,
        'confidence': 0.42,
        'similarity_scores': {'previous': 0.58, 'jaccard_previous': 0.31},
        'drift_indicators': ['low_similarity_to_previous'],
        'recommendations': ['recheck_premises']
    },
    'uncertainty_score': 0.65,
    'quality_metrics': {
        'built_in_quality': 0.75,
        'drift_confidence': 0.42,
        'uncertainty_normalized': 0.65,
        'semantic_coherence': 0.58
    },
    'dig_context': {
        'in_degree': 2,
        'out_degree': 1,
        'quality_score': 0.72,
        'importance': {
            'betweenness_centrality': 0.15,
            'closeness_centrality': 0.42,
            'connectivity_score': 0.6
        }
    },
    'flags': ['semantic_drift', 'drift_confidence_0.42'],
    'recommendations': ['recheck_premises']
}
```

## Error Handling and Fallbacks

The NoiseSenser includes robust error handling:
- Graceful degradation when embedding models are unavailable
- TF-IDF vectorization fallback for semantic similarity
- Simple count-based vectorization as last resort
- Comprehensive warnings for debugging

## Performance Considerations

- Embeddings are cached to avoid recomputation
- Batch processing for efficient analysis of multiple steps
- Configurable history window to balance context vs. performance

## Testing

Unit tests are provided in `test_sensing.py` covering:
- Basic instantiation
- Semantic similarity calculation
- Drift detection
- Uncertainty quantification
- Full batch analysis
- DIG integration
- Visualization output

Run tests with: `python -m pytest src/test_sensing.py`

## Example Usage

See `example_sensing.py` for a complete demonstration of integrating the NoiseSenser with the full FLOW-HEAL pipeline, including:
- Creating a reasoning session with multiple agents
- Building a Dynamic Interaction Graph
- Running comprehensive analysis
- Generating reports
- Saving results to JSON

## Dependencies

- **Required**: numpy, pydantic, typing_extensions
- **Optional**: sentence-transformers, nltk, scikit-learn (fallback)

If optional dependencies are unavailable, the senser will use simpler but less accurate methods.

## Innovation

The Semantic Noise Senser implements key innovations from the DenoiseFlow paper:
- Uncertainty-aware denoising for reliable agentic workflows
- Real-time semantic drift detection between reasoning turns
- Multi-modal uncertainty quantification combining model scores and linguistic cues
- Integration with DIG for causal root cause analysis

This enables the FLOW-HEAL system to proactively "heal" reasoning paths when semantic drift is detected, resulting in more truth-resilient final answers.