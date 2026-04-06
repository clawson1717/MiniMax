"""
FLOW-HEAL Reasoning Regulator Module

The ReasoningRegulator identifies root causes of semantic drift and uncertainty
in multi-agent reasoning pipelines, providing targeted recommendations for healing.

This module implements the "R" in the SRC (Sensing-Regulating-Correcting) framework,
using the NoiseSenser to detect issues and the ReasoningDIG to pinpoint exactly
which reasoning steps require intervention.

## Key Features

- **Semantic Drift Detection**: Uses embedding distance and textual similarity metrics
- **Uncertainty Quantification**: Analyzes uncertainty scores, hedging language, and coherence
- **Causal Analysis**: Leverages Dynamic Interaction Graph (DIG) to identify root causes
- **Targeted Recommendations**: Provides specific healing actions with confidence scores
- **Quality Metrics**: Calculates session quality, healing urgency, and drift prevalence
- **Visualization**: Generates DOT representations of the DIG with analysis highlights

## Integration

The ReasoningRegulator integrates with the existing FLOW-HEAL components:

1. **Payload Module** (`payload.py`): Uses `ReasoningStep` and `ReasoningSession`
2. **Sensing Module** (`sensing.py`): Uses `NoiseSenser` for semantic analysis
3. **Graph Module** (`graph.py`): Uses `ReasoningDIG` for causal tracking
4. **Main Pipeline**: Provides the regulation layer between sensing and healing

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

The ReasoningRegulator is typically used between the NoiseSenser and the HealAgent:

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
            # Initiate immediate healing
            heal_agent.heal_step(rec['target'])
        elif rec['priority'] == 'high':
            # Schedule healing
            schedule_healing(rec['target'])
```

## API Reference

### Class: ReasoningRegulator

**Attributes:**
- `config`: Regulator configuration parameters
- `noise_senser`: Semantic noise senser instance
- `reasoning_dig`: Dynamic Interaction Graph for the session
- `session_state`: Current reasoning session state
- `previous_steps`: History of processed steps
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

## Quality Metrics

The regulator calculates several session-level quality metrics:

- **session_quality**: Overall quality score (0-1, higher is better)
- **drift_prevalence**: Proportion of recent steps with drift (0-1)
- **uncertainty_level**: Average uncertainty score (0-1)
- **root_cause_severity**: Severity based on root cause quality (0-1)
- **healing_urgency**: Combined urgency score (0-1, higher = more healing needed)

## Root Cause Analysis

The regulator uses the DIG to identify root causes through:

1. **Causal Tracing**: Finds all ancestor steps of problematic nodes
2. **Quality Scoring**: Ranks by low quality scores
3. **Importance Weighting**: Considers node connectivity and centrality
4. **Composite Scoring**: Combines multiple factors for prioritization

## Healing Recommendations

Based on root cause analysis, the regulator generates specific actions:

- `re_execute_step`: Complete re-execution of a low-quality step
- `clarify_step`: Request clarification for high-uncertainty steps
- `review_step`: Careful review of critical nodes
- `targeted_healing`: Focused healing for moderate issues
- `validate_current_step`: Validate the current step before proceeding
- `general_review`: Broad review when no specific cause identified

Each recommendation includes:
- `target`: Step ID to act on
- `confidence`: Confidence in the recommendation (0-1)
- `priority`: Critical, high, medium, or low
- `description`: Human-readable explanation

## Visualization

The regulator provides two visualization methods:

1. **Regulation Report**: Text, markdown, or JSON format summarizing findings
2. **DIG Visualization**: Graphviz DOT representation showing:
   - Node color based on quality (green=good, red=poor)
   - Root causes highlighted in red/diamond shape
   - Current step highlighted in blue/box shape
   - Edge colors based on causal confidence

## Performance Considerations

- **History Window**: The regulator keeps up to 20 previous steps in memory
- **Real-time Analysis**: Designed for near real-time regulation in streaming pipelines
- **Batch Processing**: Can also analyze multiple steps in batch mode
- **Caching**: Embeddings are cached to avoid recomputation

## Error Handling

The regulator includes robust error handling:
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

See the `__main__` section of this file for a complete integration example
demonstrating the full regulation workflow with sample data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
import numpy as np
from typing_extensions import Literal

# Import FLOW-HEAL core components
from .payload import (
    ReasoningStep,
    ReasoningSession,
    StepStatus,
    AgentRole,
    CausalDependency
)
from .graph import (
    ReasoningDIG,
    ReasoningNode,
    CausalLink,
    CausalRelationship
)
from .sensing import NoiseSenser, create_noise_senser


# ============================================================================
# REGULATOR CONFIGURATION
# ============================================================================

@dataclass
class RegulatorConfig:
    """
    Configuration for the ReasoningRegulator.
    
    Attributes:
        drift_threshold: Minimum drift confidence to trigger investigation
        uncertainty_threshold: Maximum acceptable uncertainty score
        quality_threshold: Minimum acceptable quality score
        healing_trigger_threshold: Combined score triggering healing recommendation
        root_cause_limit: Maximum number of root causes to return
        use_causal_analysis: Whether to use DIG-based causal analysis
        use_uncertainty_amplification: Whether to amplify uncertainty for critical nodes
        min_causal_strength: Minimum causal link strength to consider
        similarity_threshold: Minimum semantic similarity threshold
    """
    drift_threshold: float = 0.6
    uncertainty_threshold: float = 0.7
    quality_threshold: float = 0.5
    healing_trigger_threshold: float = 0.75
    root_cause_limit: int = 5
    use_causal_analysis: bool = True
    use_uncertainty_amplification: bool = True
    min_causal_strength: float = 0.3
    similarity_threshold: float = 0.65


# ============================================================================
# REGULATOR RESULTS
# ============================================================================

@dataclass
class RegulationResult:
    """
    Result of the regulation analysis.
    
    Attributes:
        session_id: ID of the reasoning session
        current_step_id: ID of the current step being analyzed
        issues_detected: List of detected issues (flags)
        root_causes: Identified root cause step IDs with scores
        healing_recommendations: Specific healing actions recommended
        quality_metrics: Overall quality metrics for the session
        confidence: Regulator's confidence in the findings
        timestamp: When the analysis was performed
        next_actions: Suggested next steps for the system
    """
    session_id: str
    current_step_id: str
    issues_detected: List[Dict[str, Any]] = field(default_factory=list)
    root_causes: List[Dict[str, Any]] = field(default_factory=list)
    healing_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    next_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "current_step_id": self.current_step_id,
            "issues_detected": self.issues_detected,
            "root_causes": self.root_causes,
            "healing_recommendations": self.healing_recommendations,
            "quality_metrics": self.quality_metrics,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "next_actions": self.next_actions
        }


# ============================================================================
# REASONING REGULATOR CLASS
# ============================================================================

class ReasoningRegulator:
    """
    FLOW-HEAL Reasoning Regulator - The "R" in SRC (Sensing-Regulating-Correcting).
    
    This class analyzes reasoning steps using:
    1. NoiseSenser outputs to detect semantic drift and uncertainty
    2. Dynamic Interaction Graph (DIG) to identify causal relationships
    3. Quality metrics to assess reasoning step validity
    
    It then determines which steps are likely causing problems and provides
    targeted recommendations for the healing agent.
    
    Attributes:
        config: Regulator configuration
        noise_senser: Semantic noise senser instance
        reasoning_dig: Dynamic Interaction Graph for the session
        session_state: Current reasoning session state
    """
    
    def __init__(
        self,
        config: Optional[RegulatorConfig] = None,
        noise_senser: Optional[NoiseSenser] = None
    ) -> None:
        """
        Initialize the ReasoningRegulator.
        
        Args:
            config: Configuration for the regulator (default uses reasonable values)
            noise_senser: NoiseSenser instance (creates default if None)
        """
        self.config = config or RegulatorConfig()
        
        if noise_senser is not None:
            self.noise_senser = noise_senser
        else:
            # Create default noise senser with similar configuration
            self.noise_senser = create_noise_senser(
                similarity_threshold=self.config.similarity_threshold
            )
        
        # State tracking
        self.reasoning_dig: Optional[ReasoningDIG] = None
        self.session_state: Optional[ReasoningSession] = None
        self.previous_steps: List[ReasoningStep] = []
        self.current_step: Optional[ReasoningStep] = None
        
        # Analysis results
        self._analysis_results: Dict[str, Any] = {}
        self._regulation_results: Optional[RegulationResult] = None
    
    def initialize_session(
        self,
        session_id: str,
        conversation_id: str,
        reasoning_dig: Optional[ReasoningDIG] = None
    ) -> None:
        """
        Initialize a new regulation session.
        
        Args:
            session_id: Unique session identifier
            conversation_id: Conversation identifier
            reasoning_dig: Optional pre-built ReasoningDIG
        """
        self.session_state = ReasoningSession(
            session_id=session_id,
            conversation_id=conversation_id
        )
        self.reasoning_dig = reasoning_dig or ReasoningDIG(
            session_id=session_id,
            conversation_id=conversation_id
        )
        self.previous_steps = []
        self.current_step = None
        self._analysis_results = {}
        self._regulation_results = None
    
    def analyze_and_regulate(
        self,
        current_step: ReasoningStep,
        model_probs: Optional[Dict[str, float]] = None
    ) -> RegulationResult:
        """
        Analyze a reasoning step and determine regulation actions.
        
        This is the main entry point for the regulator. It:
        1. Analyzes the current step with the NoiseSenser
        2. Updates the DIG with new analysis
        3. Identifies root causes of any issues
        4. Generates healing recommendations
        
        Args:
            current_step: The current reasoning step to analyze
            model_probs: Optional probability distribution from the language model
            
        Returns:
            RegulationResult with findings and recommendations
        """
        self.current_step = current_step
        
        # Add step to session and DIG if not already present
        if self.session_state and current_step.step_id not in self.session_state.steps:
            self.session_state.add_step(current_step)
        
        if self.reasoning_dig and current_step.step_id not in self.reasoning_dig.nodes:
            # Create a ReasoningNode from the step
            node_data = current_step.payload.get('node_data', {})
            node = ReasoningNode(
                step_id=current_step.step_id,
                agent_id=current_step.agent_id,
                agent_role=current_step.agent_role.value,
                intent_hash=current_step.intent_hash,
                content=current_step.content,
                **node_data
            )
            self.reasoning_dig.add_node(node)
        
        # Perform comprehensive analysis
        analysis = self._analyze_current_step(current_step, model_probs)
        self._analysis_results[current_step.step_id] = analysis
        
        # Update step with analysis results
        updated_step = self.noise_senser.update_step_with_analysis(
            current_step, analysis, self.reasoning_dig
        )
        
        # Identify root causes
        root_causes = self._identify_root_causes()
        
        # Generate healing recommendations
        recommendations = self._generate_recommendations(root_causes, analysis)
        
        # Calculate overall quality metrics
        quality_metrics = self._calculate_quality_metrics(analysis, root_causes)
        
        # Determine regulator confidence
        confidence = self._calculate_confidence(analysis, root_causes)
        
        # Determine next actions
        next_actions = self._determine_next_actions(root_causes, recommendations)
        
        # Create regulation result
        regulation_result = RegulationResult(
            session_id=self.session_state.session_id if self.session_state else "unknown",
            current_step_id=current_step.step_id,
            issues_detected=analysis.get('flags', []),
            root_causes=root_causes,
            healing_recommendations=recommendations,
            quality_metrics=quality_metrics,
            confidence=confidence,
            next_actions=next_actions
        )
        
        self._regulation_results = regulation_result
        return regulation_result
    
    def _analyze_current_step(
        self,
        step: ReasoningStep,
        model_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the current reasoning step.
        
        Args:
            step: The reasoning step to analyze
            model_probs: Optional model probability distribution
            
        Returns:
            Analysis dictionary with drift, uncertainty, and quality metrics
        """
        if not self.previous_steps:
            # First step in the session - no drift to detect
            analysis = {
                'step_id': step.step_id,
                'agent_id': step.agent_id,
                'agent_role': step.agent_role.value,
                'analysis_timestamp': datetime.now().isoformat(),
                'drift_analysis': {
                    'drift_detected': False,
                    'confidence': 0.0,
                    'similarity_scores': {},
                    'drift_indicators': [],
                    'recommendations': []
                },
                'uncertainty_score': self.noise_senser.quantify_uncertainty(step, model_probs),
                'quality_metrics': {
                    'built_in_quality': step.calculate_overall_quality(),
                    'uncertainty_normalized': 0.0,
                    'semantic_coherence': 1.0
                },
                'dig_context': {},
                'flags': ['first_step'],
                'recommendations': ['monitor_next_step']
            }
            return analysis
        
        # Get previous step for comparison
        previous_step = self.previous_steps[-1] if self.previous_steps else None
        
        # Run NoiseSenser analysis
        analysis = self.noise_senser.analyze_step(
            current_step=step,
            previous_steps=self.previous_steps,
            dig=self.reasoning_dig,
            model_probs=model_probs
        )
        
        return analysis
    
    def _identify_root_causes(self) -> List[Dict[str, Any]]:
        """
        Identify root causes of detected issues using DIG analysis.
        
        Returns:
            List of root cause step IDs with scores and importance metrics
        """
        if not self.current_step or not self.reasoning_dig:
            return []
        
        # Check if any issues were detected
        analysis = self._analysis_results.get(self.current_step.step_id, {})
        flags = analysis.get('flags', [])
        drift_detected = any('drift' in flag for flag in flags)
        high_uncertainty = any('uncertainty' in flag for flag in flags)
        
        if not drift_detected and not high_uncertainty:
            return []
        
        # Use DIG to identify potential root causes
        target_step_id = self.current_step.step_id
        root_causes = self.reasoning_dig.identify_root_causes(target_step_id)
        
        # Rank and limit root causes
        ranked_causes = []
        for cause in root_causes[:self.config.root_cause_limit]:
            step_id = cause["step_id"]
            node = self.reasoning_dig.get_node(step_id)
            if node:
                # Calculate composite score
                quality_score = node.quality_score
                uncertainty = node.uncertainty_score
                importance = cause.get("importance_score", 0)
                
                # Adjust score based on regulator config
                score = self._calculate_root_cause_score(quality_score, uncertainty, importance)
                
                ranked_causes.append({
                    "step_id": step_id,
                    "quality_score": quality_score,
                    "uncertainty_score": uncertainty,
                    "importance_score": importance,
                    "composite_score": score,
                    "node_data": node.to_dict()
                })
        
        # Sort by composite score (descending)
        ranked_causes.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return ranked_causes
    
    def _calculate_root_cause_score(
        self,
        quality_score: float,
        uncertainty_score: float,
        importance_score: float
    ) -> float:
        """
        Calculate a composite score for root cause prioritization.
        
        Args:
            quality_score: Quality of the reasoning step (0-1, lower is worse)
            uncertainty_score: Uncertainty level (0-1, higher is worse)
            importance_score: Importance in the DIG (0-1, higher is worse)
            
        Returns:
            Composite score (0-1, higher is more likely root cause)
        """
        # Invert quality score (lower quality = higher priority)
        quality_penalty = (1 - quality_score) * 0.4
        
        # Uncertainty penalty
        uncertainty_penalty = uncertainty_score * 0.4
        
        # Importance penalty (more connected nodes have higher impact)
        importance_penalty = importance_score * 0.2
        
        # Combine penalties (higher total = more likely root cause)
        return quality_penalty + uncertainty_penalty + importance_penalty
    
    def _generate_recommendations(
        self,
        root_causes: List[Dict[str, Any]],
        current_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate healing recommendations based on root causes and analysis.
        
        Args:
            root_causes: Identified root cause steps with scores
            current_analysis: Current step analysis results
            
        Returns:
            List of healing recommendations with confidence scores
        """
        recommendations = []
        
        # If no root causes found but drift was detected, recommend general healing
        flags = current_analysis.get('flags', [])
        if not root_causes and any('drift' in flag for flag in flags):
            recommendations.append({
                "action": "general_review",
                "target": "current_step",
                "confidence": 0.7,
                "description": "General review of current reasoning step recommended",
                "priority": "high"
            })
            return recommendations
        
        # Generate specific recommendations based on root causes
        for cause in root_causes[:3]:  # Limit to top 3 causes
            step_id = cause["step_id"]
            quality = cause["quality_score"]
            uncertainty = cause["uncertainty_score"]
            importance = cause["importance_score"]
            
            # Determine recommended action based on node characteristics
            if quality < 0.3:
                # Low quality - likely needs complete re-reasoning
                recommendations.append({
                    "action": "re_execute_step",
                    "target": step_id,
                    "confidence": 0.9,
                    "description": "Low quality step detected - recommend re-execution",
                    "priority": "critical",
                    "reason": "quality_too_low"
                })
            elif uncertainty > 0.7:
                # High uncertainty - recommend clarification or additional analysis
                recommendations.append({
                    "action": "clarify_step",
                    "target": step_id,
                    "confidence": 0.8,
                    "description": "High uncertainty step - recommend clarification",
                    "priority": "high",
                    "reason": "high_uncertainty"
                })
            elif importance > 0.6:
                # High importance node - recommend careful review
                recommendations.append({
                    "action": "review_step",
                    "target": step_id,
                    "confidence": 0.7,
                    "description": "High importance node - recommend careful review",
                    "priority": "medium",
                    "reason": "critical_node"
                })
            else:
                # Moderate issue - recommend targeted healing
                recommendations.append({
                    "action": "targeted_healing",
                    "target": step_id,
                    "confidence": 0.6,
                    "description": "Moderate quality issue - recommend targeted healing",
                    "priority": "medium",
                    "reason": "moderate_quality"
                })
        
        # Add recommendations for the current step if it has issues
        if current_analysis.get('drift_detected', False):
            recommendations.append({
                "action": "validate_current_step",
                "target": self.current_step.step_id if self.current_step else "current",
                "confidence": 0.8,
                "description": "Current step shows drift - validate before proceeding",
                "priority": "high",
                "reason": "current_step_drift"
            })
        
        return recommendations
    
    def _calculate_quality_metrics(
        self,
        analysis: Dict[str, Any],
        root_causes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate overall quality metrics for the session.
        
        Args:
            analysis: Current step analysis
            root_causes: Identified root causes
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'session_quality': 0.5,
            'drift_prevalence': 0.0,
            'uncertainty_level': 0.0,
            'root_cause_severity': 0.0,
            'healing_urgency': 0.0
        }
        
        # Calculate drift prevalence (if multiple steps analyzed)
        if self.previous_steps:
            drift_count = 0
            for step in self.previous_steps[-5:]:  # Last 5 steps
                step_analysis = self._analysis_results.get(step.step_id, {})
                if step_analysis.get('drift_analysis', {}).get('drift_detected', False):
                    drift_count += 1
            metrics['drift_prevalence'] = drift_count / min(len(self.previous_steps), 5)
        
        # Current uncertainty level
        metrics['uncertainty_level'] = analysis.get('uncertainty_score', 0.0)
        
        # Root cause severity
        if root_causes:
            avg_quality = np.mean([rc.get('quality_score', 0.5) for rc in root_causes])
            metrics['root_cause_severity'] = 1 - avg_quality  # Lower quality = higher severity
        
        # Healing urgency (combination of factors)
        urgency_factors = [
            metrics['drift_prevalence'] * 0.3,
            metrics['uncertainty_level'] * 0.3,
            metrics['root_cause_severity'] * 0.4
        ]
        metrics['healing_urgency'] = min(sum(urgency_factors), 1.0)
        
        # Overall session quality (inverse of urgency)
        metrics['session_quality'] = max(0.0, 1.0 - metrics['healing_urgency'])
        
        return metrics
    
    def _calculate_confidence(
        self,
        analysis: Dict[str, Any],
        root_causes: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate regulator confidence in the analysis.
        
        Args:
            analysis: Current step analysis
            root_causes: Identified root causes
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Reduce confidence if drift detection is uncertain
        drift_confidence = analysis.get('drift_analysis', {}).get('confidence', 1.0)
        confidence *= min(drift_confidence + 0.1, 1.0)
        
        # Reduce confidence if root causes are low quality
        if root_causes:
            avg_quality = np.mean([rc.get('quality_score', 0.5) for rc in root_causes])
            confidence *= (0.5 + avg_quality * 0.5)  # Adjust based on quality
        
        # Always keep at least some confidence
        return max(confidence, 0.3)
    
    def _determine_next_actions(self, root_causes: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]) -> List[str]:
        """
        Determine suggested next actions for the system.
        
        Args:
            root_causes: Identified root causes
            recommendations: Generated recommendations
            
        Returns:
            List of suggested actions
        """
        actions = []
        
        if not root_causes and not recommendations:
            actions.append("continue_monitoring")
        elif recommendations:
            if any(rec.get('priority') == 'critical' for rec in recommendations):
                actions.append("initiate_healing_immediately")
            elif any(rec.get('priority') == 'high' for rec in recommendations):
                actions.append("schedule_healing_soon")
            else:
                actions.append("plan_healing_intervention")
        
        # Add specific action based on session state
        if self.current_step and self.current_step.agent_role in [AgentRole.HEALER, AgentRole.VALIDATOR]:
            actions.append("prepare_healing_agent")
        
        return actions
    
    def update_history(self, step: ReasoningStep) -> None:
        """
        Update the regulator's history with a new reasoning step.
        
        This should be called after each reasoning step completes.
        
        Args:
            step: The reasoning step to add to history
        """
        self.previous_steps.append(step)
        if len(self.previous_steps) > 20:  # Keep limited history
            self.previous_steps.pop(0)
        
        # Update DIG with causal relationships
        if self.reasoning_dig and self.current_step:
            # Add causal links from current step to previous (simplified)
            for prev_step in self.previous_steps[-3:]:
                if prev_step.step_id != self.current_step.step_id:
                    link = CausalLink(
                        source_step_id=prev_step.step_id,
                        target_step_id=self.current_step.step_id,
                        relationship_type=CausalRelationship.DEPENDENCY,
                        confidence=0.5
                    )
                    self.reasoning_dig.add_link(link)
    
    def get_last_regulation(self) -> Optional[RegulationResult]:
        """Get the last regulation result."""
        return self._regulation_results
    
    def visualize_regulation(
        self,
        regulation_result: Optional[RegulationResult] = None,
        output_format: Literal['text', 'markdown', 'json'] = 'text'
    ) -> str:
        """
        Create a visual representation of the regulation results.
        
        Args:
            regulation_result: Result to visualize (uses last if None)
            output_format: Format for output
            
        Returns:
            String representation of the regulation results
        """
        result = regulation_result or self._regulation_results
        if not result:
            return "No regulation results available"
        
        if output_format == 'markdown':
            return self._format_markdown(result)
        elif output_format == 'json':
            import json
            return json.dumps(result.to_dict(), indent=2, default=str)
        else:  # text
            return self._format_text(result)
    
    def _format_text(self, result: RegulationResult) -> str:
        """Format regulation result as plain text."""
        lines = [
            "=" * 60,
            "FLOW-HEAL REGULATION REPORT",
            "=" * 60,
            f"Session: {result.session_id}",
            f"Current Step: {result.current_step_id}",
            f"Timestamp: {result.timestamp}",
            f"Confidence: {result.confidence:.3f}",
            "-" * 60
        ]
        
        if result.issues_detected:
            lines.append("ISSUES DETECTED:")
            for issue in result.issues_detected:
                lines.append(f"  - {issue}")
        else:
            lines.append("No issues detected (quality within acceptable range)")
        
        if result.root_causes:
            lines.append("\nROOT CAUSES IDENTIFIED:")
            for i, cause in enumerate(result.root_causes[:3], 1):
                lines.append(f"  {i}. Step {cause['step_id']}:")
                lines.append(f"     Quality: {cause['quality_score']:.2f}")
                lines.append(f"     Uncertainty: {cause['uncertainty_score']:.2f}")
                lines.append(f"     Importance: {cause['importance_score']:.2f}")
                lines.append(f"     Composite Score: {cause['composite_score']:.3f}")
        else:
            lines.append("\nNo root causes identified")
        
        if result.healing_recommendations:
            lines.append("\nHEALING RECOMMENDATIONS:")
            for rec in result.healing_recommendations:
                lines.append(f"  - {rec['description']}")
                lines.append(f"    Action: {rec['action']}")
                lines.append(f"    Target: {rec['target']}")
                lines.append(f"    Confidence: {rec['confidence']:.2f}")
                lines.append(f"    Priority: {rec['priority']}")
        else:
            lines.append("\nNo healing recommendations at this time")
        
        lines.append("\nQUALITY METRICS:")
        for key, value in result.quality_metrics.items():
            lines.append(f"  {key:20}: {value:.3f}")
        
        lines.append("\nNEXT ACTIONS:")
        for action in result.next_actions:
            lines.append(f"  - {action}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_markdown(self, result: RegulationResult) -> str:
        """Format regulation result as markdown."""
        lines = [
            "# FLOW-HEAL Regulation Report\n",
            f"**Session ID:** {result.session_id}",
            f"**Current Step:** {result.current_step_id}",
            f"**Timestamp:** {result.timestamp}",
            f"**Confidence:** {result.confidence:.3f}\n",
            "## Issues Detected"
        ]
        
        if result.issues_detected:
            for issue in result.issues_detected:
                lines.append(f"- {issue}")
        else:
            lines.append("- No issues detected")
        
        lines.extend(["## Root Causes Identified", "No root causes identified"])
        if result.root_causes:
            lines.append("| Rank | Step ID | Quality | Uncertainty | Importance | Composite Score |")
            lines.append("|------|---------|---------|-------------|------------|-----------------|")
            for i, cause in enumerate(result.root_causes[:5], 1):
                lines.append(f"| {i} | {cause['step_id']} | {cause['quality_score']:.2f} | {cause['uncertainty_score']:.2f} | {cause['importance_score']:.2f} | {cause['composite_score']:.3f} |")
        
        lines.extend(["## Healing Recommendations", "No recommendations"])
        if result.healing_recommendations:
            for rec in result.healing_recommendations:
                lines.append(f"- **{rec['description']}**")
                lines.append(f"  - Action: {rec['action']}")
                lines.append(f"  - Target: {rec['target']}")
                lines.append(f"  - Confidence: {rec['confidence']:.2f}")
                lines.append(f"  - Priority: {rec['priority']}")
        
        lines.extend(["## Quality Metrics", "| Metric | Value |", "|-------|-------|"])
        for key, value in result.quality_metrics.items():
            lines.append(f"| {key} | {value:.3f} |")
        
        lines.extend(["## Next Actions", "No actions recommended"])
        if result.next_actions:
            for action in result.next_actions:
                lines.append(f"- {action}")
        
        return "\n".join(lines)
    
    def visualize_dig_with_analysis(
        self,
        regulation_result: Optional[RegulationResult] = None,
        highlight_root_causes: bool = True
    ) -> str:
        """
        Visualize the DIG with analysis highlights.
        
        Args:
            regulation_result: Regulation result to use (uses last if None)
            highlight_root_causes: Whether to highlight root causes
            
        Returns:
            Graphviz DOT representation of the DIG
        """
        result = regulation_result or self._regulation_results
        if not result or not self.reasoning_dig:
            return "No DIG available for visualization"
        
        import graphviz as gv
        
        dot = gv.Digraph(
            comment=f"Reasoning DIG for {result.session_id}",
            graph_attr={"rankdir": "LR", "splines": "ortho", "nodesep": "0.6", "ranksep": "0.8"}
        )
        
        # Add nodes with analysis-based styling
        root_cause_ids = [rc["step_id"] for rc in result.root_causes] if highlight_root_causes else []
        
        for step_id, node in self.reasoning_dig.nodes.items():
            node_data = self.reasoning_dig.graph.nodes[step_id].get('node_data', {})
            quality = node_data.get('quality_score', 0.5)
            uncertainty = node_data.get('uncertainty_score', 0.5)
            
            # Determine node style based on analysis
            if step_id == result.current_step_id:
                color = "lightblue"
                shape = "box"
                label = f"{step_id}\n(current)"
            elif step_id in root_cause_ids:
                color = "lightcoral"
                shape = "diamond"
                label = f"{step_id}\n(root cause)"
            else:
                # Color based on quality
                if quality > 0.7:
                    color = "palegreen2"
                elif quality > 0.4:
                    color = "gold"
                else:
                    color = "lightcoral"
                shape = "ellipse"
                label = step_id
            
            # Add tooltip with node data
            tooltip = f"Quality: {quality:.2f}\nUncertainty: {uncertainty:.2f}\nRole: {node.agent_role}"
            
            dot.node(
                step_id,
                label=label,
                shape=shape,
                style="filled,bold" if step_id in root_cause_ids else "filled",
                color=color,
                tooltip=tooltip,
                fillcolor=color,
                fontname="DejaVu Sans"
            )
        
        # Add edges with relationship types
        for (src, dst), link in self.reasoning_dig.links.items():
            link_data = self.reasoning_dig.graph.edges[src, dst].get('link_data', {})
            relationship = link_data.get('relationship_type', 'dependency')
            confidence = link_data.get('confidence', 1.0)
            
            # Edge style based on confidence
            if confidence > 0.8:
                color = "darkgreen"
                penwidth = "2.5"
            elif confidence > 0.5:
                color = "goldenrod"
                penwidth = "1.5"
            else:
                color = "red"
                penwidth = "1.0"
            
            dot.edge(
                src,
                dst,
                label=relationship.replace('_', ' '),
                color=color,
                penwidth=penwidth,
                fontname="DejaVu Sans",
                fontsize="10"
            )
        
        return dot.source


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_reasoning_regulator(
    config: Optional[RegulatorConfig] = None,
    noise_senser: Optional[NoiseSenser] = None
) -> ReasoningRegulator:
    """
    Factory function to create a ReasoningRegulator instance.
    
    Args:
        config: Configuration for the regulator
        noise_senser: NoiseSenser instance (creates default if None)
        
    Returns:
        Configured ReasoningRegulator instance
    """
    return ReasoningRegulator(config=config, noise_senser=noise_senser)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FLOW-HEAL REASONING REGULATOR - INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Initialize the regulator
    print("\n1. Initializing ReasoningRegulator...")
    regulator = create_reasoning_regulator()
    print(f"   - Config: {regulator.config}")
    print(f"   - NoiseSenser: {regulator.noise_senser}")
    
    # Initialize a session
    print("\n2. Initializing regulation session...")
    regulator.initialize_session(
        session_id="demo_session_001",
        conversation_id="demo_conversation_001"
    )
    
    # Create sample reasoning steps
    print("\n3. Creating sample reasoning steps...")
    from uuid import uuid4
    
    step1 = ReasoningStep(
        step_id="step_001",
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for the FLOW-HEAL system.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8,
        causal_dependencies=[
            CausalDependency(parent_step_id="initial_prompt", relationship_type="follows_from")
        ]
    )
    
    step2 = ReasoningStep(
        step_id="step_002",
        agent_id="researcher_agent",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will research the requirements for the FLOW-HEAL system from the three source papers.",
        uncertainty_score=0.3,
        coherence_score=0.8,
        logic_score=0.7,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_001", relationship_type="contextual")
        ]
    )
    
    step3 = ReasoningStep(
        step_id="step_003",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="design_architecture",
        content="Based on the research, I will design the overall architecture for FLOW-HEAL, focusing on the integration points between DenoiseFlow, DIG to Heal, and LOGIGEN.",
        uncertainty_score=0.5,
        coherence_score=0.7,
        logic_score=0.6,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="evidence_for")
        ]
    )
    
    step4 = ReasoningStep(
        step_id="step_004",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="design_detailed",
        content="I will now design the detailed components: the FlowRegulator, HealAgent, and Verifier, ensuring they work together seamlessly.",
        uncertainty_score=0.6,
        coherence_score=0.6,
        logic_score=0.5,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="follows_from")
        ]
    )
    
    # Simulate processing steps
    print("\n4. Processing reasoning steps through the regulator...")
    steps_to_process = [step1, step2, step3, step4]
    
    for i, step in enumerate(steps_to_process, 1):
        print(f"\n   Processing step {i}: {step.step_id}")
        print(f"     Agent: {step.agent_role.value}")
        print(f"     Content: {step.content[:60]}...")
        
        # Analyze and regulate
        result = regulator.analyze_and_regulate(step)
        
        print(f"     Regulation Result:")
        print(f"     - Issues detected: {result.issues_detected if result.issues_detected else 'None'}")
        print(f"     - Root causes: {len(result.root_causes)} identified")
        print(f"     - Healing recommendations: {len(result.healing_recommendations)}")
        print(f"     - Session quality: {result.quality_metrics.get('session_quality', 0):.2f}")
        print(f"     - Healing urgency: {result.quality_metrics.get('healing_urgency', 0):.2f}")
        
        # Update regulator history
        regulator.update_history(step)
    
    # Show final regulation report
    print("\n5. Final Regulation Report:")
    print(regulator.visualize_regulation(output_format='markdown'))
    
    # Visualize the DIG with analysis
    print("\n6. Visualizing the Dynamic Interaction Graph (DIG) with analysis highlights:")
    dig_viz = regulator.visualize_dig_with_analysis()
    print("DOT representation generated. To visualize, save to a .dot file and use Graphviz.")
    # For demonstration, we'll just show a snippet
    print(dig_viz[:500] + "..." if len(dig_viz) > 500 else dig_viz)
    
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("=" * 60)
