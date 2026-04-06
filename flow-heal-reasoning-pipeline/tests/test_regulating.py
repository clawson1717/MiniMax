"""
Unit Tests for FLOW-HEAL ReasoningRegulator

This test suite covers the ReasoningRegulator class, ensuring it correctly:
- Analyzes reasoning steps for semantic drift and uncertainty
- Identifies root causes using the Dynamic Interaction Graph (DIG)
- Generates appropriate healing recommendations
- Calculates quality metrics and confidence scores
- Handles edge cases and error conditions
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import FLOW-HEAL components
from flow_heal.src.payload import (
    ReasoningStep,
    ReasoningSession,
    StepStatus,
    AgentRole,
    CausalDependency
)
from flow_heal.src.graph import (
    ReasoningDIG,
    ReasoningNode,
    CausalLink,
    CausalRelationship
)
from flow_heal.src.sensing import NoiseSenser, create_noise_senser
from flow_heal.src.regulating import (
    ReasoningRegulator,
    RegulatorConfig,
    RegulationResult,
    create_reasoning_regulator
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_regulator():
    """Create a ReasoningRegulator instance with default configuration."""
    return create_reasoning_regulator()


@pytest.fixture
def sample_session():
    """Create a sample reasoning session."""
    return ReasoningSession(
        session_id="test_session_001",
        conversation_id="test_conversation_001"
    )


@pytest.fixture
def sample_dig():
    """Create a sample Dynamic Interaction Graph."""
    return ReasoningDIG(
        session_id="test_session_001",
        conversation_id="test_conversation_001"
    )


@pytest.fixture
def sample_step_1() -> ReasoningStep:
    """Create a sample high-quality reasoning step."""
    return ReasoningStep(
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


@pytest.fixture
def sample_step_2() -> ReasoningStep:
    """Create a sample moderate-quality reasoning step."""
    return ReasoningStep(
        step_id="step_002",
        agent_id="researcher_agent",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will research the requirements for the FLOW-HEAL system from the three source papers.",
        uncertainty_score=0.5,
        coherence_score=0.7,
        logic_score=0.6,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_001", relationship_type="contextual")
        ]
    )


@pytest.fixture
def sample_step_3() -> ReasoningStep:
    """Create a sample low-quality reasoning step with potential issues."""
    return ReasoningStep(
        step_id="step_003",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="design_architecture",
        content="I'm not sure about this, but I'll try to design the architecture based on what I remember. Maybe we should use some kind of graph or something? Could be good.",
        uncertainty_score=0.8,
        coherence_score=0.4,
        logic_score=0.3,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="evidence_for")
        ]
    )


@pytest.fixture
def sample_dig_with_nodes() -> ReasoningDIG:
    """Create a DIG with multiple nodes and causal links."""
    dig = ReasoningDIG(
        session_id="test_session_001",
        conversation_id="test_conversation_001"
    )
    
    # Create nodes
    node1 = ReasoningNode(
        step_id="step_001",
        agent_id="planner",
        agent_role="planner",
        intent_hash="plan_initial",
        content="Initial project plan"
    )
    
    node2 = ReasoningNode(
        step_id="step_002",
        agent_id="researcher",
        agent_role="researcher",
        intent_hash="research_requirements",
        content="Research requirements"
    )
    
    node3 = ReasoningNode(
        step_id="step_003",
        agent_id="thinker",
        agent_role="thinker",
        intent_hash="design_architecture",
        content="Design architecture"
    )
    
    node4 = ReasoningNode(
        step_id="step_004",
        agent_id="thinker",
        agent_role="thinker",
        intent_hash="design_detailed",
        content="Design detailed components"
    )
    
    # Add nodes
    dig.add_node(node1)
    dig.add_node(node2)
    dig.add_node(node3)
    dig.add_node(node4)
    
    # Add causal links
    link1 = CausalLink(
        source_step_id="step_001",
        target_step_id="step_002",
        relationship_type=CausalRelationship.CONTEXTUAL,
        confidence=0.9
    )
    
    link2 = CausalLink(
        source_step_id="step_002",
        target_step_id="step_003",
        relationship_type=CausalRelationship.EVIDENCE_FOR,
        confidence=0.85
    )
    
    link3 = CausalLink(
        source_step_id="step_003",
        target_step_id="step_004",
        relationship_type=CausalRelationship.FOLLOWS_FROM,
        confidence=0.7
    )
    
    dig.add_link(link1)
    dig.add_link(link2)
    dig.add_link(link3)
    
    return dig


# ============================================================================
# TEST BASIC INITIALIZATION
# ============================================================================

def test_regulator_initialization(sample_regulator):
    """Test that the regulator initializes correctly."""
    assert sample_regulator is not None
    assert isinstance(sample_regulator.config, RegulatorConfig)
    assert isinstance(sample_regulator.noise_senser, NoiseSenser)
    assert sample_regulator.reasoning_dig is None
    assert sample_regulator.session_state is None


def test_regulator_with_custom_config():
    """Test initialization with custom configuration."""
    config = RegulatorConfig(
        drift_threshold=0.5,
        uncertainty_threshold=0.6,
        quality_threshold=0.6
    )
    regulator = ReasoningRegulator(config=config)
    
    assert regulator.config.drift_threshold == 0.5
    assert regulator.config.uncertainty_threshold == 0.6
    assert regulator.config.quality_threshold == 0.6


# ============================================================================
# TEST SESSION INITIALIZATION
# ============================================================================

def test_initialize_session(sample_regulator, sample_session):
    """Test that a regulation session initializes correctly."""
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    assert sample_regulator.session_state is not None
    assert sample_regulator.reasoning_dig is not None
    assert sample_regulator.session_state.session_id == "test_session"
    assert sample_regulator.session_state.conversation_id == "test_conversation"
    assert sample_regulator.previous_steps == []
    assert sample_regulator.current_step is None


# ============================================================================
# TEST ANALYSIS AND REGULATION
# ============================================================================

def test_analyze_and_regulate_first_step(sample_regulator, sample_step_1):
    """Test analysis of the first reasoning step (no drift detection)."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Analyze first step
    result = sample_regulator.analyze_and_regulate(sample_step_1)
    
    # Verify result type
    assert isinstance(result, RegulationResult)
    assert result.session_id == "test_session"
    assert result.current_step_id == "step_001"
    assert result.issues_detected == ['first_step']
    assert result.root_causes == []
    assert result.healing_recommendations == [{'action': 'monitor_next_step', 'target': 'current_step', 'confidence': 0.7, 'description': 'General review of current reasoning step recommended', 'priority': 'high'}]
    
    # Verify quality metrics
    metrics = result.quality_metrics
    assert metrics['session_quality'] >= 0.5  # First step should be high quality
    assert metrics['healing_urgency'] <= 0.2


def test_analyze_and_regulate_with_drift(sample_regulator, sample_step_1, sample_step_2, sample_step_3):
    """Test analysis when semantic drift is detected."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Process first step (high quality)
    result1 = sample_regulator.analyze_and_regulate(sample_step_1)
    sample_regulator.update_history(sample_step_1)
    
    # Process second step (moderate quality, should be okay)
    result2 = sample_regulator.analyze_and_regulate(sample_step_2)
    sample_regulator.update_history(sample_step_2)
    
    # Process third step (low quality, should trigger drift detection)
    result3 = sample_regulator.analyze_and_regulate(sample_step_3)
    sample_regulator.update_history(sample_step_3)
    
    # Verify drift was detected
    assert result3.issues_detected is not None
    assert len(result3.issues_detected) > 0
    assert any('drift' in issue for issue in result3.issues_detected)
    
    # Verify root causes were identified
    assert result3.root_causes is not None
    assert len(result3.root_causes) > 0
    
    # Verify recommendations were generated
    assert result3.healing_recommendations is not None
    assert len(result3.healing_recommendations) > 0
    
    # Check that the current step is flagged
    assert result3.current_step_id == "step_003"


def test_analyze_and_regulate_with_dig_root_cause(sample_regulator, sample_dig_with_nodes):
    """Test root cause identification using the Dynamic Interaction Graph."""
    # Initialize regulator with the pre-built DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a current step that might have issues
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="validator_agent",
        agent_role=AgentRole.VALIDATOR,
        intent_hash="validate_architecture",
        content="I will validate the architecture design based on the requirements.",
        uncertainty_score=0.7,
        coherence_score=0.5,
        logic_score=0.4,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="evidence_for")
        ]
    )
    
    # Analyze the current step
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Verify root causes were identified using DIG
    assert result.root_causes is not None
    assert len(result.root_causes) > 0
    
    # Check that root causes are from earlier steps
    root_step_ids = [rc['step_id'] for rc in result.root_causes]
    assert 'step_001' in root_step_ids or 'step_002' in root_step_ids or 'step_003' in root_step_ids or 'step_004' in root_step_ids
    
    # Verify quality metrics
    metrics = result.quality_metrics
    assert metrics['session_quality'] <= 0.7  # Should be lower due to issues
    assert metrics['healing_urgency'] >= 0.3


# ============================================================================
# TEST ROOT CAUSE IDENTIFICATION
# ============================================================================

def test_identify_root_causes_no_issues(sample_regulator, sample_step_1):
    """Test root cause identification when no issues are detected."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Analyze first step
    result = sample_regulator.analyze_and_regulate(sample_step_1)
    
    # Should return empty list when no issues
    root_causes = sample_regulator._identify_root_causes()
    assert root_causes == []


def test_identify_root_causes_with_issues(sample_regulator, sample_dig_with_nodes):
    """Test root cause identification when issues are detected."""
    # Initialize regulator with DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a problematic current step
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="I'm really confused now. This doesn't make any sense. I think we should just give up.",
        uncertainty_score=0.9,
        coherence_score=0.2,
        logic_score=0.1,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step (will set issues)
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Identify root causes
    root_causes = sample_regulator._identify_root_causes()
    
    # Should have identified some root causes
    assert root_causes is not None
    assert len(root_causes) > 0
    
    # Check that root causes have proper structure
    for cause in root_causes:
        assert 'step_id' in cause
        assert 'quality_score' in cause
        assert 'uncertainty_score' in cause
        assert 'importance_score' in cause
        assert 'composite_score' in cause


def test_calculate_root_cause_score():
    """Test the root cause scoring algorithm."""
    regulator = create_reasoning_regulator()
    
    # Test various scenarios
    score1 = regulator._calculate_root_cause_score(
        quality_score=0.2,  # Low quality
        uncertainty_score=0.8,  # High uncertainty
        importance_score=0.6  # High importance
    )
    
    score2 = regulator._calculate_root_cause_score(
        quality_score=0.8,  # High quality
        uncertainty_score=0.2,  # Low uncertainty
        importance_score=0.3  # Low importance
    )
    
    # First score should be higher (worse)
    assert score1 > score2
    assert score1 > 0.5
    assert score2 < 0.5


# ============================================================================
# TEST RECOMMENDATION GENERATION
# ============================================================================

def test_generate_recommendations_with_root_causes(sample_regulator, sample_dig_with_nodes):
    """Test recommendation generation based on root causes."""
    # Initialize regulator with DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a problematic current step
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="This is very confusing. I'm not sure what to do next.",
        uncertainty_score=0.85,
        coherence_score=0.3,
        logic_score=0.2,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Generate recommendations
    recommendations = sample_regulator._generate_recommendations(result.root_causes, result._analysis_results.get(current_step.step_id, {}))
    
    # Should have recommendations
    assert recommendations is not None
    assert len(recommendations) > 0
    
    # Check recommendation structure
    for rec in recommendations:
        assert 'action' in rec
        assert 'target' in rec
        assert 'confidence' in rec
        assert 'description' in rec
        assert 'priority' in rec


def test_generate_recommendations_no_root_causes(sample_regulator, sample_step_2):
    """Test recommendation generation when no root causes but drift detected."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Process first step to have history
    sample_regulator.analyze_and_regulate(sample_step_2)
    sample_regulator.update_history(sample_step_2)
    
    # Create a step with drift but no clear root cause
    drift_step = ReasoningStep(
        step_id="step_003",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="drift_step",
        content="This is completely different from what we were talking about. Let's discuss something else entirely.",
        uncertainty_score=0.4,
        coherence_score=0.1,
        logic_score=0.2,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(drift_step)
    
    # Generate recommendations
    recommendations = sample_regulator._generate_recommendations(result.root_causes, result._analysis_results.get(drift_step.step_id, {}))
    
    # Should have general recommendation
    assert recommendations is not None
    assert len(recommendations) == 1
    assert recommendations[0]['action'] == 'general_review'


# ============================================================================
# TEST QUALITY METRICS CALCULATION
# ============================================================================

def test_calculate_quality_metrics(sample_regulator, sample_dig_with_nodes):
    """Test quality metrics calculation."""
    # Initialize regulator with DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a problematic current step
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="This is very confusing. I'm not sure what to do next.",
        uncertainty_score=0.85,
        coherence_score=0.3,
        logic_score=0.2,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Calculate quality metrics
    metrics = sample_regulator._calculate_quality_metrics(result._analysis_results.get(current_step.step_id, {}), result.root_causes)
    
    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert 'session_quality' in metrics
    assert 'drift_prevalence' in metrics
    assert 'uncertainty_level' in metrics
    assert 'root_cause_severity' in metrics
    assert 'healing_urgency' in metrics
    
    # Values should be between 0 and 1
    for value in metrics.values():
        assert 0 <= value <= 1


def test_calculate_confidence(sample_regulator, sample_dig_with_nodes):
    """Test confidence calculation."""
    # Initialize regulator with DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a problematic current step
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="This is very confusing. I'm not sure what to do next.",
        uncertainty_score=0.85,
        coherence_score=0.3,
        logic_score=0.2,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Calculate confidence
    confidence = sample_regulator._calculate_confidence(result._analysis_results.get(current_step.step_id, {}), result.root_causes)
    
    # Confidence should be between 0.3 and 1.0 (minimum is 0.3)
    assert 0.3 <= confidence <= 1.0


# ============================================================================
# TEST HISTORY UPDATING
# ============================================================================

def test_update_history(sample_regulator, sample_step_1, sample_step_2):
    """Test updating the regulator's history."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Update history with first step
    sample_regulator.update_history(sample_step_1)
    assert len(sample_regulator.previous_steps) == 1
    assert sample_regulator.previous_steps[0].step_id == "step_001"
    
    # Update history with second step
    sample_regulator.update_history(sample_step_2)
    assert len(sample_regulator.previous_steps) == 2
    assert sample_regulator.previous_steps[1].step_id == "step_002"
    
    # Ensure history doesn't grow indefinitely
    for i in range(20):
        new_step = ReasoningStep(
            step_id=f"step_{i+3}",
            agent_id=f"agent_{i%3}",
            agent_role=AgentRole.THINKER,
            intent_hash=f"intent_{i}",
            content=f"Step {i+3} content",
            uncertainty_score=0.5,
            coherence_score=0.5,
            logic_score=0.5
        )
        sample_regulator.update_history(new_step)
    
    assert len(sample_regulator.previous_steps) == 20  # Should be capped at 20


# ============================================================================
# TEST VISUALIZATION METHODS
# ============================================================================

def test_visualize_regulation_text(sample_regulator, sample_step_1, sample_step_2, sample_step_3):
    """Test text visualization of regulation results."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Process steps
    sample_regulator.analyze_and_regulate(sample_step_1)
    sample_regulator.update_history(sample_step_1)
    sample_regulator.analyze_and_regulate(sample_step_2)
    sample_regulator.update_history(sample_step_2)
    result = sample_regulator.analyze_and_regulate(sample_step_3)
    sample_regulator.update_history(sample_step_3)
    
    # Get visualization
    visualization = sample_regulator.visualize_regulation(result, output_format='text')
    
    # Should be a non-empty string
    assert isinstance(visualization, str)
    assert len(visualization) > 0
    assert "FLOW-HEAL REGULATION REPORT" in visualization


def test_visualize_regulation_markdown(sample_regulator, sample_step_1, sample_step_2, sample_step_3):
    """Test markdown visualization of regulation results."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Process steps
    sample_regulator.analyze_and_regulate(sample_step_1)
    sample_regulator.update_history(sample_step_1)
    sample_regulator.analyze_and_regulate(sample_step_2)
    sample_regulator.update_history(sample_step_2)
    result = sample_regulator.analyze_and_regulate(sample_step_3)
    sample_regulator.update_history(sample_step_3)
    
    # Get visualization
    visualization = sample_regulator.visualize_regulation(result, output_format='markdown')
    
    # Should be a non-empty string with markdown formatting
    assert isinstance(visualization, str)
    assert len(visualization) > 0
    assert visualization.startswith("# FLOW-HEAL Regulation Report")


def test_visualize_regulation_json(sample_regulator, sample_step_1, sample_step_2, sample_step_3):
    """Test JSON visualization of regulation results."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Process steps
    sample_regulator.analyze_and_regulate(sample_step_1)
    sample_regulator.update_history(sample_step_1)
    sample_regulator.analyze_and_regulate(sample_step_2)
    sample_regulator.update_history(sample_step_2)
    result = sample_regulator.analyze_and_regulate(sample_step_3)
    sample_regulator.update_history(sample_step_3)
    
    # Get visualization
    visualization = sample_regulator.visualize_regulation(result, output_format='json')
    
    # Should be a valid JSON string
    assert isinstance(visualization, str)
    import json
    data = json.loads(visualization)
    assert 'session_id' in data
    assert 'current_step_id' in data
    assert 'issues_detected' in data


# ============================================================================
# TEST INTEGRATION WITH PAYLOAD AND GRAPH
# ============================================================================

def test_integration_with_payload_and_graph(sample_regulator):
    """Test that the regulator properly integrates with payload and graph modules."""
    # Initialize session
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Create a step with payload data that includes node data
    step = ReasoningStep(
        step_id="step_001",
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for the FLOW-HEAL system.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8,
        payload={
            "node_data": {
                "custom_field": "custom_value",
                "importance_score": 0.8
            }
        },
        causal_dependencies=[
            CausalDependency(parent_step_id="initial_prompt", relationship_type="follows_from")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(step)
    
    # Check that the node was created with payload data
    assert sample_regulator.reasoning_dig is not None
    node = sample_regulator.reasoning_dig.get_node("step_001")
    assert node is not None
    assert 'custom_field' in node.metadata or 'importance_score' in node.metadata
    
    # Verify that the step was added to the session
    assert sample_regulator.session_state is not None
    session_step = sample_regulator.session_state.get_step("step_001")
    assert session_step is not None
    assert session_step.step_id == "step_001"


# ============================================================================
# TEST REGULATOR CONFIGURATION
# ============================================================================

def test_regulator_configuration_adjustment():
    """Test that configuration parameters affect regulator behavior."""
    # Create a regulator with conservative thresholds
    config = RegulatorConfig(
        drift_threshold=0.9,  # High threshold - less sensitive to drift
        uncertainty_threshold=0.9,  # High threshold - less sensitive to uncertainty
        quality_threshold=0.3  # Low threshold - more tolerant of low quality
    )
    regulator = create_reasoning_regulator(config=config)
    
    # Create a step that would normally trigger issues but with high thresholds might not
    problematic_step = ReasoningStep(
        step_id="step_001",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="This is terrible. I have no idea what I'm doing. This makes no sense at all.",
        uncertainty_score=0.95,  # Very high uncertainty
        coherence_score=0.1,    # Very low coherence
        logic_score=0.05        # Very low logic
    )
    
    # Initialize session
    regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Analyze the step
    result = regulator.analyze_and_regulate(problematic_step)
    
    # With high thresholds, might not trigger issues
    # But the step is so bad that it should still trigger something
    assert result.issues_detected is not None
    assert len(result.issues_detected) > 0


# ============================================================================
# TEST VISUALIZE DIG WITH ANALYSIS
# ============================================================================

def test_visualize_dig_with_analysis(sample_regulator, sample_dig_with_nodes):
    """Test visualization of DIG with analysis highlights."""
    # Initialize regulator with DIG
    sample_regulator.initialize_session(
        session_id="test_session",
        conversation_id="test_conversation",
        reasoning_dig=sample_dig_with_nodes
    )
    
    # Create a current step
    current_step = ReasoningStep(
        step_id="step_005",
        agent_id="validator_agent",
        agent_role=AgentRole.VALIDATOR,
        intent_hash="validate_architecture",
        content="I will validate the architecture design based on the requirements.",
        uncertainty_score=0.7,
        coherence_score=0.5,
        logic_score=0.4,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="evidence_for")
        ]
    )
    
    # Analyze the step
    result = sample_regulator.analyze_and_regulate(current_step)
    
    # Visualize DIG
    dig_viz = sample_regulator.visualize_dig_with_analysis(result, highlight_root_causes=True)
    
    # Should be a non-empty string
    assert isinstance(dig_viz, str)
    assert len(dig_viz) > 0
    assert "digraph" in dig_viz  # Graphviz DOT format


# ============================================================================
# TEST FACTORY FUNCTION
# ============================================================================

def test_create_reasoning_regulator_factory():
    """Test the factory function."""
    regulator = create_reasoning_regulator()
    assert isinstance(regulator, ReasoningRegulator)
    
    # Test with custom noise senser
    custom_senser = NoiseSenser(similarity_threshold=0.5)
    regulator2 = create_reasoning_regulator(noise_senser=custom_senser)
    assert regulator2.noise_senser.similarity_threshold == 0.5
    
    # Test with custom config
    config = RegulatorConfig(drift_threshold=0.5)
    regulator3 = create_reasoning_regulator(config=config)
    assert regulator3.config.drift_threshold == 0.5


# ============================================================================
# TEST REGULATION RESULT SERIALIZATION
# ============================================================================

def test_regulation_result_to_dict():
    """Test converting RegulationResult to dictionary."""
    result = RegulationResult(
        session_id="test_session",
        current_step_id="step_001",
        issues_detected=[{"issue": "test_issue"}],
        root_causes=[{"step_id": "step_001", "quality_score": 0.5}],
        healing_recommendations=[{"action": "test_action", "target": "step_001"}],
        quality_metrics={"session_quality": 0.7},
        confidence=0.8
    )
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert result_dict["session_id"] == "test_session"
    assert result_dict["current_step_id"] == "step_001"
    assert result_dict["issues_detected"] == [{"issue": "test_issue"}]
    assert result_dict["root_causes"] == [{"step_id": "step_001", "quality_score": 0.5}]
    assert result_dict["healing_recommendations"] == [{"action": "test_action", "target": "step_001"}]
    assert result_dict["quality_metrics"] == {"session_quality": 0.7}
    assert result_dict["confidence"] == 0.8
    assert "timestamp" in result_dict


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_regulator_without_session():
    """Test regulator behavior without initializing a session."""
    regulator = create_reasoning_regulator()
    
    # Should not crash when calling methods without session
    with pytest.raises(AssertionError):
        regulator.analyze_and_regulate(ReasoningStep(
            step_id="test",
            agent_id="test",
            agent_role=AgentRole.THINKER,
            intent_hash="test",
            content="test",
            uncertainty_score=0.5,
            coherence_score=0.5,
            logic_score=0.5
        ))
    
    # Should initialize session properly when called
    regulator.initialize_session("test_session", "test_conversation")
    result = regulator.analyze_and_regulate(ReasoningStep(
        step_id="test",
        agent_id="test",
        agent_role=AgentRole.THINKER,
        intent_hash="test",
        content="test",
        uncertainty_score=0.5,
        coherence_score=0.5,
        logic_score=0.5
    ))
    assert result is not None


def test_regulator_with_empty_history():
    """Test regulator behavior with empty history."""
    regulator = create_reasoning_regulator()
    regulator.initialize_session("test_session", "test_conversation")
    
    step = ReasoningStep(
        step_id="step_001",
        agent_id="test_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="test",
        content="First step",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8
    )
    
    result = regulator.analyze_and_regulate(step)
    assert result.issues_detected == ['first_step']
    assert result.healing_recommendations[0]['action'] == 'monitor_next_step'


def test_regulator_with_large_history():
    """Test regulator handles history window properly."""
    regulator = create_reasoning_regulator()
    regulator.initialize_session("test_session", "test_conversation")
    
    # Add 25 steps
    for i in range(25):
        step = ReasoningStep(
            step_id=f"step_{i+1}",
            agent_id="test_agent",
            agent_role=AgentRole.THINKER,
            intent_hash=f"intent_{i}",
            content=f"Step {i+1} content",
            uncertainty_score=0.5,
            coherence_score=0.5,
            logic_score=0.5
        )
        regulator.analyze_and_regulate(step)
        regulator.update_history(step)
    
    # History should be capped at 20
    assert len(regulator.previous_steps) == 20


def test_regulator_with_missing_dig():
    """Test regulator handles missing DIG gracefully."""
    regulator = create_reasoning_regulator()
    regulator.initialize_session("test_session", "test_conversation")
    
    # Create a step with issues but without DIG (should still work)
    step = ReasoningStep(
        step_id="step_001",
        agent_id="test_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="test",
        content="This is problematic",
        uncertainty_score=0.9,
        coherence_score=0.1,
        logic_score=0.1
    )
    
    result = regulator.analyze_and_regulate(step)
    assert result is not None
    assert len(result.root_causes) == 0  # No DIG, so no root causes from DIG
    assert len(result.healing_recommendations) > 0  # Still has recommendations


def test_regulator_with_invalid_config():
    """Test regulator handles invalid configuration gracefully."""
    # Create regulator with invalid config values
    config = RegulatorConfig(
        drift_threshold=-0.1,  # Invalid (negative)
        uncertainty_threshold=1.1,  # Invalid (greater than 1)
        quality_threshold=0.5
    )
    
    regulator = ReasoningRegulator(config=config)
    
    # Should still work with normalized values or defaults
    regulator.initialize_session("test_session", "test_conversation")
    step = ReasoningStep(
        step_id="step_001",
        agent_id="test_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="test",
        content="Test content",
        uncertainty_score=0.5,
        coherence_score=0.5,
        logic_score=0.5
    )
    
    result = regulator.analyze_and_regulate(step)
    assert result is not None