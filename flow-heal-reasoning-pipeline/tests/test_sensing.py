"""
Test module for the Semantic Noise Senser (NoiseSenser) class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import List, Dict, Any
from datetime import datetime
import numpy as np

# Import the NoiseSenser and related classes
from payload import ReasoningStep, AgentRole, StepStatus
from sensing import NoiseSenser, create_noise_senser


def test_basic_senser_creation():
    """Test basic creation of NoiseSenser."""
    senser = NoiseSenser()
    assert senser is not None
    print("✓ NoiseSenser created successfully")


def test_semantic_similarity():
    """Test semantic similarity calculation."""
    senser = NoiseSenser()
    
    # Test with similar texts
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleepy dog"
    similarity = senser.calculate_semantic_similarity(text1, text2)
    assert similarity > 0.5, f"Expected high similarity, got {similarity}"
    print(f"✓ Semantic similarity works (score: {similarity:.3f})")


def test_semantic_drift_detection():
    """Test drift detection between two steps."""
    senser = NoiseSenser(similarity_threshold=0.6)
    
    # Create two reasoning steps with similar content
    step1 = ReasoningStep(
        step_id="step_1",
        agent_id="agent_a",
        agent_role=AgentRole.THINKER,
        intent_hash="intent_1",
        content="I am analyzing the problem and forming a hypothesis."
    )
    
    step2 = ReasoningStep(
        step_id="step_2",
        agent_id="agent_b",
        agent_role=AgentRole.THINKER,
        intent_hash="intent_2",
        content="I will now calculate the trajectory of the projectile using physics equations."
    )
    
    # These steps should have some similarity but maybe not too high
    drift_result = senser.detect_semantic_drift(
        current_step=step2,
        previous_step=step1
    )
    
    print("✓ Drift detection works")
    print(f"  Drift detected: {drift_result['drift_detected']}")
    print(f"  Similarity score: {drift_result['similarity_scores'].get('previous', 0):.3f}")
    print(f"  Drift confidence: {drift_result['confidence']:.3f}")
    
    # Test with very different steps (should detect drift)
    step3 = ReasoningStep(
        step_id="step_3",
        agent_id="agent_c",
        agent_role=AgentRole.THINKER,
        intent_hash="intent_3",
        content="The moon is made of cheese and orbits the Earth."
    )
    
    drift_result2 = senser.detect_semantic_drift(
        current_step=step3,
        previous_step=step2
    )
    assert drift_result2['drift_detected'], "Expected drift to be detected"
    print(f"✓ Drift correctly detected between unrelated steps")


def test_uncertainty_quantification():
    """Test uncertainty quantification."""
    senser = NoiseSenser()
    
    # Create a step with uncertain language
    step = ReasoningStep(
        step_id="step_1",
        agent_id="agent_a",
        agent_role=AgentRole.THINKER,
        intent_hash="intent_1",
        content="I think this might work, but I'm not entirely sure. Perhaps we should consider other options."
    )
    
    uncertainty = senser.quantify_uncertainty(step)
    print("✓ Uncertainty quantification works")
    print(f"  Uncertainty score: {uncertainty:.3f}")
    assert uncertainty > 0.5, f"Expected high uncertainty, got {uncertainty}"
    
    # Create a confident step
    step2 = ReasoningStep(
        step_id="step_2",
        agent_id="agent_b",
        agent_role=AgentRole.THINKER,
        intent_hash="intent_2",
        content="The solution is X = 5. This follows directly from the equations."
    )
    
    uncertainty2 = senser.quantify_uncertainty(step2)
    print(f"  Confident step uncertainty: {uncertainty2:.3f}")
    assert uncertainty2 < 0.3, f"Expected low uncertainty, got {uncertainty2:.3f}"


def test_full_analysis():
    """Test full step analysis."""
    senser = NoiseSenser()
    
    # Create a sequence of steps
    steps = [
        ReasoningStep(
            step_id="step_1",
            agent_id="planner",
            agent_role=AgentRole.PLANNER,
            intent_hash="plan_initial",
            content="I will create an initial project plan for FLOW-HEAL."
        ),
        ReasoningStep(
            step_id="step_2",
            agent_id="researcher",
            agent_role=AgentRole.RESEARCHER,
            intent_hash="research_requirements",
            content="I will research the requirements for the FLOW-HEAL system from the three source papers."
        ),
        ReasoningStep(
            step_id="step_3",
            agent_id="thinker",
            agent_role=AgentRole.THINKER,
            intent_hash="design_architecture",
            content="Based on the research, I will design the overall architecture for FLOW-HEAL."
        )
    ]
    
    # Analyze all steps
    analyses = senser.batch_analyze(steps)
    
    print("✓ Batch analysis works")
    print(f"  Analyzed {len(analyses)} steps")
    for i, analysis in enumerate(analyses):
        print(f"    Step {i+1}: Drift={analysis['drift_analysis']['drift_detected']}, "
              f"Uncertainty={analysis['uncertainty_score']:.2f}")


def test_dig_integration():
    """Test integration with ReasoningDIG."""
    from graph import ReasoningDIG, ReasoningNode, CausalLink, CausalRelationship
    
    # Create a simple DIG
    dig = ReasoningDIG(
        session_id="test_session",
        conversation_id="test_conversation"
    )
    
    # Add nodes
    node1 = ReasoningNode(
        step_id="step_1",
        agent_id="planner",
        agent_role="planner",
        intent_hash="plan_initial",
        content="Initial plan"
    )
    dig.add_node(node1)
    
    node2 = ReasoningNode(
        step_id="step_2",
        agent_id="researcher",
        agent_role="researcher",
        intent_hash="research_requirements",
        content="Research requirements"
    )
    dig.add_node(node2)
    
    # Add link
    link = CausalLink(
        source_step_id="step_1",
        target_step_id="step_2",
        relationship_type=CausalRelationship.CONTEXTUAL,
        confidence=0.9
    )
    dig.add_link(link)
    
    # Create senser and analyze with DIG
    senser = NoiseSenser()
    step1 = ReasoningStep(
        step_id="step_1",
        agent_id="planner",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for FLOW-HEAL."
    )
    
    step2 = ReasoningStep(
        step_id="step_2",
        agent_id="researcher",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will research the requirements for the FLOW-HEAL system from the three source papers."
    )
    
    # Analyze step 2 with DIG context
    analysis = senser.analyze_step(
        current_step=step2,
        previous_steps=[step1],
        dig=dig
    )
    
    print("✓ DIG integration works")
    print(f"  Step quality from DIG: {analysis['dig_context'].get('quality_score', 'N/A'):.3f}")
    print(f"  Node importance metrics: {analysis['dig_context'].get('importance', 'N/A')}")
    
    # Update step with analysis
    updated_step = senser.update_step_with_analysis(step2, analysis, dig)
    print(f"  Updated step uncertainty: {updated_step.uncertainty_score:.3f}")


def test_visualization():
    """Test visualization of drift metrics."""
    senser = NoiseSenser()
    
    # Create sample analysis results
    analyses = []
    for i in range(3):
        analysis = {
            'step_id': f'step_{i+1}',
            'agent_id': f'agent_{i}',
            'agent_role': 'thinker',
            'drift_analysis': {
                'drift_detected': i % 2 == 0,  # Every other step has drift
                'confidence': 0.3 + (i * 0.2),
                'similarity_scores': {'previous': 0.8 - (i * 0.1)}
            },
            'uncertainty_score': 0.4 + (i * 0.1),
            'flags': ['high_uncertainty'] if i == 2 else [],
            'recommendations': ['recheck_premises'] if i % 2 == 0 else []
        }
        analyses.append(analysis)
    
    # Generate markdown report
    report = senser.visualize_drift_metrics(analyses, output_format='markdown')
    print("✓ Visualization works")
    print("\n--- Sample Report ---")
    print(report[:500] + "..." if len(report) > 500 else report)


def test_create_with_custom_model():
    """Test creating senser with custom embedding model."""
    if EMBEDDING_MODEL_AVAILABLE:
        try:
            senser = create_noise_senser(model_name='paraphrase-multilingual-MiniLM-L12-v2')
            assert senser.embedding_model is not None
            print("✓ Created senser with custom model")
        except Exception as e:
            print(f"⚠ Could not load custom model: {e}")
    else:
        print("⚠ SentenceTransformers not available, skipping custom model test")


def main():
    """Run all tests."""
    print("Testing Semantic Noise Senser Module")
    print("=" * 60)
    
    test_basic_senser_creation()
    test_semantic_similarity()
    test_semantic_drift_detection()
    test_uncertainty_quantification()
    test_full_analysis()
    test_dig_integration()
    test_visualization()
    test_create_with_custom_model()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()