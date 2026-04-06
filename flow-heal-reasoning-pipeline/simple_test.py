"""
Simple test to verify FLOW-HEAL core functionality.
This demonstrates the basic integration of payload, graph, and sensing modules.
"""

from src.payload import ReasoningStep, ReasoningSession, AgentRole, StepStatus, CausalDependency
from src.graph import ReasoningDIG, ReasoningNode, CausalLink, CausalRelationship
from src.sensing import NoiseSenser, create_noise_senser
from datetime import datetime

print("=" * 70)
print("SIMPLE FLOW-HEAL CORE FUNCTIONALITY TEST")
print("=" * 70)

# Test 1: Create a reasoning step
print("\n1. Creating ReasoningStep...")
try:
    step1 = ReasoningStep(
        step_id="step_001",
        agent_id="planner",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for FLOW-HEAL.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8,
        causal_dependencies=[
            CausalDependency(parent_step_id="user_prompt", relationship_type="follows_from")
        ]
    )
    print(f"   ✓ Created: {step1}")
    print(f"   Quality: {step1.calculate_overall_quality():.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Create a reasoning session
print("\n2. Creating ReasoningSession...")
try:
    session = ReasoningSession(
        session_id="test_session_001",
        conversation_id="test_conv_001"
    )
    session.add_step(step1)
    print(f"   ✓ Created session with {len(session.steps)} steps")
    print(f"   DIG: {session.dig}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Create a NoiseSenser
print("\n3. Creating NoiseSenser...")
try:
    senser = create_noise_senser(similarity_threshold=0.65, uncertainty_threshold=0.7)
    print(f"   ✓ Created NoiseSenser")
    print(f"   Similarity threshold: {senser.similarity_threshold}")
    print(f"   Uncertainty threshold: {senser.uncertainty_threshold}")
    
    # Test similarity
    text1 = "The capital of France is Paris."
    text2 = "Paris is the capital of France."
    similarity = senser.calculate_semantic_similarity(text1, text2)
    print(f"   Similarity: {similarity:.3f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Build a simple DIG
print("\n4. Building a simple DIG...")
try:
    from uuid import uuid4
    
    dig = ReasoningDIG(
        session_id="session_001",
        conversation_id="conv_001"
    )
    
    # Create nodes
    node1 = ReasoningNode(
        step_id="step_001",
        agent_id="planner",
        agent_role="planner",
        intent_hash="plan_initial",
        content="I will create an initial project plan for FLOW-HEAL."
    )
    
    node2 = ReasoningNode(
        step_id="step_002",
        agent_id="researcher",
        agent_role="researcher",
        intent_hash="research_requirements",
        content="I will research the requirements for FLOW-HEAL."
    )
    
    dig.add_node(node1)
    dig.add_node(node2)
    
    # Create link
    link = CausalLink(
        source_step_id="step_001",
        target_step_id="step_002",
        relationship_type=CausalRelationship.CONTEXTUAL,
        confidence=0.9
    )
    dig.add_link(link)
    
    print(f"   ✓ DIG created with {len(dig.nodes)} nodes and {len(dig.links)} links")
    print(f"   Node 1 quality: {node1.quality_score:.2f}")
    print(f"   Root causes for step_002: {dig.identify_root_causes('step_002')}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Batch analysis
print("\n5. Running batch analysis...")
try:
    steps = [step1]
    analyses = senser.batch_analyze(steps, dig)
    print(f"   ✓ Analyzed {len(analyses)} steps")
    for i, analysis in enumerate(analyses):
        print(f"   Step {i+1}: Drift={analysis['drift_analysis']['drift_detected']}, "
              f"Uncertainty={analysis['uncertainty_score']:.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
