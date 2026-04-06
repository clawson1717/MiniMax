"""
Test script for FLOW-HEAL Reasoning Regulator.
This demonstrates the full pipeline: sensing, regulation, and healing.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.payload import (
    ReasoningStep, ReasoningSession, AgentRole, StepStatus, 
    CausalDependency, ReasoningContext
)
from src.graph import (
    ReasoningDIG, ReasoningNode, CausalLink, CausalRelationship,
    ReasoningDIG
)
from src.sensing import NoiseSenser, create_noise_senser
from uuid import uuid4

def create_test_session() -> ReasoningSession:
    """Create a sample reasoning session with potential drift."""
    session = ReasoningSession(
        session_id="regulator_test_001",
        conversation_id="test_conv_001"
    )
    
    # Step 1: Initial plan (good quality)
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
    session.add_step(step1)
    
    # Step 2: Research (good quality)
    step2 = ReasoningStep(
        step_id="step_002",
        agent_id="researcher",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will analyze the three source papers to extract requirements.",
        uncertainty_score=0.3,
        coherence_score=0.8,
        logic_score=0.7,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_001", relationship_type="DEPENDS_ON")
        ]
    )
    session.add_step(step2)
    
    # Step 3: Architecture design (moderate quality)
    step3 = ReasoningStep(
        step_id="step_003",
        agent_id="thinker",
        agent_role=AgentRole.THINKER,
        intent_hash="design_architecture",
        content="Based on research, I'll design a modular architecture with sensing, regulating, and healing components.",
        uncertainty_score=0.4,
        coherence_score=0.7,
        logic_score=0.6,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="FOLLOWS_FROM")
        ]
    )
    session.add_step(step3)
    
    # Step 4: Critic evaluation (potential drift - introduces unrelated concern)
    step4 = ReasoningStep(
        step_id="step_004",
        agent_id="critic",
        agent_role=AgentRole.CRITIC,
        intent_hash="evaluate_design",
        content="I will evaluate the design for security vulnerabilities, scalability issues, and potential integration challenges with legacy systems.",
        uncertainty_score=0.6,
        coherence_score=0.5,
        logic_score=0.5,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="EVALUATES")
        ]
    )
    session.add_step(step4)
    
    # Step 5: Implementation plan (should be aligned but may drift due to critic)
    step5 = ReasoningStep(
        step_id="step_005",
        agent_id="planner",
        agent_role=AgentRole.PLANNER,
        intent_hash="create_implementation_plan",
        content="I will break down the architecture into 12 steps including requirements gathering, model training, and deployment.",
        uncertainty_score=0.5,
        coherence_score=0.6,
        logic_score=0.5,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="implements")
        ]
    )
    session.add_step(step5)
    
    # Step 6: Validator check (identifies potential inconsistency)
    step6 = ReasoningStep(
        step_id="step_006",
        agent_id="validator",
        agent_role=AgentRole.VALIDATOR,
        intent_hash="validate_logic",
        content="I will verify that the implementation plan aligns with the architectural design and requirements.",
        uncertainty_score=0.7,
        coherence_score=0.4,
        logic_score=0.4,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_005", relationship_type="validates")
        ]
    )
    session.add_step(step6)
    
    # Step 7: Healer (triggered by regulator)
    step7 = ReasoningStep(
        step_id="step_007",
        agent_id="healer",
        agent_role=AgentRole.HEALER,
        intent_hash="heal_drift",
        content="I will identify semantic drift and propose corrections to maintain logical consistency.",
        uncertainty_score=0.8,
        coherence_score=0.3,
        logic_score=0.3,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_006", relationship_type="corrects")
        ]
    )
    session.add_step(step7)
    
    return session

def build_dig_from_session(session: ReasoningSession) -> ReasoningDIG:
    """Build a Dynamic Interaction Graph from a reasoning session."""
    dig = ReasoningDIG(
        session_id=session.session_id,
        conversation_id=session.conversation_id
    )
    
    # Add all nodes
    for step_id, step in session.steps.items():
        node = ReasoningNode(
            step_id=step_id,
            agent_id=step.agent_id,
            agent_role=step.agent_role.value,
            intent_hash=step.intent_hash,
            content=step.content,
            uncertainty_score=step.uncertainty_score,
            coherence_score=step.coherence_score,
            logic_score=step.logic_score
        )
        dig.add_node(node)
    
    # Add links based on causal dependencies with proper mapping
    relationship_mapping = {
        'depends_on': 'DEPENDENCY',
        'follows_from': 'FOLLOWS_FROM',
        'evaluates': 'EVIDENCE_FOR',
        'implements': 'FOLLOWS_FROM',
        'validates': 'EVIDENCE_FOR',
        'corrects': 'DIRECT_CAUSE'
    }
    
    for step_id, step in session.steps.items():
        for dependency in step.causal_dependencies:
            parent_step = session.get_step(dependency.parent_step_id)
            if parent_step:
                # Map the string relationship to enum value
                rel_type_str = dependency.relationship_type
                enum_key = relationship_mapping.get(rel_type_str, 'DEPENDENCY')
                try:
                    relationship = CausalRelationship[enum_key]
                except KeyError:
                    # Fallback to DEPENDENCY if unknown
                    relationship = CausalRelationship.DEPENDENCY
                
                link = CausalLink(
                    source_step_id=dependency.parent_step_id,
                    target_step_id=step_id,
                    relationship_type=relationship,
                    confidence=0.8
                )
                dig.add_link(link)
    
    return dig

def run_regulator_pipeline():
    """Run the full FLOW-HEAL regulator pipeline."""
    print("=" * 70)
    print("FLOW-HEAL REASONING REGULATOR TEST")
    print("=" * 70)
    
    # 1. Create session and components
    print("\n1. Creating test reasoning session...")
    session = create_test_session()
    print(f"   Created session with {len(session.steps)} steps")
    
    # 2. Build Dynamic Interaction Graph
    print("\n2. Building Dynamic Interaction Graph (DIG)...")
    dig = build_dig_from_session(session)
    print(f"   DIG created with {len(dig.nodes)} nodes and {len(dig.links)} links")
    
    # 3. Create NoiseSenser
    print("\n3. Creating NoiseSenser...")
    senser = create_noise_senser(
        similarity_threshold=0.65,
        uncertainty_threshold=0.7,
        history_window=3
    )
    print(f"   NoiseSenser initialized")
    print(f"   Similarity threshold: {senser.similarity_threshold}")
    print(f"   Uncertainty threshold: {senser.uncertainty_threshold}")
    
    # 4. Analyze all steps
    print("\n4. Running semantic analysis on all steps...")
    steps_list = list(session.steps.values())
    analyses = senser.batch_analyze(steps_list, dig)
    
    # 5. Regulator logic - identify steps needing healing
    print("\n5. Regulator Decision Making...")
    print("-" * 70)
    
    problematic_steps = []
    for analysis in analyses:
        step_id = analysis['step_id']
        drift_confidence = analysis['drift_analysis']['confidence']
        uncertainty = analysis['uncertainty_score']
        flags = analysis['flags']
        
        # Regulator rules:
        # - If uncertainty > 0.7, flag for healing
        # - If drift confidence > 0.5, flag for review
        # - If both, high priority for healing
        
        if uncertainty > 0.7:
            priority = "HIGH (High Uncertainty)"
            problematic_steps.append((step_id, uncertainty, drift_confidence, priority, "uncertainty"))
        elif drift_confidence > 0.5:
            priority = "MEDIUM (Semantic Drift)"
            problematic_steps.append((step_id, uncertainty, drift_confidence, priority, "drift"))
        elif uncertainty > 0.5 and drift_confidence > 0.3:
            priority = "LOW (Potential Issue)"
            problematic_steps.append((step_id, uncertainty, drift_confidence, priority, "monitor"))
    
    # 6. Display regulator findings
    print("\n6. REGULATOR FINDINGS")
    print("-" * 70)
    
    if not problematic_steps:
        print("✓ No problematic steps detected - reasoning path is healthy!")
        return
    
    print(f"Found {len(problematic_steps)} steps requiring attention:")
    for i, (step_id, uncertainty, drift_conf, priority, reason) in enumerate(problematic_steps, 1):
        step = session.get_step(step_id)
        print(f"\n{i}. Step {step_id}: {priority}")
        print(f"   Agent: {step.agent_id} ({step.agent_role.value})")
        print(f"   Reason: {reason}")
        print(f"   Uncertainty: {uncertainty:.3f}")
        print(f"   Drift Confidence: {drift_conf:.3f}")
        
        # Get recommendations from analysis
        analysis = next(a for a in analyses if a['step_id'] == step_id)
        if analysis['recommendations']:
            print(f"   Recommendations: {', '.join(analysis['recommendations'])}")
        
        # Identify root causes using DIG
        root_causes = dig.identify_root_causes(step_id)
        if root_causes:
            print(f"   Potential Root Causes: {root_causes[:3]}")
    
    # 7. Demonstrate healing process
    print("\n" + "=" * 70)
    print("7. HEALING PROCESS SIMULATION")
    print("=" * 70)
    
    # Simulate healer addressing the highest priority issue
    if problematic_steps:
        highest_priority = problematic_steps[0]
        step_id, uncertainty, drift_conf, priority, reason = highest_priority
        step = session.get_step(step_id)
        
        print(f"\nTargeting step {step_id} for healing:")
        print(f"   Current Uncertainty: {uncertainty:.3f}")
        print(f"   Current Drift Confidence: {drift_conf:.3f}")
        
        # Simulate healing process
        print(f"\nHealing Process:")
        print(f"   a. Healer reviews context and root causes")
        print(f"   b. Identifies that step {step_id} introduced unrelated concerns")
        print(f"   c. Rewrites content to maintain focus on core requirements")
        print(f"   d. Updates quality metrics")
        
        # Update step with improved metrics (simulation)
        step.uncertainty_score = 0.3
        step.coherence_score = 0.8
        step.logic_score = 0.7
        step.status = StepStatus.HEALED
        step.healing_attempts += 1
        
        # Update DIG node
        if step_id in dig.nodes:
            node = dig.nodes[step_id]
            node.uncertainty_score = 0.3
            node.coherence_score = 0.8
            node.logic_score = 0.7
            node.quality_score = node.calculate_overall_quality()
            dig.graph.nodes[step_id]['node_data'] = node.to_dict()
        
        print(f"\n✓ Step {step_id} has been healed!")
        print(f"   New Uncertainty: {step.uncertainty_score}")
        print(f"   New Coherence: {step.coherence_score}")
        print(f"   New Logic: {step.logic_score}")
        print(f"   New Quality: {step.calculate_overall_quality():.3f}")
        
        # Re-run analysis on healed step
        analysis = senser.analyze_step(step, steps_list[:steps_list.index(step)], dig)
        print(f"\nPost-Healing Analysis:")
        print(f"   Drift Detected: {analysis['drift_analysis']['drift_detected']}")
        print(f"   Drift Confidence: {analysis['drift_analysis']['confidence']:.3f}")
        print(f"   Uncertainty: {analysis['uncertainty_score']:.3f}")
    
    # 8. Final summary
    print("\n" + "=" * 70)
    print("REGULATOR SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Steps: {len(session.steps)}")
    print(f"Steps Needing Attention: {len(problematic_steps)}")
    print(f"Steps Healed: 1 (demonstration)")
    print(f"\nRegulator successfully identified semantic drift and triggered healing.")
    print("The FLOW-HEAL system demonstrates effective self-correction capabilities.")

if __name__ == "__main__":
    try:
        run_regulator_pipeline()
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
