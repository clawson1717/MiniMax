#!/usr/bin/env python3
"""
FLOW-HEAL ReasoningRegulator Integration Example

This example demonstrates the complete usage of the ReasoningRegulator module
in a realistic multi-agent reasoning pipeline scenario.
"""

from flow_heal.src.regulating import create_reasoning_regulator
from flow_heal.src.payload import ReasoningStep, ReasoningSession, AgentRole, CausalDependency
from flow_heal.src.graph import ReasoningDIG, ReasoningNode, CausalLink, CausalRelationship
from flow_heal.src.sensing import create_noise_senser
from datetime import datetime

def main():
    """
    Main integration example showing the full ReasoningRegulator workflow.
    """
    print("=" * 70)
    print("FLOW-HEAL REASONING REGULATOR - INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # =========================================================================
    # 1. INITIALIZE THE REGULATOR
    # =========================================================================
    print("\n1. Initializing ReasoningRegulator...")
    
    # Create regulator with default configuration
    regulator = create_reasoning_regulator()
    
    print(f"   - Regulator created with config: {regulator.config}")
    print(f"   - NoiseSenser: {regulator.noise_senser}")
    print(f"   - History capacity: {len(regulator.previous_steps)} (empty)")
    
    # =========================================================================
    # 2. INITIALIZE A REGULATION SESSION
    # =========================================================================
    print("\n2. Initializing regulation session...")
    
    regulator.initialize_session(
        session_id="demo_session_2026",
        conversation_id="integration_example_001"
    )
    
    print(f"   - Session initialized: {regulator.session_state.session_id}")
    print(f"   - DIG created: {regulator.reasoning_dig}")
    print(f"   - Session state: {regulator.session_state}")
    
    # =========================================================================
    # 3. CREATE SAMPLE REASONING STEPS
    # =========================================================================
    print("\n3. Creating sample reasoning steps for a complex problem...")
    
    # Step 1: Initial planning (high quality)
    step1 = ReasoningStep(
        step_id="step_001",
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for developing a FLOW-HEAL system to improve reasoning reliability.",
        uncertainty_score=0.15,
        coherence_score=0.92,
        logic_score=0.88,
        causal_dependencies=[
            CausalDependency(parent_step_id="user_query_001", relationship_type="responds_to")
        ]
    )
    
    # Step 2: Research requirements (moderate quality)
    step2 = ReasoningStep(
        step_id="step_002",
        agent_id="researcher_agent",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="Based on the DenoiseFlow and DIG to Heal papers, I'll identify key requirements: uncertainty quantification, causal tracking, and targeted healing mechanisms.",
        uncertainty_score=0.35,
        coherence_score=0.78,
        logic_score=0.72,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_001", relationship_type="builds_on")
        ]
    )
    
    # Step 3: Design architecture (good quality)
    step3 = ReasoningStep(
        step_id="step_003",
        agent_id="architect_agent",
        agent_role=AgentRole.ARCHITECT,
        intent_hash="design_architecture",
        content="I'll design a modular architecture with three core components: Sensing (for drift detection), Regulating (for root cause analysis), and Correcting (for healing interventions).",
        uncertainty_score=0.25,
        coherence_score=0.85,
        logic_score=0.82,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="informs")
        ]
    )
    
    # Step 4: Detailed design (moderate quality, starting to show issues)
    step4 = ReasoningStep(
        step_id="step_004",
        agent_id="designer_agent",
        agent_role=AgentRole.DESIGNER,
        intent_hash="design_detailed",
        content="I'll design the FlowRegulator class with methods for analyzing steps, identifying root causes, and generating recommendations. It should work with ReasoningStep objects and ReasoningDIG.",
        uncertainty_score=0.45,
        coherence_score=0.72,
        logic_score=0.68,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="implements")
        ]
    )
    
    # Step 5: Implementation details (lower quality, potential issues)
    step5 = ReasoningStep(
        step_id="step_005",
        agent_id="implementer_agent",
        agent_role=AgentRole.IMPLEMENTER,
        intent_hash="implementation_details",
        content="I need to implement the ReasoningRegulator class. It should have methods like analyze_and_regulate, _identify_root_causes, and _generate_recommendations. I'm not entirely sure about the scoring algorithm though.",
        uncertainty_score=0.65,
        coherence_score=0.58,
        logic_score=0.52,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_004", relationship_type="refines")
        ]
    )
    
    # Step 6: Problem step (low quality, clear issues)
    step6 = ReasoningStep(
        step_id="step_006",
        agent_id="confused_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="problematic_step",
        content="I'm really struggling with this. The implementation isn't working and I'm not sure what to do. Maybe I should just give up or try a different approach entirely. This is too complicated.",
        uncertainty_score=0.88,
        coherence_score=0.35,
        logic_score=0.21,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_005", relationship_type="follows_from")
        ]
    )
    
    print(f"   - Created 6 reasoning steps with varying quality levels")
    print(f"   - Steps range from high quality (step1: {step1.uncertainty_score}) to low quality (step6: {step6.uncertainty_score})")
    
    # =========================================================================
    # 4. PROCESS STEPS THROUGH THE REGULATOR
    # =========================================================================
    print("\n4. Processing reasoning steps through the regulator...")
    
    steps = [step1, step2, step3, step4, step5, step6]
    
    for i, step in enumerate(steps, 1):
        print(f"\n   Processing step {i}: {step.step_id}")
        print(f"     Agent: {step.agent_role.value}")
        print(f"     Uncertainty: {step.uncertainty_score:.2f}")
        print(f"     Coherence: {step.coherence_score:.2f}")
        print(f"     Logic: {step.logic_score:.2f}")
        print(f"     Content: {step.content[:70]}...")
        
        # Analyze and regulate the step
        result = regulator.analyze_and_regulate(step)
        
        print(f"     Regulation Results:")
        print(f"     - Issues detected: {result.issues_detected if result.issues_detected else 'None'}")
        print(f"     - Root causes: {len(result.root_causes)} identified")
        print(f"     - Healing recommendations: {len(result.healing_recommendations)}")
        print(f"     - Session quality: {result.quality_metrics.get('session_quality', 0):.2f}")
        print(f"     - Healing urgency: {result.quality_metrics.get('healing_urgency', 0):.2f}")
        
        # Update regulator history
        regulator.update_history(step)
        
        # Show some recommendations
        if result.healing_recommendations:
            print(f"     Top recommendations:")
            for j, rec in enumerate(result.healing_recommendations[:2], 1):
                print(f"       {j}. {rec['description']} (confidence: {rec['confidence']:.2f}, priority: {rec['priority']})")
    
    # =========================================================================
    # 5. FINAL REGULATION REPORT
    # =========================================================================
    print("\n5. Generating final regulation report...")
    
    final_result = regulator.get_last_regulation()
    if final_result:
        print("\n   Markdown Report:")
        print("=" * 70)
        print(regulator.visualize_regulation(final_result, output_format='markdown'))
        print("=" * 70)
        
        print("\n   Text Report:")
        print("=" * 70)
        print(regulator.visualize_regulation(final_result, output_format='text'))
        print("=" * 70)
    else:
        print("   No regulation results available")
    
    # =========================================================================
    # 6. VISUALIZE THE DYNAMIC INTERACTION GRAPH (DIG)
    # =========================================================================
    print("\n6. Visualizing the Dynamic Interaction Graph (DIG) with analysis highlights...")
    
    dig_viz = regulator.visualize_dig_with_analysis(final_result, highlight_root_causes=True)
    
    print(f"   DIG visualization generated ({len(dig_viz)} characters)")
    print(f"   Sample: {dig_viz[:300]}...")
    
    # Save the DIG visualization to a file for external viewing
    with open("dig_visualization.dot", "w") as f:
        f.write(dig_viz)
    print(f"   Saved DIG visualization to: dig_visualization.dot")
    print(f"   To visualize: dot -Tpng dig_visualization.dot -o dig_visualization.png")
    
    # =========================================================================
    # 7. DEMONSTRATE FACTORY FUNCTION
    # =========================================================================
    print("\n7. Demonstrating factory function with custom configuration...")
    
    # Create regulator with custom configuration (more sensitive)
    sensitive_config = RegulatorConfig(
        drift_threshold=0.5,  # More sensitive to drift
        uncertainty_threshold=0.6,  # Less tolerant of uncertainty
        quality_threshold=0.6  # Higher quality expectations
    )
    
    sensitive_regulator = create_reasoning_regulator(config=sensitive_config)
    
    print(f"   - Created sensitive regulator with drift_threshold={sensitive_config.drift_threshold}")
    print(f"   - This regulator will flag issues earlier than default")
    
    # =========================================================================
    # 8. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("  - ReasoningRegulator successfully processed 6 reasoning steps")
    print("  - Detected quality degradation from step 4 onwards")
    print("  - Identified root causes and generated targeted recommendations")
    print("  - Visualized regulation results in multiple formats")
    print("  - Demonstrated DIG visualization with root cause highlighting")
    print("  - Showed factory function with custom configuration")
    print("\nKey Takeaways:")
    print("  - The regulator provides actionable insights for the healing agent")
    print("  - Quality metrics help monitor session health in real-time")
    print("  - Root cause analysis enables targeted interventions")
    print("  - Visualization aids in understanding reasoning flow and issues")
    print("=" * 70)


if __name__ == "__main__":
    # Run the integration example
    main()
