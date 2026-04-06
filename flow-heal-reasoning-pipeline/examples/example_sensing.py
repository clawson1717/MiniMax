from typing import List, Dict, Any, Tuple
"""
Example script demonstrating the integration of NoiseSenser with the FLOW-HEAL pipeline.
This shows how to use the Semantic Noise Senser to detect drift and uncertainty in a
multi-agent reasoning session.
"""

from typing import List, Dict, Any
from datetime import datetime
import json

# Import FLOW-HEAL components
from src.payload import ReasoningStep, ReasoningSession, AgentRole, StepStatus
from src.graph import ReasoningDIG, ReasoningNode, CausalLink, CausalRelationship
from src.sensing import NoiseSenser, create_noise_senser

def create_sample_reasoning_session() -> Tuple[ReasoningSession, List[ReasoningStep], NoiseSenser]:
    """
    Create a sample reasoning session with multiple steps.
    
    Returns:
        Tuple of (session, steps, senser)
    """
    # Create a NoiseSenser instance
    senser = create_noise_senser(
        similarity_threshold=0.65,
        uncertainty_threshold=0.7,
        history_window=2
    )
    
    # Create a reasoning session
    session = ReasoningSession(
        session_id="demo_session_001",
        conversation_id="demo_conversation_001"
    )
    
    # Create sample reasoning steps
    steps = []
    
    # Step 1: Planner creates initial plan
    step1 = ReasoningStep(
        step_id="step_001",
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="plan_initial",
        content="I will create an initial project plan for the FLOW-HEAL system based on the three source papers.",
        uncertainty_score=0.2,
        coherence_score=0.9,
        logic_score=0.8,
        causal_dependencies=[
            CausalDependency(parent_step_id="user_prompt", relationship_type="follows_from")
        ]
    )
    steps.append(step1)
    session.add_step(step1)
    
    # Step 2: Researcher gathers requirements
    step2 = ReasoningStep(
        step_id="step_002",
        agent_id="researcher_agent",
        agent_role=AgentRole.RESEARCHER,
        intent_hash="research_requirements",
        content="I will analyze the DenoiseFlow, DIG to Heal, and LOGIGEN papers to extract key requirements for the FLOW-HEAL architecture.",
        uncertainty_score=0.3,
        coherence_score=0.8,
        logic_score=0.7,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_001", relationship_type="depends_on")
        ]
    )
    steps.append(step2)
    session.add_step(step2)
    
    # Step 3: Thinker designs architecture
    step3 = ReasoningStep(
        step_id="step_003",
        agent_id="thinker_agent",
        agent_role=AgentRole.THINKER,
        intent_hash="design_architecture",
        content="Based on the research findings, I will design a modular architecture for FLOW-HEAL with separate components for sensing, regulating, healing, and verification.",
        uncertainty_score=0.4,
        coherence_score=0.7,
        logic_score=0.6,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_002", relationship_type="follows_from")
        ]
    )
    steps.append(step3)
    session.add_step(step3)
    
    # Step 4: Critic evaluates the design (potential drift point)
    step4 = ReasoningStep(
        step_id="step_004",
        agent_id="critic_agent",
        agent_role=AgentRole.CRITIC,
        intent_hash="evaluate_design",
        content="I will critically evaluate the proposed architecture for potential flaws, edge cases, and implementation challenges.",
        uncertainty_score=0.6,
        coherence_score=0.6,
        logic_score=0.5,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="evaluates")
        ]
    )
    steps.append(step4)
    session.add_step(step4)
    
    # Step 5: Planner creates implementation plan (should be aligned with step 3)
    step5 = ReasoningStep(
        step_id="step_005",
        agent_id="planner_agent",
        agent_role=AgentRole.PLANNER,
        intent_hash="create_implementation_plan",
        content="I will break down the architecture into a 12-step implementation plan with clear deliverables and milestones.",
        uncertainty_score=0.2,
        coherence_score=0.8,
        logic_score=0.9,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_003", relationship_type="implements")
        ]
    )
    steps.append(step5)
    session.add_step(step5)
    
    # Step 6: Validator checks logical consistency (potential drift)
    step6 = ReasoningStep(
        step_id="step_006",
        agent_id="validator_agent",
        agent_role=AgentRole.VALIDATOR,
        intent_hash="validate_logic",
        content="I will verify that each step of the implementation plan logically follows from the architectural design and requirements.",
        uncertainty_score=0.5,
        coherence_score=0.5,
        logic_score=0.4,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_005", relationship_type="validates")
        ]
    )
    steps.append(step6)
    session.add_step(step6)
    
    # Step 7: Healer identifies and fixes issues (healing step)
    step7 = ReasoningStep(
        step_id="step_007",
        agent_id="healer_agent",
        agent_role=AgentRole.HEALER,
        intent_hash="heal_drift",
        content="I will identify semantic drift in the reasoning path and propose corrections to maintain logical consistency.",
        uncertainty_score=0.7,
        coherence_score=0.4,
        logic_score=0.3,
        causal_dependencies=[
            CausalDependency(parent_step_id="step_006", relationship_type="corrects")
        ]
    )
    steps.append(step7)
    session.add_step(step7)
    
    return session, steps, senser


def build_dynamic_interaction_graph(session: ReasoningSession) -> ReasoningDIG:
    """
    Build a Dynamic Interaction Graph from a reasoning session.
    
    Args:
        session: Reasoning session with multiple steps
        
    Returns:
        Constructed ReasoningDIG
    """
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
    
    # Add links based on causal dependencies
    for step_id, step in session.steps.items():
        for dependency in step.causal_dependencies:
            # Find the parent step
            parent_step = session.get_step(dependency.parent_step_id)
            if parent_step:
                link = CausalLink(
                    source_step_id=dependency.parent_step_id,
                    target_step_id=step_id,
                    relationship_type=CausalRelationship[dependency.relationship_type.upper()],
                    confidence=0.8  # Default confidence
                )
                dig.add_link(link)
    
    return dig


def run_noise_senser_analysis(
    steps: List[ReasoningStep],
    dig: ReasoningDIG,
    senser: NoiseSenser
) -> List[Dict[str, Any]]:
    """
    Run comprehensive noise senser analysis on all steps.
    
    Args:
        steps: List of reasoning steps
        dig: Dynamic Interaction Graph
        senser: NoiseSenser instance
        
    Returns:
        List of analysis results for each step
    """
    print("Running NoiseSenser analysis...")
    print("-" * 60)
    
    # Batch analyze all steps
    analyses = senser.batch_analyze(steps, dig)
    
    # Display summary
    print(f"Analyzed {len(analyses)} reasoning steps")
    print(f"Steps with semantic drift: {sum(1 for a in analyses if a['drift_analysis']['drift_detected'])}")
    print(f"Steps with high uncertainty (>0.7): {sum(1 for a in analyses if a['uncertainty_score'] > 0.7)}")
    print("-" * 60)
    
    return analyses


def display_analysis_report(analyses: List[Dict[str, Any]]) -> None:
    """Display a formatted analysis report."""
    print("\nDetailed Analysis Report")
    print("=" * 60)
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\nStep {i}: {analysis['step_id']}")
        print(f"Agent: {analysis['agent_id']} ({analysis['agent_role']})")
        
        # Drift analysis
        drift = analysis['drift_analysis']
        print(f"  ✓ Drift Detected: {drift['drift_detected']}")
        print(f"  ✓ Drift Confidence: {drift['confidence']:.3f}")
        print(f"  ✓ Semantic Similarity to Previous: {drift['similarity_scores'].get('previous', 'N/A'):.3f}")
        
        # Uncertainty
        print(f"  ⚠ Uncertainty Score: {analysis['uncertainty_score']:.3f}")
        
        # Quality metrics
        print(f"  ★ Quality Metrics:")
        print(f"    - Built-in Quality: {analysis['quality_metrics']['built_in_quality']:.3f}")
        print(f"    - Drift Impact: {analysis['quality_metrics']['drift_confidence']:.3f}")
        
        # Flags and recommendations
        if analysis['flags']:
            print(f"  ⚑ Flags: {', '.join(analysis['flags'])}")
        
        if analysis['recommendations']:
            print(f"  → Recommendations: {', '.join(analysis['recommendations'])}")
    
    print("\n" + "=" * 60)


def main():
    """Main execution demonstrating FLOW-HEAL pipeline with NoiseSenser."""
    print("FLOW-HEAL Semantic Noise Senser Demonstration")
    print("=" * 60)
    
    # Step 1: Create sample session and components
    print("1. Creating sample reasoning session...")
    session, steps, senser = create_sample_reasoning_session()
    print(f"   Created {len(steps)} reasoning steps")
    
    # Step 2: Build Dynamic Interaction Graph
    print("\n2. Building Dynamic Interaction Graph (DIG)...")
    dig = build_dynamic_interaction_graph(session)
    print(f"   DIG created with {len(dig.nodes)} nodes and {len(dig.links)} links")
    
    # Step 3: Run NoiseSenser analysis
    print("\n3. Running semantic noise analysis...")
    analyses = run_noise_senser_analysis(steps, dig, senser)
    
    # Step 4: Display report
    display_analysis_report(analyses)
    
    # Step 5: Show how the results would be used by the regulator
    print("\n4. Regulator Integration Example")
    print("=" * 60)
    print("The FlowRegulator would use these results to:")
    print("  - Identify steps with high drift confidence (>0.5)")
    print("  - Flag steps with high uncertainty (>0.7) for healing")
    print("  - Target the root cause using the DIG's causal structure")
    
    # Identify steps that need healing attention
    problematic_steps = [
        (a['step_id'], a['uncertainty_score'], a['drift_analysis']['confidence'])
        for a in analyses
        if a['uncertainty_score'] > 0.7 or a['drift_analysis']['confidence'] > 0.5
    ]
    
    if problematic_steps:
        print("\nRecommended Healing Targets:")
        for step_id, uncertainty, drift_confidence in problematic_steps:
            print(f"  - {step_id}: Uncertainty={uncertainty:.2f}, Drift={drift_confidence:.2f}")
    else:
        print("\nNo immediate healing targets - reasoning path is healthy! ✓")
    
    # Step 6: Save results to JSON
    print("\n5. Saving Results")
    print("=" * 60)
    
    # Save analysis to file
    output_file = "noise_senser_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'session_id': session.session_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'step_analyses': analyses,
            'summary': {
                'total_steps': len(analyses),
                'drift_detected': sum(1 for a in analyses if a['drift_analysis']['drift_detected']),
                'high_uncertainty': sum(1 for a in analyses if a['uncertainty_score'] > 0.7)
            }
        }, f, indent=2, default=str)
    
    print(f"✓ Analysis saved to {output_file}")
    print("✓ DIG saved to noise_senser_dig.json")  # Would need to save DIG separately
    
    # Save DIG
    dig_file = "noise_senser_dig.json"
    dig.save_to_json(dig_file)
    print(f"✓ DIG saved to {dig_file}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete! The NoiseSenser is ready for integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()