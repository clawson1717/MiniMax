import os
import json
from typing import Dict, Any, List
from pathlib import Path
from src.skills import ScientificSkill

class ExpertSkillGenerator:
    """
    Converts successful discovery trajectories into reusable Expert Skills.
    """

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def generate_skill(
        self, 
        trajectory: Dict[str, Any], 
        metrics: Dict[str, Any],
        skill_name: str,
        domain: str = "Physics"
    ) -> ScientificSkill:
        """
        Extracts the procedural core from a trajectory and creates a ScientificSkill.

        Args:
            trajectory: The reasoning trace from KeplerReasoningAgent.
            metrics: Success metrics (e.g., error rates, consistency).
            skill_name: Name for the new skill.
            domain: Scientific domain.

        Returns:
            A ScientificSkill instance.
        """
        # Extract procedural steps from the analysis and hypotheses
        procedural_steps = []
        
        analysis = trajectory.get("analysis", "")
        if analysis:
            procedural_steps.append(f"Analyze physical constraints: {analysis[:200]}...")

        hypotheses = trajectory.get("hypotheses", [])
        for i, h in enumerate(hypotheses):
            procedural_steps.append(f"Evaluate hypothesis {i+1}: {h}")

        proposed = trajectory.get("proposed_expressions", [])
        if proposed:
            procedural_steps.append(f"Test symbolic expressions: {', '.join(proposed)}")

        # Create the skill object
        skill = ScientificSkill(
            name=skill_name,
            domain=domain,
            description=f"Expert skill derived from successful discovery: {skill_name}",
            preconditions=[
                "Availability of physical constraint data",
                "Numerical dataset for verification"
            ],
            procedural_steps=procedural_steps,
            expected_effects=[
                f"Discovery of governing equation with metrics: {json.dumps(metrics)}"
            ]
        )

        self.save_skill(skill)
        return skill

    def save_skill(self, skill: ScientificSkill) -> Path:
        """Saves the skill to the skills directory."""
        filename = skill.name.lower().replace(" ", "_") + ".json"
        filepath = self.skills_dir / filename
        
        with open(filepath, "w") as f:
            f.write(skill.to_json())
        
        return filepath
